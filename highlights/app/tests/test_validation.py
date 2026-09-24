"""Backend validation invariants: atomic patch, render 422, window rules."""

import json
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
SAMPLE = APP / "sample"


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "sample.mp4"
    subprocess.run(["bash", str(SAMPLE / "make_sample_video.sh"), str(out)],
                   check=True)
    return out


@pytest.fixture(scope="module")
def short_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc2=duration=6:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True)
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_WORKDIR", str(tmp_path / "wd"))
    from fastapi.testclient import TestClient

    from highlights.app.backend import store
    from highlights.app.backend.main import app

    store.STORE = store.ProjectStore()
    import highlights.app.backend.main as m

    m.STORE = store.STORE
    return TestClient(app)


def _setup(client, video):
    r = client.post("/api/video", json={"path": str(video)})
    assert r.status_code == 200, r.text
    r = client.post("/api/candidates/load",
                    json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200, r.text
    return r.json()


def _get(client, cid):
    return next(c for c in client.get("/api/candidates").json()
                if c["id"] == cid)


def test_reversed_patch_atomic(client, sample_video):
    cands = _setup(client, sample_video)
    c = cands[0]
    orig = {k: c[k] for k in ("clip_start", "clip_end", "status", "type")}
    r = client.patch(f"/api/candidates/{c['id']}",
                     json={"clip_start": 30.0, "clip_end": 28.0})
    assert r.status_code == 422
    after = _get(client, c["id"])
    for k, v in orig.items():
        assert after[k] == v
    # persisted state on disk is unchanged too
    import highlights.app.backend.main as m
    saved = json.loads(m.STORE.state_path.read_text())
    sc = next(x for x in saved["candidates"] if x["id"] == c["id"])
    for k, v in orig.items():
        assert sc[k] == v


def test_negative_and_beyond_duration_patches(client, sample_video):
    cands = _setup(client, sample_video)
    c = cands[0]
    duration = client.get("/api/video").json()["duration_s"]

    r = client.patch(f"/api/candidates/{c['id']}",
                     json={"clip_start": -2.0})
    assert r.status_code == 422
    assert _get(client, c["id"])["clip_start"] == c["clip_start"]

    r = client.patch(f"/api/candidates/{c['id']}",
                     json={"clip_end": duration + 10})
    assert r.status_code == 422
    assert _get(client, c["id"])["clip_end"] == c["clip_end"]

    # status change combined with invalid window: nothing applied
    r = client.patch(f"/api/candidates/{c['id']}",
                     json={"status": "confirmed", "clip_start": 30.0,
                           "clip_end": 28.0})
    assert r.status_code == 422
    assert _get(client, c["id"])["status"] == c["status"]


def test_invalid_render_is_422_no_job(client, sample_video):
    cands = _setup(client, sample_video)
    bad = cands[0]
    # corrupt the stored window directly (patch endpoint protects it)
    import highlights.app.backend.main as m
    stored = m.STORE.get(bad["id"])
    stored.clip_start, stored.clip_end = 30.0, 28.0
    n_jobs = len(m._jobs)
    r = client.post("/api/render",
                    json={"ids": [bad["id"]], "overlay": False})
    assert r.status_code == 422
    assert bad["id"] in r.text
    # no render job was created
    assert len(m._jobs) == n_jobs


def test_render_reel_rejects_invalid(client, sample_video, tmp_path):
    from highlights.app.backend import ffmpeg as fx
    with pytest.raises((ValueError, RuntimeError)):
        fx.render_reel(
            str(sample_video),
            [{"id": "bad1", "t": 10.0, "type": "shot",
              "clip_start": 30.0, "clip_end": 28.0}],
            tmp_path / "out", overlay=False)


def test_cut_clip_high_quality_args(client, monkeypatch):
    from highlights.app.backend import ffmpeg as fx
    calls = []

    class R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(fx, "_run",
                        lambda cmd, **kw: calls.append(cmd) or R())
    monkeypatch.setattr(fx, "drawtext_supported", lambda: True)
    monkeypatch.setattr(fx, "find_font", lambda: "/tmp/f.ttf")
    fx.cut_clip("/tmp/v.mp4", 1.0, 2.0, "/tmp/o.mp4",
                overlay_label="GOAL 00:20")
    cmd = calls[-1]
    assert cmd[cmd.index("-crf") + 1] == "14"
    assert cmd[cmd.index("-preset") + 1] == "slow"
    assert "yuv420p" in cmd
    assert "256k" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale" not in vf
    # reencode-only path: no filter at all
    calls.clear()
    fx.cut_clip("/tmp/v.mp4", 1.0, 2.0, "/tmp/o.mp4", reencode=True)
    assert "-vf" not in calls[-1]


def test_source_switch_revalidates_windows(client, sample_video, short_video):
    cands = _setup(client, sample_video)
    goal = next(c for c in cands if c["type"] == "goal")
    assert goal["clip_end"] > 6.0  # longer than the short video
    r = client.post("/api/video", json={"path": str(short_video)})
    assert r.status_code == 200
    dur = r.json()["duration_s"]
    for c in client.get("/api/candidates").json():
        assert 0 <= c["clip_start"] < c["clip_end"] <= dur
    # candidate beyond duration gets a clamped end-of-video window
    g = _get(client, goal["id"])
    assert g["clip_end"] == pytest.approx(dur)
    assert g["t"] == goal["t"]  # event time preserved
