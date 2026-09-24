"""End-to-end tests: sample video -> CLI render -> outputs; plus API tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
SAMPLE = APP / "sample"


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "sample.mp4"
    subprocess.run(["bash", str(SAMPLE / "make_sample_video.sh"), str(out)], check=True)
    return out


def test_cli_render(sample_video, tmp_path):
    out = tmp_path / "reel"
    r = subprocess.run(
        [
            sys.executable, "-m", "highlights.app.render",
            "--video", str(sample_video),
            "--candidates", str(SAMPLE / "candidates_short.json"),
            "--out", str(out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    clips = sorted((out / "clips").glob("*.mp4"))
    # 5 events minus the cross_validation=rejected one
    assert len(clips) == 4
    assert (out / "reel.mp4").is_file()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["n_candidates"] == 5
    assert stats["rendered"]["n_clips"] == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest) == 4
    for m in manifest:
        r2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", m["path"]],
            capture_output=True, text=True, check=True,
        )
        dur = json.loads(r2.stdout)["format"]["duration"]
        # stream copy snaps to keyframes; allow generous slack vs expected window
        expected = m["clip_end"] - m["clip_start"]
        assert abs(float(dur) - expected) < 8.0


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
    r = client.post("/api/candidates/load", json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200, r.text
    return r.json()


def test_api_flow(client, sample_video):
    cands = _setup(client, sample_video)
    assert len(cands) == 5
    rejected = [c for c in cands if c["status"] == "rejected"]
    assert len(rejected) == 1  # cross_validation == "rejected"

    goal = next(c for c in cands if c["type"] == "goal")
    assert goal["clip_start"] == pytest.approx(15.0)
    assert goal["clip_end"] == pytest.approx(25.0)

    r = client.patch(f"/api/candidates/{goal['id']}", json={"status": "confirmed", "clip_start": 16.0})
    assert r.status_code == 200
    assert r.json()["clip_start"] == 16.0
    assert r.json()["status"] == "confirmed"

    r = client.patch(f"/api/candidates/{goal['id']}", json={"clip_start": 30.0, "clip_end": 28.0})
    assert r.status_code == 422

    r = client.post(f"/api/candidates/{goal['id']}/reset")
    assert r.json()["clip_start"] == pytest.approx(15.0)

    r = client.get(f"/api/candidates/{goal['id']}/thumb.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    r = client.get("/api/stats")
    s = r.json()
    assert s["n_candidates"] == 5
    assert s["counts_by_type"]["goal"]["confirmed"] == 1


def test_api_render_job(client, sample_video):
    _setup(client, sample_video)
    r = client.post("/api/render", json={"overlay": False})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(600):
        j = client.get(f"/api/render/{job_id}").json()
        if j["state"] in ("done", "error"):
            break
    assert j["state"] == "done", j
    assert j["reel_url"].endswith("reel.mp4")
    assert len(j["clips"]) == 4


def test_video_404(client):
    r = client.post("/api/video", json={"path": "/nonexistent.mp4"})
    assert r.status_code == 404


def test_load_returns_confidence_order(client, sample_video):
    cands = _setup(client, sample_video)
    confs = [c["confidence"] for c in cands]
    assert confs == sorted(confs, reverse=True)
    assert cands[0]["rank"] == 1


def test_proxy_file_without_flag_not_ready(client, sample_video, tmp_path):
    _setup(client, sample_video)
    wd = tmp_path / "wd"
    (wd / "proxy.mp4").write_bytes(b"partial")  # stale/partial file, no flag
    r = client.get("/api/video").json()
    assert r["proxy_ready"] is False
    assert not (wd / "proxy.mp4").exists()  # stale file deleted
    s = client.get("/api/video/proxy/status").json()
    assert s["ready"] is False
    # also works via proxy endpoint: should start a build, not claim ready
    r = client.post("/api/video/proxy")
    assert r.json()["status"] == "started"


def test_register_new_video_invalidates(client, sample_video, tmp_path):
    _setup(client, sample_video)
    import highlights.app.backend.main as m

    wd = tmp_path / "wd"
    # simulate completed proxy + cached thumbs
    (wd / "proxy.mp4").write_bytes(b"done")
    (wd / "proxy.part.mp4").write_bytes(b"part")
    m.STORE.set_proxy_complete(str(sample_video.resolve()))
    td = m.STORE.thumb_dir()
    (td / "c001_20.0.jpg").write_bytes(b"jpg")
    assert client.get("/api/video").json()["proxy_ready"] is True

    # register a *different* video
    other = tmp_path / "other.mp4"
    other.write_bytes(sample_video.read_bytes())
    r = client.post("/api/video", json={"path": str(other)})
    assert r.status_code == 200
    assert not (wd / "proxy.mp4").exists()
    assert not (wd / "proxy.part.mp4").exists()
    assert not m.STORE.proxy_complete
    assert not (td / "c001_20.0.jpg").exists()  # thumbs cleared
    assert client.get("/api/video").json()["proxy_ready"] is False


def test_thumb_clamp_and_404(client, sample_video, monkeypatch):
    cands = _setup(client, sample_video)
    goal = next(c for c in cands if c["type"] == "goal")

    # t beyond duration is clamped to duration-0.1, not a 500
    r = client.get(f"/api/candidates/{goal['id']}/thumb.jpg?t=99999")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    # ffmpeg failure / no output -> 404, not 500
    import highlights.app.backend.main as m

    def _boom(src, t, out):
        raise RuntimeError("boom")

    monkeypatch.setattr(m.fx, "thumbnail", _boom)
    r = client.get(f"/api/candidates/{goal['id']}/thumb.jpg?t=1.5")
    assert r.status_code == 404

    def _nofile(src, t, out):
        return None

    monkeypatch.setattr(m.fx, "thumbnail", _nofile)
    r = client.get(f"/api/candidates/{goal['id']}/thumb.jpg?t=2.5")
    assert r.status_code == 404
