"""End-to-end tests: sample video -> CLI render -> outputs; plus API tests."""

import json
import subprocess
import sys

import pytest
from conftest import REPO, SAMPLE, new_project, scoped


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


def _setup(client, video):
    pid = new_project(client, video)
    r = client.post(scoped(pid, "/video"), json={"path": str(video)})
    assert r.status_code == 200, r.text
    r = client.post(scoped(pid, "/candidates/load"), json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200, r.text
    return pid, r.json()


def test_api_flow(client, sample_video):
    pid, cands = _setup(client, sample_video)
    assert len(cands) == 5
    rejected = [c for c in cands if c["status"] == "rejected"]
    assert len(rejected) == 1  # cross_validation == "rejected"

    goal = next(c for c in cands if c["type"] == "goal")
    assert goal["clip_start"] == pytest.approx(15.0)
    assert goal["clip_end"] == pytest.approx(25.0)

    r = client.patch(scoped(pid, f"/candidates/{goal['id']}"), json={"status": "confirmed", "clip_start": 16.0})
    assert r.status_code == 200
    assert r.json()["clip_start"] == 16.0
    assert r.json()["status"] == "confirmed"

    r = client.patch(scoped(pid, f"/candidates/{goal['id']}"), json={"clip_start": 30.0, "clip_end": 28.0})
    assert r.status_code == 422

    r = client.post(scoped(pid, f"/candidates/{goal['id']}/reset"))
    assert r.json()["clip_start"] == pytest.approx(15.0)

    r = client.get(scoped(pid, f"/candidates/{goal['id']}/thumb.jpg"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    r = client.get(scoped(pid, ""))
    s = r.json()
    assert s["n_candidates"] == 5
    assert s["n_confirmed"] == 1


def test_api_render_job(client, sample_video):
    pid, _ = _setup(client, sample_video)
    r = client.post(scoped(pid, "/render"), json={"overlay": False})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(600):
        j = client.get(scoped(pid, f"/render/{job_id}")).json()
        if j["state"] in ("done", "error"):
            break
    assert j["state"] == "done", j
    assert j["reel_url"].endswith("reel.mp4")
    assert len(j["clips"]) == 4


def test_video_404(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.post(scoped(pid, "/video"), json={"path": "/nonexistent.mp4"})
    assert r.status_code == 404


def test_load_returns_confidence_order(client, sample_video):
    _pid, cands = _setup(client, sample_video)
    confs = [c["confidence"] for c in cands]
    assert confs == sorted(confs, reverse=True)
    assert cands[0]["rank"] == 1


def test_proxy_file_without_flag_not_ready(client, sample_video, tmp_path):
    pid, _ = _setup(client, sample_video)
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    (proot / "proxy.mp4").write_bytes(b"partial")  # stale/partial file, no flag
    r = client.get(scoped(pid, "/video")).json()
    assert r["proxy_ready"] is False
    assert not (proot / "proxy.mp4").exists()  # stale file deleted
    s = client.get(scoped(pid, "/video/proxy/status")).json()
    assert s["ready"] is False
    # also works via proxy endpoint: should start a build, not claim ready
    r = client.post(scoped(pid, "/video/proxy"))
    assert r.json()["status"] == "started"


def test_register_new_video_invalidates(client, sample_video, tmp_path):
    pid, _ = _setup(client, sample_video)
    import highlights.app.backend.main as m

    store = m.get_registry().get(pid)
    proot = store.root
    # simulate completed proxy + cached thumbs
    (proot / "proxy.mp4").write_bytes(b"done")
    (proot / "proxy.part.mp4").write_bytes(b"part")
    store.set_proxy_complete(str(sample_video.resolve()))
    td = store.thumb_dir()
    (td / "c001_20.0.jpg").write_bytes(b"jpg")
    assert client.get(scoped(pid, "/video")).json()["proxy_ready"] is True

    # register a *different* video
    other = tmp_path / "other.mp4"
    other.write_bytes(sample_video.read_bytes())
    r = client.post(scoped(pid, "/video"), json={"path": str(other)})
    assert r.status_code == 200
    assert not (proot / "proxy.mp4").exists()
    assert not (proot / "proxy.part.mp4").exists()
    assert not store.proxy_complete
    assert not (td / "c001_20.0.jpg").exists()  # thumbs cleared
    assert client.get(scoped(pid, "/video")).json()["proxy_ready"] is False


def test_thumb_clamp_and_404(client, sample_video, monkeypatch):
    pid, cands = _setup(client, sample_video)
    goal = next(c for c in cands if c["type"] == "goal")

    # t beyond duration is clamped to duration-0.1, not a 500
    r = client.get(scoped(pid, f"/candidates/{goal['id']}/thumb.jpg?t=99999"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    # ffmpeg failure / no output -> 404, not 500
    import highlights.app.backend.main as m

    def _boom(src, t, out):
        raise RuntimeError("boom")

    monkeypatch.setattr(m.fx, "thumbnail", _boom)
    r = client.get(scoped(pid, f"/candidates/{goal['id']}/thumb.jpg?t=1.5"))
    assert r.status_code == 404

    def _nofile(src, t, out):
        return None

    monkeypatch.setattr(m.fx, "thumbnail", _nofile)
    r = client.get(scoped(pid, f"/candidates/{goal['id']}/thumb.jpg?t=2.5"))
    assert r.status_code == 404
