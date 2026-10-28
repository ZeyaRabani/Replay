"""Backend validation invariants: atomic patch, render 422, window rules."""

import json
from pathlib import Path

import pytest
from conftest import SAMPLE, new_project, scoped

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]


def _setup(client, video):
    pid = new_project(client, video)
    r = client.post(scoped(pid, "/video"), json={"path": str(video)})
    assert r.status_code == 200, r.text
    r = client.post(scoped(pid, "/candidates/load"),
                    json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200, r.text
    return pid, r.json()


def _get(client, pid, cid):
    return next(c for c in client.get(scoped(pid, "/candidates")).json()
                if c["id"] == cid)


def test_reversed_patch_atomic(client, sample_video):
    pid, cands = _setup(client, sample_video)
    c = cands[0]
    orig = {k: c[k] for k in ("clip_start", "clip_end", "status", "type")}
    r = client.patch(scoped(pid, f"/candidates/{c['id']}"),
                     json={"clip_start": 30.0, "clip_end": 28.0})
    assert r.status_code == 422
    after = _get(client, pid, c["id"])
    for k, v in orig.items():
        assert after[k] == v
    # persisted state on disk is unchanged too
    import highlights.app.backend.main as m
    store = m.get_registry().get(pid)
    saved = json.loads(store.state_path.read_text())
    sc = next(x for x in saved["candidates"] if x["id"] == c["id"])
    for k, v in orig.items():
        assert sc[k] == v


def test_negative_and_beyond_duration_patches(client, sample_video):
    pid, cands = _setup(client, sample_video)
    c = cands[0]
    duration = client.get(scoped(pid, "/video")).json()["duration_s"]

    r = client.patch(scoped(pid, f"/candidates/{c['id']}"),
                     json={"clip_start": -2.0})
    assert r.status_code == 422
    assert _get(client, pid, c["id"])["clip_start"] == c["clip_start"]

    r = client.patch(scoped(pid, f"/candidates/{c['id']}"),
                     json={"clip_end": duration + 10})
    assert r.status_code == 422
    assert _get(client, pid, c["id"])["clip_end"] == c["clip_end"]

    # status change combined with invalid window: nothing applied
    r = client.patch(scoped(pid, f"/candidates/{c['id']}"),
                     json={"status": "confirmed", "clip_start": 30.0,
                           "clip_end": 28.0})
    assert r.status_code == 422
    assert _get(client, pid, c["id"])["status"] == c["status"]


def test_invalid_render_is_422_no_job(client, sample_video):
    pid, cands = _setup(client, sample_video)
    bad = cands[0]
    # corrupt the stored window directly (patch endpoint protects it)
    import highlights.app.backend.main as m
    stored = m.get_registry().get(pid).get(bad["id"])
    stored.clip_start, stored.clip_end = 30.0, 28.0
    n_jobs = len(m._jobs)
    r = client.post(scoped(pid, "/render"),
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
    # reencode-only path: bt709 tag filter, but no scale/overlay
    calls.clear()
    fx.cut_clip("/tmp/v.mp4", 1.0, 2.0, "/tmp/o.mp4", reencode=True)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    assert "setparams=color_primaries=bt709" in vf
    assert "scale" not in vf
    assert "drawtext" not in vf


def test_source_switch_revalidates_windows(client, sample_video, short_video):
    pid, cands = _setup(client, sample_video)
    goal = next(c for c in cands if c["type"] == "goal")
    assert goal["clip_end"] > 6.0  # longer than the short video
    r = client.post(scoped(pid, "/video"), json={"path": str(short_video)})
    assert r.status_code == 200
    dur = r.json()["duration_s"]
    for c in client.get(scoped(pid, "/candidates")).json():
        assert 0 <= c["clip_start"] < c["clip_end"] <= dur
    # candidate beyond duration gets a clamped end-of-video window;
    # it is hidden from the list endpoint, so read it from the store
    import highlights.app.backend.main as m
    g = m.get_registry().get(pid).get(goal["id"])
    assert g.clip_end == pytest.approx(dur)
    assert g.t == goal["t"]  # event time preserved


def test_load_candidates_into_short_video_valid(client, short_video):
    pid = new_project(client, short_video)
    r = client.post(scoped(pid, "/video"), json={"path": str(short_video)})
    dur = r.json()["duration_s"]
    r = client.post(scoped(pid, "/candidates/load"),
                    json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200
    for c in r.json():
        assert 0 <= c["clip_start"] < c["clip_end"] <= dur


def test_out_of_range_candidates_hidden_from_list(client, short_video):
    pid = new_project(client, short_video)
    r = client.post(scoped(pid, "/video"), json={"path": str(short_video)})
    dur = r.json()["duration_s"]
    r = client.post(scoped(pid, "/candidates/load"),
                    json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200
    # every sample event has t > 6s video duration -> hidden from the list,
    # but kept in the store (not deleted)
    listed = client.get(scoped(pid, "/candidates")).json()
    assert listed == []
    import highlights.app.backend.main as m
    store = m.get_registry().get(pid)
    assert len(store.candidates) == 5
    assert all(c.t > dur + 1 for c in store.candidates)


class _FakeProxyJob:
    def __init__(self, *a, **k):
        self.done = False
        self.error = None
        self.progress = 0.0

    def start(self):
        pass


def test_proxy_job_cmd_niced_two_threads():
    import shutil

    from highlights.app.backend import ffmpeg as fx
    cmd = fx.ProxyJob("/a.mp4", "/b.mp4", 10.0)._cmd()
    if shutil.which("nice"):
        assert cmd[:3] == ["nice", "-n", "19"]
    else:
        assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-threads") + 1] == "2"
    assert f"scale=-2:{fx.PROXY_HEIGHT}" in cmd


def test_maybe_auto_proxy_gates(client, short_video, monkeypatch):
    import highlights.app.backend.main as m
    from highlights.app.backend import ffmpeg as fx
    from highlights.app.backend import pipeline

    pid = new_project(client, short_video)
    p = m.get_registry().get(pid)
    monkeypatch.setattr(fx, "ProxyJob", _FakeProxyJob)

    builds = []
    orig_build = m._build_proxy
    monkeypatch.setattr(m, "_build_proxy",
                        lambda pr: builds.append(pr.id) or orig_build(pr))

    # idle -> a proxy job is started for this project
    m._maybe_auto_proxy(p)
    assert builds == [pid]
    assert pid in m._proxy_jobs
    assert not m._proxy_jobs[pid].done
    m._proxy_jobs.pop(pid)
    builds.clear()

    # proxy already complete -> nothing
    src = p.video.path
    p.set_proxy_complete(src)
    (p.root / "proxy.mp4").write_bytes(b"x")
    m._maybe_auto_proxy(p)
    assert builds == []

    # another project's proxy job running -> nothing
    m._proxy_jobs["other"] = _FakeProxyJob()
    m._maybe_auto_proxy(p)
    assert builds == []
    m._proxy_jobs.pop("other")

    # this project's pipeline running -> nothing
    pipeline.write_status(p, {"state": "running", "pid": 111, "updated_at": 1})
    monkeypatch.setattr(pipeline, "pid_owned_by", lambda pid, pr: True)
    m._maybe_auto_proxy(p)
    assert builds == []
    pipeline.write_status(p, {"state": "done", "pid": None})

    # another project's multiangle pipeline running -> nothing
    pid2 = new_project(client, short_video)
    p2 = m.get_registry().get(pid2)
    pipeline.write_status(p2, {"state": "running", "pid": 222, "updated_at": 1})
    m._maybe_auto_proxy(p)
    assert builds == []
