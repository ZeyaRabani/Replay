"""Self-healing watchdog tests — pass_once() with recorded spawns."""

import os
import time

from conftest import new_project

import highlights.app.backend.main as m
from highlights.app.backend import pipeline, watchdog
from highlights.pipeline.download import BOT_CHECK_MSG, COOKIES_REJECTED_MSG


def _set_status(p, **kw):
    status = {
        "state": "failed", "stage": "download", "progress": 0.0,
        "stage_progress": 0.0, "message": kw.get("error") or "failed",
        "error": kw.get("error"), "started_at": time.time() - 100,
        "updated_at": time.time() - 10, "finished_at": time.time() - 10,
        "pid": 999999999,
    }
    status.update(kw)
    pipeline.write_status(p, status)
    p.set_pipeline_state(status["state"])
    return status


def _record_spawns(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "spawn",
        lambda p, **kw: calls.append(("spawn", p, kw)) or {"state": "queued"})
    monkeypatch.setattr(
        pipeline, "spawn_multiangle",
        lambda p, **kw: calls.append(("spawn_multiangle", p, kw))
        or {"state": "queued"})
    return calls


def _fail_state(p, **kw):
    watchdog._save_state(p, kw)


def test_watchdog_transient_respects_backoff(client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    _set_status(p, error="HTTP Error 503: Service Unavailable")

    # first pass: schedules attempt 1 and respawns
    watchdog.pass_once()
    assert len(calls) == 1
    ar = watchdog._load_state(p)
    assert ar["n"] == 1 and ar["next_at"] > time.time()

    # again before next_at -> nothing (failed status re-written by the
    # fake spawn in real life; here we re-fail it to simulate)
    _set_status(p, error="HTTP Error 503: Service Unavailable")
    watchdog.pass_once()
    assert len(calls) == 1

    # due now -> attempt 2
    _fail_state(p, n=1, next_at=0)
    watchdog.pass_once()
    assert len(calls) == 2
    assert watchdog._load_state(p)["n"] == 2


def test_watchdog_gives_up_after_12(client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    _set_status(p, error="HTTP Error 503: Service Unavailable")
    _fail_state(p, n=watchdog.MAX_ATTEMPTS, next_at=0)
    watchdog.pass_once()
    assert calls == []
    status = pipeline.read_status(p)
    assert status["error"].startswith("gave up after 12 automatic retries")


def test_watchdog_cookie_failure_waits_for_new_cookies(
        client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    _set_status(p, error=COOKIES_REJECTED_MSG)

    # no cookies saved -> no retry
    watchdog.pass_once()
    assert calls == []

    # save cookies -> next pass respawns once
    r = client.put("/api/me/youtube-cookies",
                   json={"cookies_text":
                         "youtube.com\tTRUE\t/\tFALSE\t0\tX\tY"})
    assert r.status_code == 200, r.text
    watchdog.pass_once()
    assert len(calls) == 1

    # same mtime -> no repeat
    _set_status(p, error=COOKIES_REJECTED_MSG)
    watchdog.pass_once()
    assert len(calls) == 1

    # newer cookies -> retries again
    ck = m._user_cookies_path(p.owner)
    time.sleep(0.02)
    ck.write_text("youtube.com\tTRUE\t/\tFALSE\t0\tX\tZ")
    watchdog.pass_once()
    assert len(calls) == 2

    # generic bot check (no cookies used) is also cookie-failure
    _set_status(p, error=BOT_CHECK_MSG)
    watchdog.pass_once()
    assert len(calls) == 2  # mtime already tried


def test_watchdog_never_retries_cancelled_or_plain(
        client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    _set_status(p, error="cancelled")
    watchdog.pass_once()
    _set_status(p, error="cancelled by user")
    watchdog.pass_once()
    _set_status(p, error="angle 0: no file and no url")
    watchdog.pass_once()
    assert calls == []


def test_watchdog_interrupted_and_stale_error(
        client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)

    # interrupted (dead pid after restart) -> scheduled like transient
    _set_status(p, error=watchdog.INTERRUPTED_MSG)
    watchdog.pass_once()
    assert len(calls) == 1

    # stale stage error -> one immediate retry
    _set_status(p, error="scoreboard.json missing — request it via the Score card")
    watchdog.pass_once()
    assert len(calls) == 2
    # but only once
    _set_status(p, error="scoreboard.json missing — request it via the Score card")
    watchdog.pass_once()
    assert len(calls) == 2


def test_watchdog_leaves_running_alive(client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    import os
    _set_status(p, state="running", pid=os.getpid(), error=None,
                message="downloading")
    watchdog.pass_once()
    assert calls == []


def test_cookies_source_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_WORKDIR", str(tmp_path))
    user_f = m._user_cookies_path("alice")
    user_f.parent.mkdir(parents=True, exist_ok=True)
    user_f.write_text("user-cookies")
    shared_f = m._shared_cookies_path()
    shared_f.parent.mkdir(parents=True, exist_ok=True)
    shared_f.write_text("shared-cookies")
    os.utime(user_f, (1000, 1000))
    os.utime(shared_f, (2000, 2000))
    assert m._cookies_source("alice") == shared_f
    os.utime(user_f, (3000, 3000))
    assert m._cookies_source("alice") == user_f
    user_f.unlink()
    assert m._cookies_source("alice") == shared_f
    shared_f.unlink()
    assert m._cookies_source("alice") is None


def test_cookie_failed_respawns_on_newer_shared_cookies(
        client, sample_video, monkeypatch):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    calls = _record_spawns(monkeypatch)
    _set_status(p, error=COOKIES_REJECTED_MSG)

    # owner's file older than tried; only the shared (admin) file is newer
    owner_f = m._user_cookies_path(p.owner)
    owner_f.parent.mkdir(parents=True, exist_ok=True)
    owner_f.write_text("old")
    os.utime(owner_f, (1000, 1000))
    shared_f = m._shared_cookies_path()
    shared_f.parent.mkdir(parents=True, exist_ok=True)
    shared_f.write_text("fresh")
    os.utime(shared_f, (2000, 2000))

    watchdog.pass_once()
    assert len(calls) == 1
    assert watchdog._load_state(p)["cookies_mtime_tried"] == 2000

    # already-tried mtime -> no further respawn
    _set_status(p, error=COOKIES_REJECTED_MSG)
    watchdog.pass_once()
    assert len(calls) == 1


def test_admin_cookies_put_writes_shared_too(client, monkeypatch):
    text = "youtube.com\tTRUE\t/\tFALSE\t0\tk\tv\n"

    # admin put (no share flag) writes user + shared
    monkeypatch.setenv("REPLAY_ADMINS", "tester")
    r = client.put("/api/me/youtube-cookies", json={"cookies_text": text})
    assert r.status_code == 200, r.text
    assert m._user_cookies_path("tester").is_file()
    assert m._shared_cookies_path().is_file()

    # non-admin: share is 403; plain put writes only the user file
    monkeypatch.setenv("REPLAY_ADMINS", "someone-else")
    m._shared_cookies_path().unlink()
    r = client.put("/api/me/youtube-cookies",
                   json={"cookies_text": text, "share": True})
    assert r.status_code == 403
    r = client.put("/api/me/youtube-cookies", json={"cookies_text": text})
    assert r.status_code == 200, r.text
    assert m._user_cookies_path("tester").is_file()
    assert not m._shared_cookies_path().is_file()


def test_user_default_cookies_writes_all_distinct(client, sample_video):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    user_f = m._user_cookies_path(p.owner)
    user_f.parent.mkdir(parents=True, exist_ok=True)
    user_f.write_text("user-cookies")
    shared_f = m._shared_cookies_path()
    shared_f.parent.mkdir(parents=True, exist_ok=True)
    shared_f.write_text("shared-cookies")

    joined = m._user_default_cookies(p, p.owner)
    parts = joined.split(os.pathsep)
    assert len(parts) == 2
    assert (p.source_dir / "cookies.txt").is_file()
    assert (p.source_dir / "cookies.1.txt").is_file()
    assert oct((p.source_dir / "cookies.txt").stat().st_mode)[-3:] == "600"
    # _project_cookies picks both back up for reruns
    assert m._project_cookies(p) == joined


def test_user_default_cookies_dedupes_identical(client, sample_video):
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    user_f = m._user_cookies_path(p.owner)
    user_f.parent.mkdir(parents=True, exist_ok=True)
    user_f.write_text("same-cookies")
    shared_f = m._shared_cookies_path()
    shared_f.parent.mkdir(parents=True, exist_ok=True)
    shared_f.write_text("same-cookies")

    joined = m._user_default_cookies(p, p.owner)
    assert os.pathsep not in joined
    assert (p.source_dir / "cookies.txt").is_file()
    assert not (p.source_dir / "cookies.1.txt").exists()
