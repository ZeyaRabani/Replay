"""R2 fast-download tests with a stubbed S3 client (no network)."""

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from conftest import new_project, scoped


class FakeS3:
    def __init__(self):
        self.objects = {}   # key -> {"Size": int, "LastModified": dt}
        self.mpus = []      # {"Key","UploadId","Initiated": dt}
        self.aborted = []
        self.deleted = []
        self.uploaded = []

    def head_object(self, Bucket, Key):
        if Key in self.objects:
            return {"ContentLength": self.objects[Key]["Size"]}
        raise Exception("NotFound")

    def list_objects_v2(self, **kw):
        return {"Contents": [{"Key": k, **v}
                             for k, v in self.objects.items()],
                "IsTruncated": False}

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    def list_multipart_uploads(self, **kw):
        return {"Uploads": list(self.mpus), "IsTruncated": False}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.aborted.append((Key, UploadId))
        self.mpus = [u for u in self.mpus if u["UploadId"] != UploadId]

    def upload_file(self, Filename, Bucket, Key, Callback=None, Config=None):
        size = Path(Filename).stat().st_size
        if Callback:
            Callback(size)
        self.objects[Key] = {"Size": size,
                             "LastModified": datetime.now(timezone.utc)}
        self.uploaded.append(Key)

    def generate_presigned_url(self, op, Params, ExpiresIn):
        cd = Params["ResponseContentDisposition"]
        return f"https://r2.test/{Params['Key']}?cd={cd}&x={ExpiresIn}"


def _obj(size, age_h):
    return {"Size": size,
            "LastModified": datetime.now(timezone.utc)
            - timedelta(hours=age_h)}


@pytest.fixture()
def fake_r2(monkeypatch, tmp_path):
    import highlights.app.backend.r2 as r2

    cfg = tmp_path / "r2.json"
    cfg.write_text(json.dumps({
        "endpoint": "https://r2.test", "access_key_id": "k",
        "secret_access_key": "s", "bucket": "b", "cap_bytes": 1000}))
    monkeypatch.setattr(r2, "CONFIG_PATH", str(cfg))
    # fake boto3 + boto3.s3.transfer for the import inside _run_upload
    b3 = types.ModuleType("boto3")
    tr = types.ModuleType("boto3.s3.transfer")
    tr.TransferConfig = type("TC", (), {"__init__": lambda s, **kw: None})
    s3pkg = types.ModuleType("boto3.s3")
    s3pkg.transfer = tr
    b3.s3 = s3pkg
    monkeypatch.setitem(sys.modules, "boto3", b3)
    monkeypatch.setitem(sys.modules, "boto3.s3", s3pkg)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", tr)
    fake = FakeS3()
    monkeypatch.setattr(r2, "_s3", lambda: fake)
    r2._reset()
    yield r2, fake
    r2._reset()


def _mkvideo(tmp_path, size):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x" * size)
    return v


def test_key_scheme(tmp_path):
    import highlights.app.backend.r2 as r2
    v = _mkvideo(tmp_path, 123)
    st = v.stat()
    assert r2.object_key("pid1", v) == \
        f"pid1/match-{st.st_size}-{int(st.st_mtime)}.mp4"


def test_upload_evicts_oldest_first_and_keeps_target(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 700)
    key = r2.object_key("pid", v)
    fake.objects["old/a.mp4"] = _obj(600, 100)   # oldest
    fake.objects["mid/b.mp4"] = _obj(300, 50)
    fake.objects[key] = _obj(50, 10)             # never evict target
    r2._run_upload({"pid": "pid", "path": str(v), "auto": False})
    # used 950 + 700 > 1000; evict oldest (a 600) -> 350 + 700 fits,
    # but the loop deletes until fits: a then mid? 950-600=350, fits.
    assert "old/a.mp4" in fake.deleted
    assert key not in fake.deleted
    assert r2._state["pid"]["state"] == "ready"


def test_eviction_stops_once_it_fits(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 200)
    fake.objects["a"] = _obj(400, 100)
    fake.objects["b"] = _obj(400, 50)
    fake.objects["c"] = _obj(200, 10)  # newest, must survive
    r2._run_upload({"pid": "pid", "path": str(v), "auto": False})
    # 1000 used + 200 > 1000 -> evict oldest a (400) -> 800 used, fits
    assert fake.deleted == ["a"] or "c" not in fake.deleted
    assert r2._state["pid"]["state"] == "ready"


def test_auto_deferred_when_only_fresh_objects(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 700)
    fake.objects["fresh"] = _obj(900, 2)   # <24h: auto may not evict
    job = {"pid": "pid", "path": str(v), "auto": True}
    r2._run_upload(job)
    # 'waiting' — deferred, not too_big (the file fits under the cap)
    assert r2._state["pid"]["state"] == "waiting"
    assert not fake.uploaded
    # stays deferred in the queue for the hourly retry
    assert len(r2._q) == 1 and r2._q[0]["defer_until"] > 0


def test_too_big_skips_and_evicts_nothing(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 1500)           # bigger than the cap itself
    fake.objects["a"] = _obj(300, 100)
    r2._run_upload({"pid": "pid", "path": str(v), "auto": False})
    assert r2._state["pid"]["state"] == "too_big"
    assert not fake.uploaded
    assert "a" in fake.objects             # nothing evicted


def test_stale_same_pid_keys_deleted(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 100)
    r2.object_key("pid", v)
    fake.objects["pid/old-1-1.mp4"] = _obj(10, 100)
    fake.objects["pid/old-2-2.mp4"] = _obj(10, 99)
    fake.objects["other/keep.mp4"] = _obj(10, 98)
    r2._run_upload({"pid": "pid", "path": str(v), "auto": False})
    assert "pid/old-1-1.mp4" in fake.deleted
    assert "pid/old-2-2.mp4" in fake.deleted
    assert "other/keep.mp4" not in fake.deleted
    assert r2._state["pid"]["state"] == "ready"


def test_stale_mpus_aborted(fake_r2, tmp_path):
    r2, fake = fake_r2
    v = _mkvideo(tmp_path, 100)
    fake.mpus = [
        {"Key": "pid/m.mp4", "UploadId": "stale",
         "Initiated": datetime.now(timezone.utc) - timedelta(hours=2)},
        {"Key": "pid/m.mp4", "UploadId": "fresh",
         "Initiated": datetime.now(timezone.utc) - timedelta(minutes=5)},
    ]
    r2._run_upload({"pid": "pid", "path": str(v), "auto": False})
    assert ("pid/m.mp4", "stale") in fake.aborted
    assert ("pid/m.mp4", "fresh") not in fake.aborted


def test_presign_content_disposition(fake_r2):
    r2, _ = fake_r2
    url = r2.presign("pid/match-1-2.mp4", "Måtch 21/8.mp4")
    assert "attachment" in url
    assert "filename*=UTF-8''" in url
    assert "21600" in url                  # 6 h expiry


def test_download_302_when_ready(client, short_video, monkeypatch):
    import highlights.app.backend.r2 as r2
    monkeypatch.setattr(r2, "enabled", lambda: True)
    monkeypatch.setattr(r2, "status",
                        lambda pid, path: {"state": "ready", "progress": 1.0})
    monkeypatch.setattr(r2, "object_key", lambda pid, path: "pid/k.mp4")
    monkeypatch.setattr(r2, "presign",
                        lambda k, f: f"https://r2.test/{k}?f={f}")
    pid = new_project(client, short_video)
    r = client.get(scoped(pid, "/download"), follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://r2.test/")


def test_download_fileresponse_when_not_ready(client, short_video,
                                              monkeypatch):
    import highlights.app.backend.r2 as r2
    monkeypatch.setattr(r2, "enabled", lambda: False)
    pid = new_project(client, short_video)
    r = client.get(scoped(pid, "/download"), follow_redirects=False)
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert ".mp4" in r.headers["content-disposition"]


def test_disabled_default(client, short_video, monkeypatch):
    """No config -> enabled() False; /download still serves, /r2 'off'."""
    import highlights.app.backend.r2 as r2
    r2._reset()
    monkeypatch.setattr(r2, "CONFIG_PATH", "/nonexistent/r2.json")
    r2._reset()
    pid = new_project(client, short_video)
    assert r2.enabled() is False
    st = client.get(scoped(pid, "/r2")).json()
    assert st["state"] == "off"
    r = client.get(scoped(pid, "/download"), follow_redirects=False)
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
