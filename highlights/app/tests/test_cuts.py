"""Director-cut version snapshots (multiangle/cuts/)."""

import json
import os
import shutil

from conftest import scoped
from test_multiangle import _create, _wait


def test_cuts_snapshot_activate_delete(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    r = _create(client, 2, title="cuts")
    pid = r.json()["id"]
    _wait(client, pid)

    import highlights.app.backend.main as m
    from highlights.multiangle.cuts import snapshot_cut

    p = m.get_registry().get(pid)
    live = p.root / "match.mp4"
    assert live.is_file()

    # legacy project: first GET snapshots the current cut
    res = client.get(scoped(pid, "/multiangle/cuts"))
    assert res.status_code == 200, res.text
    d = res.json()
    assert len(d["cuts"]) == 1
    cut0 = d["cuts"][0]
    assert d["active"] == cut0["id"]
    assert cut0["label"] == "Normal (AI)"
    cdir = p.multiangle_dir / "cuts" / cut0["id"]
    assert os.stat(cdir / "match.mp4").st_ino == os.stat(live).st_ino

    # a second, different cut lands (simulate a re-cut): new video + director
    os.replace(live, live.with_name("match_old.mp4"))
    shutil.copyfile(short_video, live)
    (p.multiangle_dir / "director.json").write_text(
        json.dumps({"n_cuts": 999, "style": "fast", "ratios": {},
                    "angle_share": {}, "segments": []}))
    meta2 = snapshot_cut(p.root, "fast")
    assert meta2 and meta2["label"] == "Fast (AI)"
    # restore live state to the second cut's content (as a real recut would)
    cuts = client.get(scoped(pid, "/multiangle/cuts")).json()
    assert len(cuts["cuts"]) == 2
    assert cuts["active"] == meta2["id"]

    # activating a cut restores the range it was made with (written to
    # multiangle/cut_range.json, not inside cuts/)
    cr = p.multiangle_dir / "cut_range.json"
    cr.write_text('{"lo": 9.0, "hi": 99.0}')
    mpath = cdir / "meta.json"
    mm = json.loads(mpath.read_text()); mm["range"] = [5.0, 10.0]
    mpath.write_text(json.dumps(mm))
    r = client.post(scoped(pid, f"/multiangle/cuts/{cut0['id']}/activate"))
    assert r.status_code == 200, r.text
    assert r.json()["active"] == cut0["id"]
    assert json.loads(cr.read_text()) == {"lo": 5.0, "hi": 10.0}
    assert not (p.multiangle_dir / "cuts" / "cut_range.json").exists()
    assert os.stat(live).st_ino == os.stat(cdir / "match.mp4").st_ino
    dj = json.loads((p.multiangle_dir / "director.json").read_text())
    assert dj["n_cuts"] == 133          # fake runner's original director
    assert p.meta["cut_style"] == "normal"
    assert p.video is not None and p.video.path.endswith("match.mp4")

    # unknown cut -> 404; active cannot be deleted; non-active can
    assert client.post(scoped(pid, "/multiangle/cuts/nope/activate")) \
        .status_code == 404
    assert client.delete(scoped(pid, f"/multiangle/cuts/{cut0['id']}")) \
        .status_code == 409
    r = client.delete(scoped(pid, f"/multiangle/cuts/{meta2['id']}"))
    assert r.status_code == 200, r.text
    assert len(r.json()["cuts"]) == 1

    # download route serves the snapshot with a filename
    r = client.get(scoped(pid, f"/multiangle/cuts/{cut0['id']}/match.mp4"))
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
