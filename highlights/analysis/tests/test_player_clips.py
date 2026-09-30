"""Per-identity reel selection + build on a synthetic testsrc match."""

import json
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from highlights.analysis import player_clips as pc


def _walk(tid, start, xs, team="A"):
    return {"id": tid, "team": team, "start": start,
            "end": start + 0.5 * (len(xs) - 1),
            "xy": [[x, 20.0] for x in xs]}


def _dash_track(tid=1, start=100.0):
    """Stand still, then two 7 m/s dashes 20 s apart."""
    xs = [10.0] * 20
    for _ in range(6):
        xs.append(xs[-1] + 3.5)
    xs += [xs[-1]] * 34
    for _ in range(6):
        xs.append(xs[-1] - 3.5)
    xs += [xs[-1]] * 20
    return _walk(tid, start, xs)


def test_fastest_runs_are_peaks_and_separated():
    t = _dash_track()
    runs = pc.fastest_runs(t["start"], t["xy"])
    assert len(runs) == 2
    assert all(r["type"] == "sprint" and r["speed_ms"] >= 6.5 for r in runs)
    assert abs(runs[0]["t"] - runs[1]["t"]) >= pc.PRE_S + pc.POST_S


def test_identity_path_averages_duplicates():
    a = _walk(1, 0.0, [10.0, 10.0, 10.0])
    b = _walk(2, 0.5, [12.0, 12.0])
    start, xy = pc.identity_path({"track_ids": [1, 2]}, [a, b])
    assert start == 0.0
    assert [p[0] for p in xy] == [10.0, 11.0, 11.0]


def test_select_items_output_time_events_and_merge():
    tr = _dash_track(start=100.0)
    ident = {"id": "A1", "track_ids": [1], "name": "Nine"}
    lo = 50.0                         # shared 100 == output 50
    ball = [[112.0, 25.0, 21.0], [140.0, 40.0, 40.0]]
    cands = [
        # near the ball + the player -> in (merged with the first dash)
        {"id": "e1", "t": 62.0, "type": "shot", "status": "confirmed"},
        # far from the ball -> out
        {"id": "e2", "t": 90.0, "type": "shot", "status": "confirmed"},
        # scorer by roster name -> in, even with no ball
        {"id": "e3", "t": 20.0, "type": "goal", "status": "confirmed"},
        # unconfirmed / non-goal types -> out
        {"id": "e4", "t": 62.0, "type": "goal", "status": "pending"},
        {"id": "e5", "t": 62.0, "type": "attack", "status": "confirmed"},
    ]
    roster = {"players": [{"id": "p1", "name": "nine "}],
              "scorers": {"e3": "p1"}}
    items = pc.select_items(ident, [tr], ball=ball, candidates=cands,
                            roster=roster, lo=lo, duration=200.0)
    kinds = [i["type"] for i in items]
    assert kinds[0] == "goal" and items[0]["parts"][0]["reason"] == "scorer"
    shot = next(i for i in items if i["type"] == "shot")
    assert {p["type"] for p in shot["parts"]} == {"shot", "sprint"}
    assert any(p.get("candidate_id") == "e1" for p in shot["parts"])
    assert not any(p.get("candidate_id") in ("e2", "e4", "e5")
                   for i in items for p in i["parts"])
    # sprint windows are output time: shared peak - lo, 4 s pre / 3 s post
    spr = [i for i in items if i["type"] == "sprint"]
    assert len(spr) == 1
    assert spr[0]["clip_end"] - spr[0]["clip_start"] == pytest.approx(7.0)
    assert 50.0 < spr[0]["t"] < 100.0
    starts = [i["clip_start"] for i in items]
    assert starts == sorted(starts)
    assert all(a["clip_end"] < b["clip_start"]
               for a, b in pairwise(items))


def test_output_offset_prefers_active_cut(tmp_path):
    ma = tmp_path / "multiangle"
    (ma / "cuts" / "c1").mkdir(parents=True)
    (ma / "sync.json").write_text(json.dumps(
        {"coverage": {"union": [-10.0, 500.0]}}))
    (ma / "cut_range.json").write_text(json.dumps({"lo": 30.0, "hi": 400.0}))
    assert pc.output_offset(tmp_path) == 30.0
    (ma / "cuts" / "c1" / "meta.json").write_text(json.dumps(
        {"range": [42.0, 300.0]}))
    (ma / "cuts" / "active.json").write_text(json.dumps({"id": "c1"}))
    assert pc.output_offset(tmp_path) == 42.0


def _testsrc(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(path)], check=True, capture_output=True)


def test_build_reel_cli_on_testsrc(tmp_path):
    v2 = tmp_path / "analysis" / "players_v2"
    v2.mkdir(parents=True)
    (tmp_path / "multiangle").mkdir()
    (tmp_path / "multiangle" / "cut_range.json").write_text(
        json.dumps({"lo": 90.0, "hi": 190.0}))
    (v2 / "tracks.json").write_text(json.dumps(
        {"step": 0.5, "t0": 90.0, "tracks": [_dash_track(start=100.0)],
         "ball": []}))
    (v2 / "identities.json").write_text(json.dumps(
        {"identities": [{"id": "A1", "team": "A", "track_ids": [1],
                         "name": None}]}))
    _testsrc(tmp_path / "match.mp4", 60)
    assert pc.main(["--project-dir", str(tmp_path), "--identity", "A1"]) == 0
    out = pc.reels_dir(tmp_path, "A1")
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "done" and st["n_clips"] == 2
    man = json.loads((out / "reel.json").read_text())
    assert man["lo_shared"] == 90.0
    assert man["reel_s"] == pytest.approx(14.0, abs=0.6)
    assert (out / "reel.mp4").stat().st_size > 0
    # unknown identity -> failed status, non-zero exit
    assert pc.main(["--project-dir", str(tmp_path), "--identity", "B7"]) == 1
    st = json.loads((pc.reels_dir(tmp_path, "B7") / "status.json").read_text())
    assert st["state"] == "failed" and "B7" in st["error"]
