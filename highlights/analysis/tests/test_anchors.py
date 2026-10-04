"""anchors: still-frame clicks resolve to nearest same-team active track
per moment; resolution feeds hard constraints for the linker."""

from highlights.analysis import anchors as anch

# identity homography: frame coords == pitch coords
CAL = {"angles": {"0": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}


def _tr(tid, team, start, end, x, y):
    n = round((end - start) / 0.5) + 1
    return {"id": tid, "team": team, "start": start, "end": end,
            "xy": [[x, y]] * n}


def _clk(cid, x, y, team="A", num=7, moment="mid", angle=0):
    return {"id": cid, "moment": moment, "angle": angle,
            "fx": x, "fy": y, "team": team, "number": num}


def _resolve(clicks, tracks, moments=((("mid", 100.0),))):
    doc = {"moments": [{"id": m, "t": t} for m, t in moments],
           "clicks": clicks}
    return anch.resolve_clicks(doc, {"tracks": tracks}, CAL)


def test_nearest_same_team_track():
    tracks = [_tr(1, "A", 0, 200, 10, 10), _tr(2, "A", 0, 200, 30, 10)]
    out = _resolve([_clk("c", 11.5, 10)], tracks)
    assert out["clicks"][0]["track_id"] == 1
    assert out["clicks"][0]["dist_m"] == 1.5


def test_wrong_team_and_inactive_ignored():
    tracks = [_tr(1, "B", 0, 200, 10, 10),      # other team
              _tr(2, "A", 0, 50, 10, 10),       # not active at t=100
              _tr(3, "A", 0, 200, 12, 10)]
    out = _resolve([_clk("c", 10.2, 10, team="A")], tracks)
    assert out["clicks"][0]["track_id"] == 3


def test_far_click_unresolved_with_note():
    tracks = [_tr(1, "A", 0, 200, 10, 10)]
    out = _resolve([_clk("c", 30, 30)], tracks)
    c = out["clicks"][0]
    assert c["track_id"] is None
    assert "within 3 m" in c["note"]
    assert "nearest" in c["note"]


def test_greedy_never_double_assigns():
    # c2 hugs track 2; greedy must leave track 1 for c1
    tracks = [_tr(1, "A", 0, 200, 0, 0), _tr(2, "A", 0, 200, 1, 0)]
    out = _resolve([_clk("c1", 0.9, 0, num=7), _clk("c2", 0.95, 0, num=9)],
                   tracks)
    by_id = {c["id"]: c for c in out["clicks"]}
    assert by_id["c1"]["track_id"] == 1
    assert by_id["c2"]["track_id"] == 2


def test_duplicate_number_second_unresolved():
    tracks = [_tr(1, "A", 0, 200, 10, 10)]
    out = _resolve([_clk("a", 10, 10, num=7), _clk("b", 10.5, 10, num=7)],
                   tracks)
    by_id = {c["id"]: c for c in out["clicks"]}
    assert by_id["a"]["track_id"] == 1
    assert by_id["b"]["track_id"] is None
    assert by_id["b"]["note"] == "duplicate number"


def test_uncalibrated_camera():
    out = anch.resolve_clicks(
        {"moments": [{"id": "mid", "t": 100.0}],
         "clicks": [_clk("c", 10, 10, angle=1)]},
        {"tracks": [_tr(1, "A", 0, 200, 10, 10)]}, CAL)
    assert out["clicks"][0]["track_id"] is None
    assert out["clicks"][0]["note"] == "camera not calibrated"


def test_conflicting_numbers_void_both():
    tracks = [_tr(1, "A", 0, 200, 10, 10)]
    out = anch.resolve_clicks(
        {"moments": [{"id": "start", "t": 50.0}, {"id": "end", "t": 150.0}],
         "clicks": [_clk("a", 10, 10, num=7, moment="start"),
                    _clk("b", 10, 10, num=9, moment="end")]},
        {"tracks": tracks}, CAL)
    for c in out["clicks"]:
        assert c["track_id"] is None
        assert c["note"] == "conflicting numbers"
    assert anch.constraints(out) == {}


def test_constraints_map():
    doc = {"clicks": [
        {"id": "a", "moment": "start", "angle": 0, "fx": 0, "fy": 0,
         "team": "A", "number": 7, "track_id": 42},
        {"id": "b", "moment": "start", "angle": 0, "fx": 0, "fy": 0,
         "team": "B", "number": 3, "track_id": None},
    ]}
    assert anch.constraints(doc) == {42: ("A", 7)}


def test_default_moments_spaced():
    doc = {"t0": 100.0, "summary": {"visible_hist": [1] * 1001},
           "tracks": []}
    ms = {m["id"]: m["t"] for m in anch.default_moments(doc)}
    assert ms["start"] == 190.0
    assert ms["mid"] == 350.0
    assert ms["end"] == 510.0
