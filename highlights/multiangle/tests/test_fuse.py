"""Candidate fusion across angles."""

from highlights.multiangle.fuse import fuse_events


def _ev(t, conf=0.7, etype="shot"):
    return {"t": t, "type": etype, "confidence": conf}


def test_two_angles_cross_confirm():
    out = fuse_events(
        [[_ev(100.0, 0.7, "shot")],
         [_ev(102.5, 0.8, "shot")]],
        tm=[0.0, 0.0], labels=["a0", "a1"])
    assert len(out["events"]) == 1
    e = out["events"][0]
    assert e["cross_validation"] == "confirmed"
    assert e["signals"]["angles"] == [0, 1]
    assert 100 <= e["t"] <= 102.5
    assert "cross-confirmed" in e["notes"]
    assert "disputed" not in e["signals"]


def test_single_angle_pipeline_only():
    out = fuse_events([[_ev(50.0, 0.9)], [_ev(300.0, 0.6)]],
                      tm=[0.0, 0.0], labels=["a0", "a1"])
    assert len(out["events"]) == 2
    assert all(e["cross_validation"] == "pipeline_only" for e in out["events"])
    # single-angle confidence damped by 0.85
    e = next(e for e in out["events"] if e["t"] == 300.0)
    assert e["confidence"] <= 0.6 * 0.85 + 1e-3


def test_type_disagreement_disputed():
    out = fuse_events(
        [[_ev(200.0, 0.9, "goal")], [_ev(201.0, 0.6, "chance")]],
        tm=[0.0, 0.0], labels=["a0", "a1"])
    e = out["events"][0]
    assert e["type"] == "goal"  # highest priority wins
    assert e["signals"]["disputed"] is True
    assert e["signals"]["types"] == ["chance", "goal"]


def test_offset_mapping():
    """offset shifts angle's file time onto T."""
    out = fuse_events([[], [_ev(100.0)]], tm=[0.0, 5.0], labels=["a0", "a1"])
    assert abs(out["events"][0]["t"] - 105.0) < 0.01


def test_ids_and_ranking():
    evs = [[_ev(10 + i * 20, 0.5 + 0.01 * i) for i in range(5)]]
    out = fuse_events(evs + [[]], [0.0, 0.0], ["a0", "a1"])
    confs = [e["confidence"] for e in out["events"]]
    assert confs == sorted(confs, reverse=True)
    assert [e["id"] for e in out["events"]] == [f"event_{i:03d}" for i in range(1, 6)]


def test_fuse_events_piecewise_map():
    """Events on either side of an angle's offset jump land on the right
    shared times."""
    tm = [[{"file_lo": 0, "file_hi": 1000, "offset": 0.0}],
          [{"file_lo": 0, "file_hi": 50, "offset": 10.0},
           {"file_lo": 53, "file_hi": 100, "offset": 12.0}]]
    out = fuse_events(
        [[], [_ev(10.0, 0.9), _ev(80.0, 0.9)]], tm, ["a0", "a1"])
    ts = sorted(e["t"] for e in out["events"])
    assert ts == [20.0, 92.0]


def test_to_output_time():
    """shared-T events shift by -lo into rendered-video time."""
    from highlights.multiangle.fuse import to_output_time
    evs = [{"t": -640.2, "t_start": -643.2, "t_end": -635.2, "type": "goal"},
           {"t": 100.0, "t_start": 97.0, "t_end": 105.0, "clip_start": 95.0,
            "clip_end": 110.0}]
    out = to_output_time(evs, -748.2)
    assert out[0]["t"] == 108.0
    assert out[0]["t_start"] == 105.0 and out[0]["t_end"] == 113.0
    assert out[1]["t"] == 848.2
    assert out[1]["clip_start"] == 843.2 and out[1]["clip_end"] == 858.2
    # original untouched, absent keys not invented
    assert evs[0]["t"] == -640.2 and "clip_start" not in out[0]


def test_drop_outside_window():
    """events outside [0, dur_live] are dropped; kept ones re-id'd."""
    from highlights.multiangle.fuse import drop_outside_window
    evs = [{"t": -5.0, "confidence": 0.9, "id": "event_001", "rank": 1},
           {"t": 10.0, "confidence": 0.7, "id": "event_002", "rank": 2},
           {"t": 101.0, "confidence": 0.8, "id": "event_003", "rank": 3}]
    out = drop_outside_window(evs, 100.0)
    assert [e["t"] for e in out] == [10.0]
    assert out[0]["id"] == "event_001" and out[0]["rank"] == 1
