"""Candidate fusion across angles."""

from highlights.multiangle.fuse import fuse_events


def _ev(t, conf=0.7, etype="shot"):
    return {"t": t, "type": etype, "confidence": conf}


def test_two_angles_cross_confirm():
    out = fuse_events(
        [[_ev(100.0, 0.7, "shot")],
         [_ev(102.5, 0.8, "shot")]],
        offsets=[0.0, 0.0], labels=["a0", "a1"])
    assert len(out["events"]) == 1
    e = out["events"][0]
    assert e["cross_validation"] == "confirmed"
    assert e["signals"]["angles"] == [0, 1]
    assert 100 <= e["t"] <= 102.5
    assert "cross-confirmed" in e["notes"]
    assert "disputed" not in e["signals"]


def test_single_angle_pipeline_only():
    out = fuse_events([[_ev(50.0, 0.9)], [_ev(300.0, 0.6)]],
                      offsets=[0.0, 0.0], labels=["a0", "a1"])
    assert len(out["events"]) == 2
    assert all(e["cross_validation"] == "pipeline_only" for e in out["events"])
    # single-angle confidence damped by 0.85
    e = next(e for e in out["events"] if e["t"] == 300.0)
    assert e["confidence"] <= 0.6 * 0.85 + 1e-3


def test_type_disagreement_disputed():
    out = fuse_events(
        [[_ev(200.0, 0.9, "goal")], [_ev(201.0, 0.6, "chance")]],
        offsets=[0.0, 0.0], labels=["a0", "a1"])
    e = out["events"][0]
    assert e["type"] == "goal"  # highest priority wins
    assert e["signals"]["disputed"] is True
    assert e["signals"]["types"] == ["chance", "goal"]


def test_offset_mapping():
    """offset shifts angle's file time onto T."""
    out = fuse_events([[], [_ev(100.0)]], offsets=[0.0, 5.0], labels=["a0", "a1"])
    assert abs(out["events"][0]["t"] - 105.0) < 0.01


def test_ids_and_ranking():
    evs = [[_ev(10 + i * 20, 0.5 + 0.01 * i) for i in range(5)]]
    out = fuse_events(evs + [[]], [0.0, 0.0], ["a0", "a1"])
    confs = [e["confidence"] for e in out["events"]]
    assert confs == sorted(confs, reverse=True)
    assert [e["id"] for e in out["events"]] == [f"event_{i:03d}" for i in range(1, 6)]
