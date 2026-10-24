"""Output-time segment boundaries use one shared integer-second grid."""

from highlights.multiangle.render import plan_segments


def test_plan_segments_snap_identically_across_angles():
    plans = plan_segments([
        {"angle": 0, "t_start": 0.0, "t_end": 1.49},
        {"angle": 1, "t_start": 1.49, "t_end": 2.51},
        {"angle": 2, "t_start": 2.51, "t_end": 4.0},
    ])
    assert [(p["angle"], p["k0"], p["k1"]) for p in plans] == [
        (0, 0, 1), (1, 1, 3), (2, 3, 4)]
    assert sum(p["frames"] for p in plans) == 120
    assert sum(p["samples"] for p in plans) == 4 * 48000


def test_empty_rounded_piece_does_not_consume_time():
    plans = plan_segments([
        {"angle": 0, "t_start": 0.0, "t_end": 0.49},
        {"angle": 1, "t_start": 0.49, "t_end": 1.51},
        {"angle": 0, "t_start": 1.51, "t_end": 2.0},
    ])
    assert [(p["angle"], p["k0"], p["k1"]) for p in plans] == [(1, 0, 2)]
    assert sum(p["frames"] for p in plans) == 2 * 30


def test_plan_segments_repairs_overlap_gap_and_inverted():
    plans = plan_segments([
        {"t_start": 0, "t_end": 10, "angle": 0},
        {"t_start": 10, "t_end": 9, "angle": 2},   # inverted -> skipped
        {"t_start": 9, "t_end": 14, "angle": 1},   # overlap -> clamped
        {"t_start": 17, "t_end": 20, "angle": 0},  # gap -> prev extended
    ])
    assert [(p["k0"], p["k1"], p["angle"]) for p in plans] == [
        (0, 10, 0), (10, 17, 1), (17, 20, 0)]
    assert plans[1]["frames"] == 7 * 30 and plans[2]["frames"] == 3 * 30
    total = sum(p["frames"] for p in plans)
    assert total == (plans[-1]["k1"] - plans[0]["k0"]) * 30
