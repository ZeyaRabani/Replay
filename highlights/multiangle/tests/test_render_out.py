"""plan_segments: output-time segments -> per-angle file time.

Shared T is a0-referenced; segment t is output time (0 = union_lo), so
angle a's file time for output second t is t + union_lo - offsets[a].
"""

from highlights.multiangle.render import plan_segments

OFFSETS = [0.0, -748.2, -520.2]
LO, HI = -748.2, 5414.341
DURS = [5414.341, 5400.0, 5200.0]


def test_plan_segments_maps_output_time_to_file_time():
    segs = [
        {"angle": 1, "t_start": 0.0, "t_end": 233.0},
        {"angle": 0, "t_start": 1000.0, "t_end": 1100.0},
    ]
    plans = plan_segments(segs, OFFSETS, LO, HI, DURS)
    assert len(plans) == 2
    # output 0 on angle 1 = its file start (union lo == -offsets[1])
    assert plans[0]["t_file"] == 0.0
    assert plans[0]["dur"] == 233.0
    # output 1000 on angle 0 = file 1000 + lo - 0
    assert abs(plans[1]["t_file"] - 251.8) < 0.01
    assert plans[1]["dur"] == 100.0


def test_plan_segments_skip_beyond_duration():
    segs = [
        {"angle": 0, "t_start": 6000.0, "t_end": 6100.0},   # file 5251.8 ok
        {"angle": 1, "t_start": 6000.0, "t_end": 6100.0},   # file 6000 > 5400
    ]
    plans = plan_segments(segs, OFFSETS, LO, HI, DURS)
    assert plans[0]["seg_index"] == 0 and not plans[0].get("skip")
    assert plans[1]["seg_index"] == 1 and plans[1]["skip"] is True


def test_plan_segments_clips_to_output_range():
    segs = [
        {"angle": 0, "t_start": -50.0, "t_end": 100.0},          # clipped to 0
        {"angle": 0, "t_start": HI - LO + 10.0, "t_end": HI - LO + 100.0},
    ]
    plans = plan_segments(segs, OFFSETS, LO, HI, DURS)
    assert plans[0]["t0"] == 0.0 and plans[0]["t1"] == 100.0
    # fully out of range -> dropped
    assert all(p["seg_index"] != 1 or p.get("skip") for p in plans[1:])
    # second segment is dropped entirely (t1 clipped to dur_out <= t0... but
    # t_start itself is beyond dur_out: clipped t1 == dur_out < t0 -> dropped)
    assert len([p for p in plans if p["seg_index"] == 1]) == 0
