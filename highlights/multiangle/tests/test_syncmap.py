"""syncmap: piecewise file<->shared mapping, gaps, clamps, fallback."""


from highlights.multiangle import render
from highlights.multiangle.syncmap import (
    boundaries_shared,
    file_range_for_shared,
    file_to_shared,
    scalar_offsets,
    shared_to_file,
    timemap_from_sync,
)

TM = [
    [{"file_lo": 0.0, "file_hi": 100.0, "offset": 0.0}],
    [{"file_lo": 0.0, "file_hi": 50.0, "offset": 10.0},     # 10..60
     {"file_lo": 53.0, "file_hi": 100.0, "offset": 12.0}],  # 65..112 gap 60-65
]


def test_file_to_shared_per_segment():
    assert file_to_shared(TM, 1, 10.0) == 20.0
    assert file_to_shared(TM, 1, 60.0) == 72.0
    # outside the file range clamps to the nearest edge
    assert file_to_shared(TM, 1, 200.0) == 112.0


def test_shared_to_file_gap_and_clamp():
    assert shared_to_file(TM, 1, 30.0) == 20.0
    assert shared_to_file(TM, 1, 80.0) == 68.0
    assert shared_to_file(TM, 1, 62.0) is None        # gap 60-65
    assert shared_to_file(TM, 1, 62.0, clamp=True) == 50.0  # nearest edge


def test_overlap_later_segment_wins():
    tm = [[{"file_lo": 0, "file_hi": 10, "offset": 0}],
          [{"file_lo": 0, "file_hi": 60, "offset": 10},
           {"file_lo": 50, "file_hi": 100, "offset": 8}]]  # overlap 58..70
    assert shared_to_file(tm, 1, 60.0) == 52.0   # 60-8 (not 50 of seg A)


def test_boundaries_and_file_range():
    assert boundaries_shared(TM, 1) == [60.0, 65.0]
    assert boundaries_shared(TM, 0) == []
    f0, f1 = file_range_for_shared(TM, 1, 30.0, 80.0)
    assert (f0, f1) == (20.0, 68.0)
    # a range spanning only the gap clamps to edges
    f0, f1 = file_range_for_shared(TM, 1, 61.0, 64.0)
    assert (f0, f1) == (50.0, 53.0)


def test_legacy_fallback():
    sync = {"offsets": [0.0, -748.2]}
    tm = timemap_from_sync(sync, durations=[6000.0, 5400.0])
    assert tm[1] == [{"file_lo": 0.0, "file_hi": 5400.0, "offset": -748.2}]
    assert shared_to_file(tm, 1, 0.0) == 748.2
    assert file_to_shared(tm, 1, 0.0) == -748.2
    assert scalar_offsets(tm) == [0.0, -748.2]


def test_plan_segments_splits_at_boundary():
    tm = [[{"file_lo": 0, "file_hi": 1000, "offset": 0}],
          [{"file_lo": 0, "file_hi": 50, "offset": 10},
           {"file_lo": 53, "file_hi": 100, "offset": 12}]]
    seg = {"angle": 1, "t_start": 20.0, "t_end": 90.0}
    plans = render.plan_segments([seg], tm, 0.0, 1000.0, [1000.0, 100.0])
    # shared bounds of seg A: 10..60; seg B: 65..112 -> boundary edges
    # 60 and 65 inside [20, 90) -> pieces 20-60, 60-65, 65-90
    assert len(plans) == 3
    assert [p["dur"] for p in plans] == [40.0, 5.0, 25.0]
    assert [p["t_file"] for p in plans] == [10.0, 50.0, 53.0]
    assert all(p["seg_index"] == 0 for p in plans)


def test_mezz_range_via_map():
    m0, m1 = render.mezz_range(1, TM, 0.0, 90.0, [100.0])
    assert m0 <= m1
    assert m0 >= 0.0 and m1 <= 100.0


def test_scalar_offsets_picks_longest_segment():
    assert scalar_offsets(TM) == [0.0, 10.0]
