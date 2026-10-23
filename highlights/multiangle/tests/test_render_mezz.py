"""Output-time mezzanine piece mapping and cache-key tests."""

import pytest

from highlights.multiangle.render import mezz_key, mezz_pieces


def test_mezz_pieces_fill_gaps_and_later_timemap_wins():
    tm = [[
        {"file_lo": 0.0, "file_hi": 10.0, "offset": 3.37},
        {"file_lo": 10.0, "file_hi": 24.0, "offset": 2.77},
    ]]
    pieces = mezz_pieces(0, tm, 0.0, 20.0, [24.0])
    # initial coverage gap, first mapping, then second mapping wins overlap
    assert not pieces[0]["real"]
    real = [p for p in pieces if p["real"]]
    assert real[0]["file_start"] == 0
    assert real[1]["file_start"] == 10
    # last piece resumes mid-file at shared 13.37 -> file 10.5967
    assert real[-1]["file_start"] == pytest.approx(10.596666666666668)
    assert sum(p["frames"] for p in pieces) == 20 * 30
    assert next(p["n0"] for p in pieces) == 0
    assert pieces[-1]["n1"] == 600


def test_mezz_key_changes_for_mapping_and_has_new_salt():
    base = [{"n0": 0, "n1": 30, "frames": 30,
             "a0": 0, "a1": 48000, "a_start": 1.3,
             "file_start": 1.3, "real": True}]
    changed = [{**base[0], "file_start": 1.4}]
    assert mezz_key(0, base) != mezz_key(0, changed)
    assert mezz_key(0, base) != mezz_key(1, base)
