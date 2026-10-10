"""scoreboard.py: timeline, abbreviations, filtergraph, ffmpeg e2e."""

import shutil
import subprocess
from pathlib import Path

import pytest

from highlights.multiangle.scoreboard import (
    ABBR,
    _clock_text,
    apply_scoreboard,
    find_bold_font,
    score_timeline,
    scoreboard_filter,
)

OUT = Path("/tmp/scoreboard_test")


def test_abbr():
    assert ABBR("Home") == "HOM"
    assert ABBR("Away") == "AWA"
    assert ABBR("Arsenal") == "ARS"
    assert ABBR("Real Madrid") == "RM"
    assert ABBR("Manchester United FC") == "MUF"
    assert ABBR("FC Bayern München") == "FBM"


def test_score_timeline():
    tl = score_timeline([
        {"t": 30.0, "team": "away"},
        {"t": 10.0, "team": "home"},
        {"t": 60.0, "team": "home"},
        {"t": 75.0, "team": None},          # unassigned: no change
    ])
    assert tl == [(0.0, 0, 0), (10.0, 1, 0), (30.0, 1, 1), (60.0, 2, 1)]


def test_score_timeline_same_t_collapses():
    tl = score_timeline([{"t": 5, "team": "home"},
                         {"t": 5, "team": "away"}])
    assert tl == [(0.0, 0, 0), (5.0, 1, 1)]


def test_filter_structure():
    font = find_bold_font() or "DejaVuSans-Bold.ttf"
    vf = scoreboard_filter(
        [{"t": 4.0, "team": "home"}, {"t": 9.0, "team": "away"}],
        "G:nats", "Blue End", "#112233", "not-a-hex", 2.5, font, dur=12.0)
    # one enable interval per score state (3 states)
    assert vf.count("enable='between(t,") == 5   # 3 score + 2 goal flashes
    assert "between(t,0.000,4.000)" in vf
    assert "between(t,9.000,13.000)" in vf      # last state -> dur + 1
    assert "G\\:N" in vf                         # ':' escaped in label abbr
    assert "0x112233@0.9" in vf                  # valid hex -> colour
    assert "0xea580c@0.9" in vf                  # bad hex -> default away
    assert "max(0\\,t-2.500)" in vf              # kickoff clamp present
    assert "GOAL  BE" in vf                      # away flash uses abbr


def test_filter_without_replays_matches_legacy():
    font = find_bold_font() or "DejaVuSans-Bold.ttf"
    args = ([], "Home", "Away", None, None, 10.0, font)
    legacy = scoreboard_filter(*args, dur=200.0)
    assert scoreboard_filter(*args, dur=200.0, replays=None) == legacy
    assert scoreboard_filter(*args, dur=200.0, replays=[]) == legacy


def test_filter_clock_stays_on_live_time_across_replay():
    font = find_bold_font() or "DejaVuSans-Bold.ttf"
    vf = scoreboard_filter(
        [], "Home", "Away", None, None, 10.0, font, dur=200.0,
        replays=[{"t_out_start": 100.0, "t_out_end": 112.0}])

    assert f"text='{_clock_text(10.0)}'" in vf
    assert "enable='between(t,0.000,100.000)'" in vf
    assert "text='01\\:30'" in vf
    assert "enable='between(t,100.000,112.000)'" in vf
    assert f"text='{_clock_text(22.0)}'" in vf
    assert "enable='between(t,112.000,201.000)'" in vf


def test_filter_clock_bases_accumulate_across_replays():
    font = find_bold_font() or "DejaVuSans-Bold.ttf"
    vf = scoreboard_filter(
        [], "Home", "Away", None, None, 10.0, font, dur=200.0,
        replays=[
            {"t_out_start": 100.0, "t_out_end": 112.0},
            {"t_out_start": 150.0, "t_out_end": 162.0},
        ])

    assert f"text='{_clock_text(10.0)}'" in vf
    assert f"text='{_clock_text(22.0)}'" in vf
    assert f"text='{_clock_text(34.0)}'" in vf
    assert "text='01\\:30'" in vf
    assert "text='02\\:08'" in vf
    assert "enable='between(t,112.000,150.000)'" in vf
    assert "enable='between(t,162.000,201.000)'" in vf


def test_filter_clock_kickoff_inside_replay_uses_live_mapping():
    font = find_bold_font() or "DejaVuSans-Bold.ttf"
    vf = scoreboard_filter(
        [], "Home", "Away", None, None, 105.0, font, dur=200.0,
        replays=[{"t_out_start": 100.0, "t_out_end": 112.0}])

    assert "text='00\\:00'" in vf
    assert f"text='{_clock_text(117.0)}'" in vf


@pytest.mark.skipif(shutil.which("ffmpeg") is None or
                    shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not installed")
def test_apply_scoreboard_end_to_end():
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc2=duration=12:size=1920x1080:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True)
    font = find_bold_font()
    assert font, "no font found"
    vf = scoreboard_filter([{"t": 4.0, "team": "home"}],
                           "Home", "Away", None, None, 0.0, font, dur=12.0)
    fracs = []
    out = OUT / "scored.mp4"
    apply_scoreboard(src, out, vf, progress_cb=fracs.append,
                     log=lambda m: None)
    from highlights.pipeline.probe import probe
    dur = probe(out)["duration_s"]
    assert abs(dur - 12.0) < 0.5
    assert fracs and fracs[-1] == 1.0
    for t in (2, 8):
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(out),
             "-frames:v", "1", str(OUT / f"frame_t{t}.png")],
            check=True, capture_output=True)
