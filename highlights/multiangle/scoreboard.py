"""Burn a persistent scoreboard + match clock into a rendered cut.

The overlay is one -vf filtergraph (drawbox + drawtext) for a 1920x1080
frame, top-left at x=48, y=40, height 64:

    [team colour block: ABBR home] [black: "H - A"] [team colour: ABBR away]
    [black: mm:ss clock]

The score text swaps once per score state via `enable='between(t,a,b)'`.
The clock uses drawtext `eif` for live play and static text during replays.
Each goal flashes "GOAL  <ABBR>" beneath the bar for 6 s.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

BAR_X = 48
BAR_Y = 40
BAR_H = 64
TEAM_W = 120          # bold 3-letter abbr at fontsize 40 is ~90 px + pad
SCORE_W = 110
CLOCK_W = 150
FONTSIZE = 40
FLASH_S = 6.0
FLASH_Y = BAR_Y + BAR_H + 12

HOME_DEFAULT = "2563eb"
AWAY_DEFAULT = "ea580c"

BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_bold_font() -> str | None:
    """DejaVuSans-Bold preferred, then the shared FONT_CANDIDATES list."""
    for p in BOLD_CANDIDATES:
        if Path(p).exists():
            return p
    try:
        from highlights.app.backend.ffmpeg import find_font
        return find_font()
    except Exception:
        return None


def ABBR(label: str) -> str:
    """3-letter uppercase abbreviation: initials when the label has >=2
    words ('Real Madrid' -> 'RM'), else the first 3 letters ('Home' ->
    'HOM')."""
    words = [w for w in str(label).split() if w]
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).upper()
    return str(label)[:3].upper()


def score_timeline(goals: list[dict]) -> list[tuple[float, int, int]]:
    """goals = [{"t": float, "team": "home"|"away"}] -> cumulative score
    states [(t_from, home, away)] starting with (0.0, 0, 0), sorted.
    Goals at the same t collapse into one state change."""
    home = away = 0
    out: list[tuple[float, int, int]] = [(0.0, 0, 0)]
    for g in sorted(goals, key=lambda g: float(g.get("t", 0.0))):
        t = float(g.get("t", 0.0))
        if g.get("team") == "home":
            home += 1
        elif g.get("team") == "away":
            away += 1
        else:
            continue
        if out[-1][0] == t:
            out[-1] = (t, home, away)
        else:
            out.append((t, home, away))
    return out


def _esc(text: str) -> str:
    """Escape a drawtext string the way _overlay_filter does."""
    return (str(text).replace("\\", "\\\\").replace(":", "\\:")
            .replace("'", ""))


def _colour(hexv: str | None, default: str, alpha: float = 0.9) -> str:
    """'#rrggbb' -> '0xrrggbb@alpha'; anything else falls back."""
    h = str(hexv or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        h = default
    return f"0x{h}@{alpha}"


def _clock_text(base: float) -> str:
    """mm:ss of max(0, t-base) as a drawtext expansion string."""
    k = f"{float(base):.3f}"
    return (r"%{eif\:trunc(max(0\,t-" + k + r")/60)\:d\:2}\:"
            r"%{eif\:mod(trunc(max(0\,t-" + k + r"))\,60)\:d\:2}")


def scoreboard_filter(goals: list[dict], home_label: str, away_label: str,
                      home_hex: str | None, away_hex: str | None,
                      kickoff: float, font: str, dur: float,
                      replays: list[dict] | None = None) -> str:
    """-vf filtergraph for a 1920x1080 frame."""
    f = f"fontfile={font}"
    fs = f"fontsize={FONTSIZE}"
    filters = [
        # home block
        f"drawbox=x={BAR_X}:y={BAR_Y}:w={TEAM_W}:h={BAR_H}:"
        f"color={_colour(home_hex, HOME_DEFAULT)}:t=fill",
        f"drawtext={f}:text='{_esc(ABBR(home_label))}':{fs}:fontcolor=white:"
        f"x={BAR_X}+({TEAM_W}-text_w)/2:y={BAR_Y}+({BAR_H}-text_h)/2",
        # score block
        f"drawbox=x={BAR_X + TEAM_W}:y={BAR_Y}:w={SCORE_W}:h={BAR_H}:"
        "color=black@0.8:t=fill",
        # away block
        f"drawbox=x={BAR_X + TEAM_W + SCORE_W}:y={BAR_Y}:w={TEAM_W}:h={BAR_H}:"
        f"color={_colour(away_hex, AWAY_DEFAULT)}:t=fill",
        f"drawtext={f}:text='{_esc(ABBR(away_label))}':{fs}:fontcolor=white:"
        f"x={BAR_X + TEAM_W + SCORE_W}+({TEAM_W}-text_w)/2:"
        f"y={BAR_Y}+({BAR_H}-text_h)/2",
        # clock block
        f"drawbox=x={BAR_X + 2 * TEAM_W + SCORE_W}:y={BAR_Y}:w={CLOCK_W}:"
        f"h={BAR_H}:color=black@0.6:t=fill",
    ]

    def add_clock_drawtext(
            text: str, interval: tuple[float, float] | None = None) -> None:
        clock_filter = (
            f"drawtext={f}:text='{text}':{fs}:fontcolor=white:"
            f"x={BAR_X + 2 * TEAM_W + SCORE_W}+({CLOCK_W}-text_w)/2:"
            f"y={BAR_Y}+({BAR_H}-text_h)/2")
        if interval is not None:
            clock_filter += (
                f":enable='between(t,{interval[0]:.3f},{interval[1]:.3f})'")
        filters.append(clock_filter)

    replay_intervals = []
    for replay in replays or []:
        try:
            start = float(replay["t_out_start"])
            end = float(replay["t_out_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < 0 or start > float(dur) or end <= start:
            continue
        replay_intervals.append((start, end))
    replay_intervals.sort()

    if not replay_intervals:
        add_clock_drawtext(_clock_text(kickoff))
    else:
        def live(output_time: float) -> float:
            return output_time - sum(
                end - start for start, end in replay_intervals
                if end <= output_time)

        live_kickoff = live(float(kickoff))
        cursor = 0.0
        replay_duration = 0.0
        for start, end in replay_intervals:
            if start > cursor:
                add_clock_drawtext(
                    _clock_text(live_kickoff + replay_duration),
                    (cursor, start))
            frozen_s = int(max(0.0, live(start) - live_kickoff))
            frozen_text = f"{frozen_s // 60:02d}:{frozen_s % 60:02d}"
            add_clock_drawtext(_esc(frozen_text), (start, end))
            replay_duration += end - start
            cursor = end
        if cursor < float(dur) + 1.0:
            add_clock_drawtext(
                _clock_text(live_kickoff + replay_duration),
                (cursor, float(dur) + 1.0))

    timeline = score_timeline(goals)
    for i, (t_from, h, a) in enumerate(timeline):
        t_to = (timeline[i + 1][0] if i + 1 < len(timeline)
                else float(dur) + 1.0)
        filters.append(
            f"drawtext={f}:text='{h} - {a}':{fs}:fontcolor=white:"
            f"x={BAR_X + TEAM_W}+({SCORE_W}-text_w)/2:"
            f"y={BAR_Y}+({BAR_H}-text_h)/2:"
            f"enable='between(t,{t_from:.3f},{t_to:.3f})'")
    for g in sorted(goals, key=lambda g: float(g.get("t", 0.0))):
        t = float(g.get("t", 0.0))
        label = home_label if g.get("team") == "home" else away_label
        filters.append(
            f"drawtext={f}:text='{_esc('GOAL  ' + ABBR(label))}':"
            "fontsize=36:fontcolor=yellow:"
            f"x={BAR_X}:y={FLASH_Y}:box=1:boxcolor=black@0.6:boxborderw=8:"
            f"enable='between(t,{t:.3f},{t + FLASH_S:.3f})'")
    return ",".join(filters)


def apply_scoreboard(src: Path, out: Path, vf: str, *,
                     progress_cb=None, log=print) -> Path:
    """Re-encode src with the scoreboard vf, writing to a tmp next to out
    then os.replace (src == out is supported)."""
    from highlights.pipeline.probe import probe as ffprobe
    src, out = Path(src), Path(out)
    try:
        dur = float(ffprobe(src).get("duration_s") or 0.0)
    except Exception:
        dur = 0.0
    tmp = out.with_name(out.stem + ".scoreboard.tmp.mp4")
    tmp.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
           "-progress", "pipe:1", "-nostats", str(tmp)]
    log(f"scoreboard: encoding {src.name}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms=") and dur > 0 and progress_cb:
            try:
                ms = float(line.split("=", 1)[1])
                progress_cb(min(1.0, ms / 1e6 / dur))
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"scoreboard ffmpeg exited {proc.returncode}")
    os.replace(tmp, out)
    if progress_cb:
        progress_cb(1.0)
    return out
