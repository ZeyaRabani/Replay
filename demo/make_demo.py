#!/usr/bin/env python3
"""One-command demo video export for Replay.

    python demo/make_demo.py --recording rec.mp4 --clips cam0.mp4 cam1.mp4 cam2.mp4 \
        [--jumps jumps.json] [--narration voice.wav] [--tracking tracking.json] \
        [--out replay-demo.mp4]

Output timeline (all 1280x720 @ 25 fps, H.264 + AAC):
  1. title card   "Replay"  (black, white/yellow)                       ~3 s
  2. N-up grid    of the original synced camera angles                  ~6 s
  3. recording    of the viewer, captions burnt in at every camera jump
  4. outro card   fidelity disclaimer (from tracking.json["quality"])   ~5 s

Captions come from `jumps.json` (written by demo/record_viewer.py or the viewer)
or, when only narration is given, from a local WhisperX transcript.

Every step is plain ffmpeg. VEED OpenEdit (`.agents/skills/open-edit`) was
installed but its renderer refuses non-macOS/Windows hosts
("unsupported platform linux/x64"), so on Linux this script writes an
OpenEdit-style `.wv` composition next to the output (`--wv-out`) that a Mac can
render with `veed-engine-cli`, and does the actual cut with ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

W, H, FPS = 1280, 720, 25
FONT = next(
    (
        f
        for f in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        )
        if Path(f).exists()
    ),
    "",
)
YELLOW = "0xFFD500"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(path)]
    )
    d = json.loads(out)
    info = {"duration": float(d["format"]["duration"]), "audio": False, "w": 0, "h": 0}
    for s in d["streams"]:
        if s["codec_type"] == "audio":
            info["audio"] = True
        elif s["codec_type"] == "video":
            info["w"], info["h"] = int(s["width"]), int(s["height"])
    return info


def esc(text: str) -> str:
    """Escape for ffmpeg drawtext."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\\\\\'").replace("%", "%%")


def drawtext(text: str, size: int, y: str, color: str = "white", x: str = "(w-text_w)/2", alpha: str | None = None) -> str:
    s = f"drawtext=fontfile={FONT}:text='{esc(text)}':fontsize={size}:fontcolor={color}:x={x}:y={y}"
    if alpha:
        s += f":alpha='{alpha}'"
    return s


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-shortest"]
SILENCE = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]


def card(out: Path, dur: float, lines: list[tuple[str, int, str, int]]) -> None:
    """lines = (text, fontsize, color, y_px). Fades in/out."""
    vf = [f"color=c=black:s={W}x{H}:r={FPS}:d={dur}[bg]"]
    chain = "[bg]"
    for i, (text, size, color, y) in enumerate(lines):
        vf.append(f"{chain}{drawtext(text, size, str(y), color)}[t{i}]")
        chain = f"[t{i}]"
    vf.append(f"{chain}fade=t=in:d=0.5,fade=t=out:st={dur - 0.5}:d=0.5[v]")
    run(["ffmpeg", "-y", "-v", "error", *SILENCE, "-filter_complex", ";".join(vf), "-map", "[v]", "-map", "0:a",
         "-t", str(dur), *ENC, out])


def nup(out: Path, clips: list[Path], dur: float, start: float) -> None:
    n = len(clips)
    cols = 1 if n == 1 else (2 if n <= 4 else 3)
    if n == 3:
        cols = 3
    rows = math.ceil(n / cols)
    tw, th = W // cols, H // rows
    inputs: list[str] = []
    fc: list[str] = []
    for i, c in enumerate(clips):
        inputs += ["-ss", str(start), "-t", str(dur), "-i", str(c)]
        fc.append(
            f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},setsar=1,fps={FPS},"
            f"{drawtext(f'cam{i}', 36, '18', YELLOW, '18')}[c{i}]"
        )
    layout = "|".join(f"{(i % cols) * tw}_{(i // cols) * th}" for i in range(n))
    if n == 1:
        fc.append("[c0]copy[g0]")
    else:
        fc.append(f"{''.join(f'[c{i}]' for i in range(n))}xstack=inputs={n}:layout={layout}:fill=black[g0]")
    fc.append(f"[g0]pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,{drawtext(f'{n} synced camera angles -> one pitch frame', 40, 'h-70', 'white')},"
              f"fade=t=in:d=0.4,fade=t=out:st={dur - 0.4}:d=0.4[v]")
    run(["ffmpeg", "-y", "-v", "error", *inputs, *SILENCE, "-filter_complex", ";".join(fc), "-map", "[v]",
         "-map", f"{n}:a", "-t", str(dur), *ENC, out])


def load_jumps(path: Path | None) -> list[dict]:
    if not path:
        return []
    d = json.loads(path.read_text())
    items = d.get("jumps", d) if isinstance(d, dict) else d
    out = []
    for j in items:
        t = float(j.get("t", j.get("time", 0)))
        label = j.get("label") or j.get("target") or j.get("mode") or "jump"
        if str(label).startswith("player:"):
            label = "Player " + str(label).split(":", 1)[1]
        out.append({"t": t, "label": str(label)})
    return sorted(out, key=lambda j: j["t"])


def captions_from_whisperx(narration: Path, workdir: Path) -> list[dict]:
    """Run local WhisperX (CPU) and return [{t, end, label}] per segment."""
    wx = shutil.which("whisperx") or str(Path(__file__).resolve().parents[1] / ".venv-whisperx/bin/whisperx")
    if not Path(wx).exists():
        print("whisperx not found; skipping transcript captions (run demo/setup.sh)", file=sys.stderr)
        return []
    out_dir = workdir / "wx"
    run([wx, str(narration), "--model", "small", "--compute_type", "int8", "--device", "cpu",
         "--output_format", "json", "--output_dir", str(out_dir)])
    js = next(out_dir.glob("*.json"))
    d = json.loads(js.read_text())
    return [{"t": float(s["start"]), "end": float(s["end"]), "label": s["text"].strip()} for s in d["segments"]]


def ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(path: Path, caps: list[dict], jump_style: bool) -> None:
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Jump,DejaVu Sans,44,&H0000D5FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,3,0,2,40,40,48,1
Style: Talk,DejaVu Sans,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,40,40,48,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for i, c in enumerate(caps):
        end = c.get("end") or (caps[i + 1]["t"] if i + 1 < len(caps) else c["t"] + 3.0)
        end = min(end, c["t"] + 3.0) if jump_style else end
        text = ("\u25b6 CAMERA JUMP: " + c["label"]) if jump_style else c["label"]
        lines.append(f"Dialogue: 0,{ass_time(c['t'])},{ass_time(end)},{'Jump' if jump_style else 'Talk'},,0,0,0,,{{\\fad(150,250)}}{text}")
    path.write_text(hdr + "\n".join(lines) + "\n")


def recording_segment(out: Path, rec: Path, ass: Path | None, narration: Path | None, max_dur: float | None,
                      crop: str | None) -> float:
    info = ffprobe(rec)
    dur = info["duration"] if max_dur is None else min(info["duration"], max_dur)
    vf = (f"crop={crop}," if crop else "") + f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={FPS}"
    if ass:
        vf += f",ass={ass}"
    vf += f",{drawtext('Replay  |  explorable 3D replay', 22, '14', 'white', 'w-text_w-18')}:box=1:boxcolor=black@0.6:boxborderw=8"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(rec)]
    if narration:
        cmd += ["-i", str(narration), "-map", "0:v", "-map", "1:a"]
    elif info["audio"]:
        cmd += ["-map", "0:v", "-map", "0:a"]
    else:
        cmd += [*SILENCE, "-map", "0:v", "-map", "1:a"]
    cmd += ["-vf", vf, "-t", str(dur), *ENC, str(out)]
    run(cmd)
    return dur


def disclaimer_lines(tracking: Path | None) -> list[str]:
    lines = ["Fidelity disclaimer"]
    if tracking and tracking.exists():
        q = json.loads(tracking.read_text()).get("quality", {})
        dis = q.get("cross_camera_disagreement_m", {})
        worst = max(dis.values()) if dis else None
        if q.get("calibration_low_confidence"):
            lines.append("Camera calibration: LOW confidence (no full pitch-template fit)")
        if worst:
            lines.append(f"Cameras disagree on player positions by up to {worst:.1f} m")
        st = q.get("stats", {})
        if st:
            lines.append(f"{st.get('ids_seen_over_half', '?')} stable tracks from {st.get('unique_ids', '?')} raw IDs; "
                         f"{int(100 * st.get('frac_multi_camera', 0))} in 100 seen by more than one camera")
    else:
        lines.append("Positions are estimates from fixed-camera tracking, not ground truth")
    lines.append("Player identities and positions are approximate. Not for officiating.")
    return lines


def write_wv(path: Path, shots: list[dict], caps: list[dict]) -> None:
    """OpenEdit `.wv` (HTML+CSS) composition mirroring the ffmpeg cut, for a Mac render."""
    t0 = 0.0
    vids, cues = [], []
    for s in shots:
        vids.append(f'<video class="vid" src="{s["file"]}" style="animation-delay:{int(t0 * 1000)}ms;'
                    f'animation-duration:{int(s["dur"] * 1000)}ms"></video>')
        if s.get("caps"):
            for c in caps:
                cues.append(f'<div class="cue" style="animation-delay:{int((t0 + c["t"]) * 1000)}ms;animation-duration:3000ms">'
                            f'&#9654; CAMERA JUMP: {c["label"]}</div>')
        t0 += s["dur"]
    html = f"""<!-- OpenEdit .wv composition generated by demo/make_demo.py; render: veed-engine-cli <dir> --record -->
<style>
:root{{--W:{{W}};--H:{{H}}}}
body{{margin:0;width:{{W}}px;height:{{H}}px;background:#000;font-family:Inter,DejaVu Sans,sans-serif}}
.vid{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;animation:show 1ms both}}
.cue{{position:absolute;left:0;right:0;bottom:48px;text-align:center;color:#ffd500;font:700 44px/1.2 Inter;
  text-shadow:0 0 6px #000;opacity:0;animation:show 1ms both}}
@keyframes show{{from{{opacity:1}}to{{opacity:1}}}}
</style>
{chr(10).join(vids)}
{chr(10).join(cues)}
"""
    path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True, type=Path, help="screen recording of the viewer (MP4)")
    ap.add_argument("--clips", nargs="+", required=True, type=Path, help="original synced camera clips (N of them)")
    ap.add_argument("--jumps", type=Path, help="jumps.json sidecar with camera-jump timestamps (s into recording)")
    ap.add_argument("--narration", type=Path, help="optional narration audio; transcribed with WhisperX if no --jumps")
    ap.add_argument("--tracking", type=Path, help="tracking.json, used for the honest outro card")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "replay-demo.mp4")
    ap.add_argument("--wv-out", type=Path, help="also write an OpenEdit .wv composition here (default: next to --out)")
    ap.add_argument("--nup-seconds", type=float, default=6.0)
    ap.add_argument("--nup-start", type=float, default=2.0, help="offset into the clips for the N-up")
    ap.add_argument("--max-recording-seconds", type=float, default=None)
    ap.add_argument("--crop", help="ffmpeg crop w:h:x:y applied to the recording, e.g. 1600:1040:0:85 to drop browser chrome")
    ap.add_argument("--keep-temp", action="store_true")
    a = ap.parse_args()

    if not FONT:
        print("no TTF font found for drawtext", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="replay-demo-"))
    print("workdir", tmp)
    shots: list[dict] = []

    title = tmp / "00_title.mp4"
    card(title, 3.0, [("Replay", 150, "white", 250), ("multi-camera football moment  ->  explorable 3D replay", 36, YELLOW, 440)])
    shots.append({"file": str(title), "dur": 3.0})

    grid = tmp / "01_angles.mp4"
    nup(grid, a.clips, a.nup_seconds, a.nup_start)
    shots.append({"file": str(grid), "dur": a.nup_seconds})

    caps = load_jumps(a.jumps)
    jump_style = True
    if not caps and a.narration:
        caps = captions_from_whisperx(a.narration, tmp)
        jump_style = False
    ass = None
    if caps:
        ass = tmp / "captions.ass"
        write_ass(ass, caps, jump_style)
    rec = tmp / "02_recording.mp4"
    rec_dur = recording_segment(rec, a.recording, ass, a.narration, a.max_recording_seconds, a.crop)
    shots.append({"file": str(rec), "dur": rec_dur, "caps": True})

    outro = tmp / "03_outro.mp4"
    lines = disclaimer_lines(a.tracking)
    spec = [(lines[0], 64, YELLOW, 150)] + [(ln, 28, "white", 280 + 60 * i) for i, ln in enumerate(lines[1:])]
    spec.append(("github.com/ZeyaRabani/Replay", 26, "white@0.6", H - 80))
    card(outro, 5.0, spec)
    shots.append({"file": str(outro), "dur": 5.0})

    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{s['file']}'\n" for s in shots))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out])

    wv = a.wv_out or a.out.with_suffix(".wv.html")
    write_wv(wv, shots, caps if jump_style else [])
    (a.out.with_suffix(".shots.json")).write_text(json.dumps({"shots": shots, "captions": caps}, indent=2))

    total = sum(s["dur"] for s in shots)
    print(f"\nwrote {a.out}  ({total:.1f} s, {len(caps)} captions)\nwrote {wv}")
    if not a.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    sys.exit(main())
