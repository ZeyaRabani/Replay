"""Demo-asset builder: turn a pitchworld output dir into VEED OpenEdit intermediates + a shot list.

    python -m pitchworld.demo_assets out/ [--demo out/demo] [--highlight START END] [--no-preview]

Renders, fully offline via ffmpeg: title cards, 2x2 camera grids, a titled pitch-map
render and a stats card from tracking.json quality fields; writes shot_list.json,
a concatenated demo_preview.mp4, and a stub VEED OpenEdit request (veed_request.json
+ veed_composition.html). No network calls are made.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)


def _esc(text: str) -> str:
    s = str(text).replace("\\", "\\\\")
    for ch in (":", "'", ",", "%"):
        s = s.replace(ch, "\\" + ch)
    return s


def _dt(text: str, size: int, x: str, y: str, color: str = "white", box: bool = False) -> str:
    f = f"drawtext=text='{_esc(text)}':fontsize={size}:x={x}:y={y}:fontcolor={color}"
    f += f":fontfile={FONT}" if FONT else ":font=Sans"
    if box:
        f += ":box=1:boxcolor=black@0.45:boxborderw=12"
    return f


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:3])} ...):\n{r.stderr[-2000:]}")


def _probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    dur = float(v.get("duration") or j.get("format", {}).get("duration") or 0.0)
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "duration": dur, "fps": fps,
            "has_audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


def load_output(out_dir: Path) -> dict:
    warnings: list[str] = []
    tracking = None
    tj = out_dir / "tracking.json"
    if tj.exists():
        tracking = json.loads(tj.read_text())
    else:
        warnings.append(f"missing {tj}")
    synced = sorted((out_dir / "synced").glob("cam*.mp4")) if (out_dir / "synced").is_dir() else []
    if not synced:
        warnings.append("no synced/cam*.mp4 clips found")
    pitch_map = out_dir / "debug" / "pitch_map.mp4"
    if not pitch_map.exists():
        warnings.append(f"missing {pitch_map}")
        pitch_map = None
    overlays = sorted((out_dir / "debug").glob("overlay_cam*.mp4")) if (out_dir / "debug").is_dir() else []
    return {"tracking": tracking, "synced": synced, "pitch_map": pitch_map,
            "overlays": overlays, "quality": (tracking or {}).get("quality") or {},
            "warnings": warnings}


def _av_out(out: Path) -> list[str]:
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            "-y", str(out)]


def _fit(text: str, size: int, w: int) -> int:
    # rough DejaVu width estimate (~0.62 em per char) to keep text inside the frame
    return min(size, max(18, int((w - 80) / max(1, 0.62 * len(text)))))


def render_title_card(out: Path, title: str, subtitle: str = "", dur: float = 3.0,
                      size: tuple[int, int] = (1280, 720), fps: int = 25, bg: str = "0x0b1f14") -> Path:
    w, h = size
    filters = [_dt(title, _fit(title, 72, w), "(w-text_w)/2", "(h-text_h)/2 - 20")]
    if subtitle:
        filters.append(_dt(subtitle, _fit(subtitle, 36, w), "(w-text_w)/2", "(h-text_h)/2 + 70"))
    filters.append(_dt("Replay · pitchworld", 22, "(w-text_w)/2", "h - 60", color="white@0.5"))
    _run(["ffmpeg", "-f", "lavfi", "-i", f"color=c={bg}:s={w}x{h}:r={fps}:d={dur}",
          "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
          "-vf", ",".join(filters), "-t", str(dur), "-shortest", *_av_out(out)])
    return out


def _tile(i: int, label: str, fps: int) -> str:
    return (f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,"
            f"pad=640:360:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1,"
            + _dt(label, 26, "12", "10", box=True) + f"[t{i}]")


def render_grid(clips: list[Path], out: Path, size: tuple[int, int] = (1280, 720), fps: int = 25,
                labels: list[str] | None = None, extra_tile: Path | None = None,
                start: float = 0.0, dur: float | None = None) -> Path:
    w, h = size
    labels = list(labels) if labels else [f"cam{i}" for i in range(len(clips))]
    cmd = ["ffmpeg"]
    for c in clips:
        if start:
            cmd += ["-ss", str(start)]
        if dur:
            cmd += ["-t", str(dur)]
        cmd += ["-i", str(c)]
    probe0 = _probe(clips[0])
    have_audio = probe0["has_audio"]
    limit = dur if dur else probe0["duration"]
    tiles = [_tile(i, labels[i], fps) for i in range(len(clips))]
    next_in = len(clips)
    n = len(clips)
    if n >= 3:
        if extra_tile:
            cmd += ["-i", str(extra_tile)]
            tiles.append(_tile(next_in, "pitch map", fps))
        else:
            cmd += ["-f", "lavfi", "-i", f"color=c=black:s=640x360:r={fps}:d={limit}"]
            tiles.append(f"[{next_in}:v]fps={fps},setsar=1[t{next_in}]")
        n = 4
    if not have_audio:
        cmd += ["-f", "lavfi", "-t", str(limit), "-i", "anullsrc=r=44100:cl=stereo"]
        audio_map = f"{next_in + 1}:a"
    else:
        audio_map = "0:a"
    if n == 2:
        stack = "[t0][t1]hstack=inputs=2[row]"
        post = f"[row]pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[v]"
    else:
        stack = "[t0][t1][t2][t3]xstack=inputs=4:layout=0_0|640_0|0_360|640_360[v]"
        post = None
    fc = ";".join(tiles + [stack] + ([post] if post else []))
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", audio_map, "-shortest", *_av_out(out)]
    _run(cmd)
    return out


def render_pitch_map_titled(pitch_map: Path, out: Path, title: str = "Fused pitch-coordinate tracking",
                            subtitle: str = "", size: tuple[int, int] = (1280, 720), fps: int = 25,
                            start: float = 0.0, dur: float | None = None) -> Path:
    w, h = size
    filters = [f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1",
               _dt(title, 34, "(w-text_w)/2", "18", box=True)]
    if subtitle:
        filters.append(_dt(subtitle, 24, "(w-text_w)/2", "18 + 52", color="white@0.85"))
    cmd = ["ffmpeg"]
    if start:
        cmd += ["-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(pitch_map), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-vf", ",".join(filters), "-shortest", *_av_out(out)]
    _run(cmd)
    return out


def _fmt_disagreement(v) -> str:
    vals = []
    if isinstance(v, dict):
        vals = [x for x in v.values() if isinstance(x, (int, float))]
    elif isinstance(v, (int, float)):
        vals = [v]
    if not vals:
        return "cross-camera disagreement: n/a"
    if len(vals) == 1:
        return f"cross-camera disagreement: {vals[0]:.2f} m median"
    return f"cross-camera disagreement: {min(vals):.1f}-{max(vals):.1f} m median"


def render_stats_card(quality: dict, tracking_meta: dict, out: Path, dur: float = 5.0,
                      size: tuple[int, int] = (1280, 720), fps: int = 25) -> Path:
    w, h = size
    stats = quality.get("stats") or {}
    lines: list[tuple[str, str]] = [("Pipeline quality (honest)", "white")]
    if stats:
        lines.append((f"unique player IDs: {stats.get('unique_ids', 'n/a')}"
                      f"   mean players/frame: {stats.get('mean_players_per_frame', 'n/a')}", "white"))
        lines.append((f"fraction tracked by 2+ cameras: {stats.get('frac_multi_camera', 'n/a')}"
                      f"   IDs seen > half the clip: {stats.get('ids_seen_over_half', 'n/a')}", "white"))
    lines.append((_fmt_disagreement(quality.get("cross_camera_disagreement_m")), "white"))
    nf = tracking_meta.get("num_frames")
    tfps = tracking_meta.get("fps")
    dur_s = tracking_meta.get("duration_s")
    ncams = len(tracking_meta.get("cameras") or [])
    lines.append((f"{nf or 'n/a'} frames at {tfps or 'n/a'} fps ({dur_s or 'n/a'} s) - {ncams} cameras", "white@0.8"))
    lines.append((f"sync: {'LOW confidence' if quality.get('sync_low_confidence') else 'OK'}",
                  "0xffa500" if quality.get("sync_low_confidence") else "0x3ddc84"))
    lines.append((f"calibration: {'LOW confidence' if quality.get('calibration_low_confidence') else 'OK'}",
                  "0xffa500" if quality.get("calibration_low_confidence") else "0x3ddc84"))
    for warn in (quality.get("warnings") or [])[:3]:
        warn = warn if len(warn) <= 90 else warn[:87] + "..."
        lines.append(("! " + warn, "0xffa500", 22))
    filters = []
    y = 90
    for i, line in enumerate(lines):
        text, color = line[0], line[1]
        fsize = line[2] if len(line) > 2 else (46 if i == 0 else 26)
        filters.append(_dt(text, fsize, "(w-text_w)/2" if i == 0 else "80", str(y), color=color))
        y += 52
    filters.append(_dt("Replay · pitchworld", 22, "(w-text_w)/2", "h - 60", color="white@0.5"))
    _run(["ffmpeg", "-f", "lavfi", "-i", f"color=c=0x0b1f14:s={w}x{h}:r={fps}:d={dur}",
          "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
          "-vf", ",".join(filters), "-t", str(dur), "-shortest", *_av_out(out)])
    return out


def render_preview(shot_list: dict, demo_dir: Path) -> Path:
    files = [demo_dir / s["file"] for s in shot_list["shots"]]
    out = demo_dir / "demo_preview.mp4"
    lst = demo_dir / "_concat.txt"
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    r = subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-y", str(out)], capture_output=True, text=True)
    if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
        lst.unlink()
        return out
    cmd = ["ffmpeg"]
    for f in files:
        cmd += ["-i", str(f)]
    ins = "".join(f"[{i}:v][{i}:a]" for i in range(len(files)))
    cmd += ["-filter_complex", f"{ins}concat=n={len(files)}:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]", *_av_out(out)]
    _run(cmd)
    lst.unlink()
    return out


def build_shot_list(out_dir: Path, demo_dir: Path, fps: int = 25, size: tuple[int, int] = (1280, 720),
                    highlight: tuple[float, float] | None = None) -> dict:
    out_dir = Path(out_dir)
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    src = load_output(out_dir)
    warnings = list(src["warnings"])
    tracking = src["tracking"] or {}
    quality = src["quality"]

    if highlight is None:
        durs = [_probe(c)["duration"] for c in src["synced"]] or [8.0]
        mind = min(durs)
        start = max(0.0, (mind - 8.0) / 2)
        highlight = (start, min(mind, start + 8.0))
    hstart, hend = highlight
    hdur = max(0.1, hend - hstart)

    cams = tracking.get("cameras") or []
    offsets = {c.get("index"): c.get("sync_offset_s") for c in cams}

    def offset_label(i: int) -> str:
        off = offsets.get(i)
        return f"cam{i}  {off:+.3f} s" if isinstance(off, (int, float)) else f"cam{i}"

    shots: list[dict] = []

    def add(shot_id: str, kind: str, path: Path, duration: float, title: str, caption: str, notes: str = ""):
        shots.append({"id": shot_id, "kind": kind, "file": path.name, "duration_s": round(duration, 3),
                      "title": title, "caption": caption, "notes": notes})

    add("01-title", "title_card",
        render_title_card(demo_dir / "01_title.mp4", "Replay",
                          "One football moment. Every angle. Any camera.", dur=3.0, size=size, fps=fps),
        3.0, "Replay", "One football moment. Every angle. Any camera.")

    if src["synced"]:
        cap = "Synced by audio - offsets from tracking.json cameras[*].sync_offset_s: " + ", ".join(
            f"cam{i} {offsets.get(i):+.3f} s" if isinstance(offsets.get(i), (int, float)) else f"cam{i}"
            for i in range(len(src["synced"])))
        add("02-synced-grid", "grid",
            render_grid(src["synced"], demo_dir / "02_synced_grid.mp4", size=size, fps=fps,
                        labels=[offset_label(i) for i in range(len(src["synced"]))],
                        start=hstart, dur=hdur),
            hdur, "Synced camera angles", cap)

    add("03-title-pipeline", "title_card",
        render_title_card(demo_dir / "03_pipeline.mp4", "Calibrate -> track -> fuse",
                          "YOLOv8 + ByteTrack on Modal - homography/pose fit per camera",
                          dur=3.0, size=size, fps=fps),
        3.0, "Calibrate -> track -> fuse", "YOLOv8 + ByteTrack on Modal - homography/pose fit per camera")

    if src["overlays"]:
        add("04-overlay-grid", "grid",
            render_grid(src["overlays"][:4], demo_dir / "04_overlay_grid.mp4", size=size, fps=fps,
                        labels=[f"overlay cam{i}" for i in range(len(src["overlays"][:4]))],
                        start=hstart, dur=hdur),
            hdur, "Tracking overlays", "Per-camera detections, IDs and calibration overlays")
    else:
        warnings.append("no overlay_cam*.mp4 found - overlay shot skipped")

    if src["pitch_map"]:
        add("05-pitch-map", "pitch_map",
            render_pitch_map_titled(src["pitch_map"], demo_dir / "05_pitch_map.mp4",
                                    subtitle="player positions in pitch coordinates",
                                    size=size, fps=fps, start=hstart, dur=hdur),
            hdur, "Fused pitch-coordinate tracking", "player positions in pitch coordinates")
    else:
        warnings.append("no pitch_map.mp4 found - pitch-map shot skipped")

    add("06-stats", "stats_card",
        render_stats_card(quality, tracking, demo_dir / "06_stats.mp4", dur=5.0, size=size, fps=fps),
        5.0, "Pipeline quality (honest)", "tracking.json quality flags and stats")

    add("07-title-next", "title_card",
        render_title_card(demo_dir / "07_next.mp4", "Stage 2: Reactor world model -> Stage 3: 3D viewer",
                          "tracking.json is the handoff", dur=3.0, size=size, fps=fps),
        3.0, "Stage 2/3", "tracking.json is the handoff")

    shot_list = {"version": 1, "fps": fps, "size": list(size), "source_out_dir": str(out_dir.resolve()),
                 "highlight": [round(hstart, 3), round(hend, 3)],
                 "quality_flags": {k: quality.get(k) for k in
                                   ("sync_low_confidence", "calibration_low_confidence")},
                 "shots": shots, "warnings": warnings}
    (demo_dir / "shot_list.json").write_text(json.dumps(shot_list, indent=2))
    shot_list["preview"] = render_preview(shot_list, demo_dir).name
    (demo_dir / "shot_list.json").write_text(json.dumps(shot_list, indent=2))
    return shot_list


def submit_to_veed(shot_list: dict, demo_dir: Path, dry_run: bool = True) -> Path:
    if not dry_run:
        raise NotImplementedError("OpenEdit runs as a local agent skill on macOS; see docs/stage4_veed.md")
    demo_dir = Path(demo_dir)
    w, h = shot_list["size"]
    fps = shot_list["fps"]
    shots = shot_list["shots"]
    total_ms = 0.0
    videos = []
    cues = []
    for s in shots:
        start_ms = total_ms
        total_ms += s["duration_s"] * 1000
        delay = f"animation-delay:{start_ms:.0f}ms;"
        videos.append(
            f'  <video class="vid" src="{s["file"]}" '
            f'style="position:absolute;left:0;top:0;width:{w}px;height:{h}px;opacity:0;'
            f'animation:show 1ms linear forwards;{delay}" muted></video>')
        cues.append(
            f'  <div class="cue" style="position:absolute;left:40px;bottom:36px;opacity:0;'
            f'animation:show 1ms linear forwards;{delay}">'
            f'{s["caption"] or s["title"]}</div>')
    html = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { margin:0; width:{W}px; height:{H}px; background:#000; position:relative; overflow:hidden; }
  .vid { object-fit:contain; }
  .cue { color:#fff; font:24px sans-serif; text-shadow:0 1px 3px #000; }
  @keyframes show { to { opacity:1; } }
</style></head><body>
<!-- OpenEdit-style composition: {W}x{H} @{FPS}fps, {DUR}ms total -->
""" + "\n".join(videos) + "\n" + "\n".join(cues) + "\n</body></html>\n"
    html = (html.replace("{W}", str(w)).replace("{H}", str(h))
            .replace("{FPS}", str(fps)).replace("{DUR}", str(int(total_ms))))
    comp = demo_dir / "veed_composition.html"
    comp.write_text(html)
    brief = ("Assemble a 16:9 1280x720 demo reel from the provided shots in order. "
             "Each shot lists its file, duration and a caption to show as a lower-third overlay. "
             f"Highlight window: {shot_list['highlight'][0]}s-{shot_list['highlight'][1]}s. "
             "Keep cuts on shot boundaries; no transitions needed.")
    req = {"tool": "veed-open-edit", "mode": "dry_run", "brief": brief, "shots": shots,
           "canvas": {"w": w, "h": h, "fps": fps}, "composition_html": str(comp.resolve())}
    req_path = demo_dir / "veed_request.json"
    req_path.write_text(json.dumps(req, indent=2))
    return req_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pitchworld.demo_assets",
                                 description="Render VEED OpenEdit intermediates + shot list from a pitchworld output dir")
    ap.add_argument("out_dir", type=Path, help="pitchworld output dir (synced/, debug/, tracking.json)")
    ap.add_argument("--demo", type=Path, default=None, help="demo output dir (default OUT_DIR/demo)")
    ap.add_argument("--highlight", type=float, nargs=2, metavar=("START", "END"), default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=str, default="1280x720")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)
    w, _, h = args.size.partition("x")
    size = (int(w), int(h))
    demo_dir = args.demo or (args.out_dir / "demo")
    shot_list = build_shot_list(args.out_dir, demo_dir, fps=args.fps, size=size,
                                highlight=tuple(args.highlight) if args.highlight else None)
    if args.no_preview and "preview" in shot_list:
        (demo_dir / shot_list["preview"]).unlink(missing_ok=True)
        shot_list.pop("preview")
        (demo_dir / "shot_list.json").write_text(json.dumps(shot_list, indent=2))
    for s in shot_list["shots"]:
        print(f"  {s['id']:<20} {s['kind']:<12} {s['duration_s']:>5.1f}s  {s['file']}")
    print(f"shot list -> {demo_dir / 'shot_list.json'}")
    if "preview" in shot_list:
        print(f"preview   -> {demo_dir / shot_list['preview']}")
    print(f"veed req  -> {submit_to_veed(shot_list, demo_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
