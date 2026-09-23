"""highlights — single-camera football highlight detection.

    highlights run <video|modal://football-footage/x.mp4> --out out/ [--pitch p.json] [--calib c.json]
                [--goal-zones-px z.json] [--start-minutes M] [--max-minutes N] [--local] [--reuse] [--no-clips]
    highlights frame <src> --time T --out f.jpg        # save a frame (for picking --goal-zones-px)
    highlights calibrate video.mp4 --pitch p.json --out calib.json
    highlights check-calib video.mp4 --pitch p.json [--calib c.json] [--goal-zones-px z.json] --out check.jpg
    highlights recombine --out out/ [--config c.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .calib import Space, calibrate, default_landmarks, render_check
from .config import Config

ZONES_HINT = ("tip: `highlights frame <video> --time T --out f.jpg`, pick 4 pixel corners around each goal mouth, "
              "and pass them as --goal-zones-px zones.json ({\"A\": [[u,v]x4], \"B\": [[u,v]x4]})")


def _parse_source(video: str, cfg: Config) -> tuple[str, bool]:
    """modal://<volume>/<path> -> (path_in_volume, True); else (path, False)."""
    if video.startswith("modal://"):
        rest = video[len("modal://"):]
        vol, _, path = rest.partition("/")
        if vol != cfg.source_volume:
            raise ValueError(f"volume {vol!r} not mounted; only {cfg.source_volume!r} is (Config.source_volume)")
        return path, True
    return video, False


def _warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def _decode_jpg(jpg: bytes) -> np.ndarray:
    import cv2

    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def cmd_frame(args) -> int:
    video, remote = _parse_source(args.video, Config.load(Path(args.config) if args.config else None))
    if remote:
        from .modal_app import app, read_frame_remote

        with app.run():
            Path(args.out).write_bytes(read_frame_remote.remote(video, args.time))
    else:
        import cv2

        from pitchworld.calibrate import read_frame

        cv2.imwrite(str(args.out), read_frame(Path(video), args.time))
    print(f"wrote {args.out}")
    return 0


def cmd_calibrate(args) -> int:
    import pitchworld.calib_tool as calib_tool
    from pitchworld.pitch import PitchModel

    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    video, remote = _parse_source(args.video, Config.load(Path(args.config) if args.config else None))
    if remote:
        from .modal_app import app, read_frame_remote

        with app.run():
            frame = _decode_jpg(read_frame_remote.remote(video, args.time))
        calib_tool.read_frame = lambda v, t=1.0: frame
    calib_tool.run_click_tool(Path(video), 0, pitch, Path(args.out),
                              args.landmarks or default_landmarks(pitch), frame_time=args.time)
    return 0


def cmd_check_calib(args) -> int:
    from pitchworld.pitch import PitchModel

    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    video, remote = _parse_source(args.video, Config.load(Path(args.config) if args.config else None))
    if remote:
        from .calib import _manual_entry, calibrate_frame, render_check_frame
        from .modal_app import app, read_frame_remote

        with app.run():
            frame = _decode_jpg(read_frame_remote.remote(video, args.time))
        res = calibrate_frame(frame, pitch, _manual_entry(Path(args.calib)) if args.calib else None,
                              Path(args.goal_zones_px) if args.goal_zones_px else None, auto=args.auto_calib)
        for w in res.warnings:
            print("note:", w, file=sys.stderr)
        render_check_frame(frame, res.cal, pitch, res.zones, Path(args.out))
    else:
        res = calibrate(Path(video), pitch, Path(args.calib) if args.calib else None,
                        frame_time=args.time, goal_zones_px=Path(args.goal_zones_px) if args.goal_zones_px else None,
                        auto=args.auto_calib)
        for w in res.warnings:
            print("note:", w, file=sys.stderr)
        render_check(Path(video), res.cal, pitch, res.zones, Path(args.out), frame_time=args.time)
    print(f"wrote {args.out} (space={res.space.value})")
    return 0


def _detections_to_frames(paths: list[Path]) -> dict:
    """Range detection JSONs -> global per-frame grid keyed by absolute frame index."""
    recs = []
    for p in paths:
        r = json.loads(p.read_text())
        r["_ns"] = int(p.stem.split("_")[1]) * 100000
        for fr in r["players"]:
            for d in fr:
                if d["id"] >= 0:
                    d["id"] += r["_ns"]
        recs.append(r)
    if not recs:
        return {"players": [], "ball": [], "fps_eff": None, "w": 0, "h": 0, "frame0": 0}
    fps = recs[0]["fps"]
    stride = recs[0]["stride"]
    base = min(r["frame0"] for r in recs)
    last = max((r["frame_idx"] or [r["frame0"]])[-1] for r in recs)
    n = (last - base) // stride + 1
    players, ball = [[] for _ in range(n)], [[] for _ in range(n)]
    for r in recs:
        for k, gi in enumerate(r["frame_idx"]):
            idx = (gi - base) // stride
            if 0 <= idx < n:
                players[idx], ball[idx] = r["players"][k], r["ball"][k]
    return {"players": players, "ball": ball, "fps_eff": fps / stride,
            "w": recs[0]["width"], "h": recs[0]["height"], "frame0": base}


def _signals(video, det: dict, res, pitch, cfg: Config, remote: bool, t_off: float,
             audio_cache: Path | None = None) -> tuple[dict, dict]:
    from .audio import audio_envelope, audio_spikes
    from .ball import ball_signals, link_ball
    from .players import PlayerSigCfg, player_signals

    fps_eff = det["fps_eff"] or 1.0
    w, h = det["w"], det["h"]
    diag: dict = {}

    track = link_ball(det["ball"], fps_eff, w, cfg.ball_max_jump_frac, cfg.ball_min_tracklet,
                      cfg.ball_interp_gap_s, cfg.ball_smooth_window)
    sig = ball_signals(track, res.zones, res.space, cfg.bin_s, cfg.v_shot_pitch, cal=res.cal,
                       lost_s=cfg.ball_lost_s, player_frames=det["players"],
                       v_shot_fb=cfg.v_shot_pixel, frame_h=h)
    diag["ball_seen"] = float(sig["ball_seen"].mean())

    pcfg = PlayerSigCfg(
        v_run=cfg.v_run_pitch,
        cluster_radius=cfg.cluster_radius_pitch,
        cluster_min_players=cfg.cluster_min_players,
        cluster_slow=cfg.cluster_slow_pitch,
        cluster_min_dur_s=cfg.cluster_min_dur_s,
        restart_half_width=cfg.restart_half_width,
        restart_centre_r=cfg.restart_centre_r)
    sig.update(player_signals(det["players"], fps_eff, res.cal, res.space, res.zones, w, h, pitch, pcfg, cfg.bin_s))

    if audio_cache and audio_cache.exists():
        env = json.loads(audio_cache.read_text())
        t_a, db = np.array(env["t"]), np.array(env["rms_db"])
    elif remote:
        from .modal_app import audio_envelope_remote

        env = audio_envelope_remote.remote(video, cfg.audio_hop_s)
        t_a, db = np.array(env["t"]), np.array(env["rms_db"])
        if audio_cache:
            audio_cache.write_text(json.dumps({"hop_s": cfg.audio_hop_s, "t": env["t"], "rms_db": env["rms_db"]}))
    else:
        t_a, db = audio_envelope(Path(video), cfg.audio_hop_s)
        if audio_cache:
            audio_cache.write_text(json.dumps({"hop_s": cfg.audio_hop_s,
                                               "t": t_a.tolist(), "rms_db": db.tolist()}))
    audio_bins = audio_spikes(t_a, db, cfg.bin_s, cfg.audio_baseline_win_s,
                              cfg.audio_thresh_db, cfg.audio_min_dur_s)
    # audio is over the whole file; slice to the processed window [t_off, t_off + window]
    n_bins = max(len(v) for v in sig.values())
    b0 = round(t_off / cfg.bin_s)
    sig["audio"] = audio_bins[b0:b0 + n_bins] if len(audio_bins) > b0 else np.zeros(n_bins)
    for k, v in list(sig.items()):
        if len(v) < n_bins:
            sig[k] = np.pad(v, (0, n_bins - len(v)))
    return sig, diag


def _run_ranges(video: str, remote: bool, start: float, end: float, cfg: Config,
                det_dir: Path, reuse: bool) -> list[Path]:
    """Detection over [start, end) cut into chunk_s ranges; returns cached JSON paths."""
    from .modal_app import detect_local, detect_range

    chunk = cfg.chunk_s
    t = start
    ranges = []
    while t < end - 0.05:
        ranges.append((t, min(chunk, end - t)))
        t += chunk
    det_paths = [det_dir / f"chunk_{i:03d}.json" for i in range(len(ranges))]
    todo = [(r, p) for r, p in zip(ranges, det_paths) if not (reuse and p.exists())]
    if not todo:
        return det_paths
    if remote:
        from .modal_app import detect_range_cpu

        fn = detect_range if cfg.gpu else detect_range_cpu
        results = fn.map([video] * len(todo), [r[0] for r, _ in todo],
                         [r[1] for r, _ in todo], [cfg.to_dict()] * len(todo))
    else:
        results = [detect_local(video, r[0], r[1], cfg.to_dict()) for r, _ in todo]
    for (_, p), res in zip(todo, results):
        p.write_text(json.dumps(res))
    return det_paths


def cmd_run(args) -> int:
    from pitchworld.pitch import PitchModel
    from pitchworld.sync import probe

    from .calib import _manual_entry, calibrate_frame
    from .chunks import extract_clip
    from .combine import combine
    from .modal_app import app, extract_clips_remote, probe_remote, read_frame_remote
    from .report import write_review_html

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(Path(args.config) if args.config else None)
    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    warnings: list[str] = []

    video, remote = _parse_source(args.video, cfg)
    if remote and args.local:
        raise ValueError("--local is only for local files")

    det_dir = out / "detections"
    det_dir.mkdir(exist_ok=True)

    def _work() -> tuple:
        """Everything that may need the Modal app context."""
        info = probe_remote.remote(video) if remote else probe(Path(video))
        t_start = (args.start_minutes or 0.0) * 60.0
        t_end = min(info["duration"], t_start + args.max_minutes * 60.0) if args.max_minutes else info["duration"]

        if remote:
            frame_jpg = out / "calib_frame.jpg"
            if not frame_jpg.exists():
                frame_jpg.write_bytes(read_frame_remote.remote(video, args.frame_time))
            frame = _decode_jpg(frame_jpg.read_bytes())
            res = calibrate_frame(frame, pitch, _manual_entry(Path(args.calib)) if args.calib else None,
                                  Path(args.goal_zones_px) if args.goal_zones_px else None,
                                  auto=args.auto_calib)
        else:
            res = calibrate(Path(video), pitch, Path(args.calib) if args.calib else None,
                            frame_time=args.frame_time,
                            goal_zones_px=Path(args.goal_zones_px) if args.goal_zones_px else None,
                            auto=args.auto_calib)
        det_paths = _run_ranges(video, remote, t_start, t_end, cfg, det_dir, args.reuse)
        return info, t_start, t_end, res, det_paths

    if remote:
        with app.run():
            info, t_start, t_end, res, det_paths = _work()
    else:
        info, t_start, t_end, res, det_paths = _work()

    warnings += res.warnings
    if res.space is Space.PIXEL:
        print("WARNING: calibration failed -> PIXEL space heuristics in use", file=sys.stderr)
        if not res.zones:
            _warn("no goal zones (need --goal-zones-px in pixel space); ball/attack signals will be zero", warnings)

    det = _detections_to_frames(det_paths)
    t_off = det["frame0"] / (det["fps_eff"] * cfg.stride) if det["players"] else t_start

    def _audio_and_signals() -> tuple[dict, dict]:
        if remote:
            with app.run():
                return _signals(video, det, res, pitch, cfg, remote, t_off, out / "audio_env.json")
        return _signals(video, det, res, pitch, cfg, remote, t_off, out / "audio_env.json")

    sig, diag = _audio_and_signals()
    np.savez(out / "signals.npz", **{k: np.asarray(v) for k, v in sig.items()})
    (out / "signal_meta.json").write_text(json.dumps({"bin_s": cfg.bin_s, "t_offset": t_off, **diag}))

    window_s = t_end - t_start
    cands, act_windows = combine(sig, cfg.bin_s, cfg, info["duration"], t_offset=t_off)
    if not args.no_clips:
        if remote:
            clips = [{"id": c.id, "start": c.start, "end": min(c.end, info["duration"])} for c in cands]
            with app.run():
                blob = extract_clips_remote.remote(video, clips) if clips else {}
            (out / "clips").mkdir(exist_ok=True)
            for c in cands:
                mm, ss = int(c.t_event // 60), int(c.t_event % 60)
                name = f"{c.id}_{c.type}_{mm}m{ss:02d}s.mp4"
                (out / "clips" / name).write_bytes(blob[c.id])
                c.clip = name
        else:
            for c in cands:
                mm, ss = int(c.t_event // 60), int(c.t_event % 60)
                name = f"{c.id}_{c.type}_{mm}m{ss:02d}s.mp4"
                extract_clip(Path(video), c.start, min(c.end, info["duration"]), out / "clips" / name)
                c.clip = name

    write_review_html(out, cands, act_windows)
    payload = {"video": args.video, "duration_s": info["duration"], "window": [t_start, t_end],
               "active_windows": act_windows,
               "zones_px": ({g: z.poly for g, z in res.zones.items()} if res.space is Space.PIXEL else None),
               "calibration": ({"method": res.cal.method, "confidence": res.cal.confidence,
                                "warnings": warnings, "full": res.cal.to_dict()} if res.cal else None),
               "space": res.space.value, "config": cfg.to_dict(), "diagnostics": diag,
               "warnings": warnings, "candidates": [c.to_dict() for c in cands]}
    (out / "candidates.json").write_text(json.dumps(payload, indent=2))
    if diag["ball_seen"] < 0.2:
        print(f"WARNING: ball_seen={diag['ball_seen']:.2f} (<0.2) — ball model unreliable on this footage",
              file=sys.stderr)
    print(f"space={res.space.value}  ball_seen={diag['ball_seen']:.2f}  candidates={len(cands)}  "
          f"window={window_s:.0f}s @ {t_off:.1f}s")
    print(f"{'rank':<5}{'type':<8}{'conf':<7}{'time':<8}{'goal':<5}signals")
    for c in cands:
        mm, ss = int(c.t_event // 60), int(c.t_event % 60)
        print(f"{c.rank:<5}{c.type:<8}{c.confidence:<7.2f}{mm}:{ss:02d}    {c.goal:<5}"
              + " ".join(f"{k}={v:.2f}" for k, v in c.signals.items()))
    if res.space is Space.PIXEL:
        print(ZONES_HINT)
    print(f"-> {out}/candidates.json, review.html, clips/")
    return 0


def _resignals(out: Path, payload: dict, cfg: Config) -> tuple[dict, dict]:
    """Rebuild signals from cached detection JSONs + cached audio envelope (no Modal detection)."""
    from pitchworld.calibrate import CameraCalibration
    from pitchworld.pitch import PitchModel

    from .calib import CalibResult, GoalZone, Space

    det = _detections_to_frames(sorted((out / "detections").glob("chunk_*.json")))
    space = Space(payload.get("space", "pixel"))
    cal = None
    if payload.get("calibration") and payload["calibration"].get("full"):
        cal = CameraCalibration(**payload["calibration"]["full"])
    if space is Space.PIXEL and payload.get("zones_px"):
        zones = {g: GoalZone(g, 1.0 if np.mean([p[0] for p in poly]) > det["w"] / 2 else -1.0,
                             float("nan"), poly) for g, poly in payload["zones_px"].items()}
    else:
        from .calib import goal_zones_pitch

        zones = goal_zones_pitch(PitchModel.standard())
    res = CalibResult(cal, zones, space, [])
    meta = json.loads((out / "signal_meta.json").read_text()) if (out / "signal_meta.json").exists() else {}
    t_off = meta.get("t_offset", det["frame0"] / (det["fps_eff"] * cfg.stride) if det["players"] else 0.0)
    video, remote = _parse_source(payload["video"], cfg)
    if remote and not (out / "audio_env.json").exists():
        from .modal_app import app

        with app.run():
            return _signals(video, det, res, PitchModel.standard(), cfg, remote, t_off, out / "audio_env.json")
    return _signals(video, det, res, PitchModel.standard(), cfg, remote, t_off, out / "audio_env.json")


def cmd_recombine(args) -> int:
    from .combine import combine
    from .report import write_review_html

    out = Path(args.out)
    cfg = Config.load(Path(args.config) if args.config else None)
    payload = json.loads((out / "candidates.json").read_text())
    if args.resignal:
        sig, diag = _resignals(out, payload, cfg)
        np.savez(out / "signals.npz", **{k: np.asarray(v) for k, v in sig.items()})
        meta = json.loads((out / "signal_meta.json").read_text()) if (out / "signal_meta.json").exists() else {}
        meta.update(diag)
        (out / "signal_meta.json").write_text(json.dumps(meta))
    else:
        sig = dict(np.load(out / "signals.npz"))
    meta = json.loads((out / "signal_meta.json").read_text()) if (out / "signal_meta.json").exists() else {}
    duration = payload["duration_s"]
    t_off = meta.get("t_offset", 0.0)
    cands, act_windows = combine(sig, cfg.bin_s, cfg, duration, t_offset=t_off)
    if not args.no_clips:
        video, remote = _parse_source(payload["video"], cfg)
        clips = [{"id": c.id, "start": c.start, "end": min(c.end, duration)} for c in cands]
        if remote:
            from .modal_app import app, extract_clips_remote

            with app.run():
                blob = extract_clips_remote.remote(video, clips) if clips else {}
            (out / "clips").mkdir(exist_ok=True)
            for c in cands:
                mm, ss = int(c.t_event // 60), int(c.t_event % 60)
                name = f"{c.id}_{c.type}_{mm}m{ss:02d}s.mp4"
                (out / "clips" / name).write_bytes(blob[c.id])
                c.clip = name
        else:
            from .chunks import extract_clip
            for c in cands:
                mm, ss = int(c.t_event // 60), int(c.t_event % 60)
                name = f"{c.id}_{c.type}_{mm}m{ss:02d}s.mp4"
                extract_clip(Path(video), c.start, min(c.end, duration), out / "clips" / name)
                c.clip = name
    write_review_html(out, cands, act_windows)
    payload["active_windows"] = act_windows
    payload["candidates"] = [c.to_dict() for c in cands]
    payload["config"] = cfg.to_dict()
    (out / "candidates.json").write_text(json.dumps(payload, indent=2))
    print(f"{len(cands)} candidates -> {out}/candidates.json")
    return 0


def cmd_reel(args) -> int:
    """Concatenate chosen clips chronologically into out/reel.mp4."""
    import subprocess
    import tempfile

    out = Path(args.out)
    payload = json.loads((out / "candidates.json").read_text())
    cands = payload.get("candidates", [])
    if args.ids:
        want = {f"c{int(i):02d}" for i in args.ids.split(",")}
        cands = [c for c in cands if c["id"] in want]
    else:
        cands = [c for c in cands if c["confidence"] >= args.min_conf]
        if args.top:
            cands = sorted(cands, key=lambda c: -c["confidence"])[: args.top]
    cands = [c for c in cands if c.get("clip")]
    cands.sort(key=lambda c: c["t_event"])
    if not cands:
        print("no clips selected", file=sys.stderr)
        return 1
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for c in cands:
            f.write(f"file '{(out / 'clips' / c['clip']).resolve()}'\n")
        list_path = f.name
    dst = out / "reel.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
         "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)],
        check=True)
    print(f"{len(cands)} clips -> {dst}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="highlights")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--pitch")
    p.add_argument("--calib")
    p.add_argument("--goal-zones-px")
    p.add_argument("--config")
    p.add_argument("--local", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--start-minutes", type=float)
    p.add_argument("--max-minutes", type=float)
    p.add_argument("--frame-time", type=float, default=1.0)
    p.add_argument("--auto-calib", action="store_true",
                   help="try line-based auto calibration first (off by default; spurious on ground-level footage)")
    p.add_argument("--no-clips", action="store_true")
    p.set_defaults(f=cmd_run)

    p = sub.add_parser("frame")
    p.add_argument("video")
    p.add_argument("--time", type=float, default=1.0)
    p.add_argument("--config")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_frame)

    p = sub.add_parser("calibrate")
    p.add_argument("video")
    p.add_argument("--pitch")
    p.add_argument("--config")
    p.add_argument("--out", required=True)
    p.add_argument("--landmarks", nargs="*")
    p.add_argument("--time", type=float, default=1.0)
    p.set_defaults(f=cmd_calibrate)

    p = sub.add_parser("check-calib")
    p.add_argument("video")
    p.add_argument("--pitch")
    p.add_argument("--config")
    p.add_argument("--calib")
    p.add_argument("--goal-zones-px")
    p.add_argument("--time", type=float, default=1.0)
    p.add_argument("--auto-calib", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_check_calib)

    p = sub.add_parser("recombine")
    p.add_argument("--out", required=True)
    p.add_argument("--pitch")
    p.add_argument("--config")
    p.add_argument("--resignal", action="store_true",
                   help="recompute signals from cached detections + audio_env.json (no detection calls)")
    p.add_argument("--no-clips", action="store_true")
    p.set_defaults(f=cmd_recombine)

    p = sub.add_parser("reel", help="concatenate selected clips into out/reel.mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=0, help="keep only the N highest-confidence clips")
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--ids", default="", help="comma-separated candidate ids, e.g. 1,3,5")
    p.set_defaults(f=cmd_reel)

    args = ap.parse_args()
    t0 = time.time()
    rc = args.f(args)
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
