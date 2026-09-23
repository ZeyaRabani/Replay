"""highlights — single-camera football highlight detection.

    highlights run video.mp4 --out out/ [--pitch p.json] [--calib c.json]
                [--goal-zones-px z.json] [--local] [--reuse] [--no-clips]
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


def cmd_calibrate(args) -> int:
    import pitchworld.calib_tool as calib_tool
    from pitchworld.pitch import PitchModel

    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    video, remote = _parse_source(args.video, Config.load(Path(args.config) if args.config else None))
    if remote:
        import cv2

        from .modal_app import read_frame_remote

        jpg = read_frame_remote.remote(video, args.time)
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        calib_tool.read_frame = lambda v, t=1.0: frame
    calib_tool.run_click_tool(Path(video), 0, pitch, Path(args.out),
                              args.landmarks or default_landmarks(pitch), frame_time=args.time)
    return 0


def cmd_check_calib(args) -> int:
    from pitchworld.pitch import PitchModel

    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    video, remote = _parse_source(args.video, Config.load(Path(args.config) if args.config else None))
    if remote:
        import cv2

        from .calib import _manual_entry, calibrate_frame, render_check_frame
        from .modal_app import read_frame_remote

        frame = cv2.imdecode(np.frombuffer(read_frame_remote.remote(video, args.time), np.uint8),
                             cv2.IMREAD_COLOR)
        res = calibrate_frame(frame, pitch, _manual_entry(Path(args.calib)) if args.calib else None,
                              Path(args.goal_zones_px) if args.goal_zones_px else None)
        for w in res.warnings:
            print("note:", w, file=sys.stderr)
        render_check_frame(frame, res.cal, pitch, res.zones, Path(args.out))
    else:
        res = calibrate(Path(video), pitch, Path(args.calib) if args.calib else None,
                        frame_time=args.time, goal_zones_px=Path(args.goal_zones_px) if args.goal_zones_px else None)
        for w in res.warnings:
            print("note:", w, file=sys.stderr)
        render_check(Path(video), res.cal, pitch, res.zones, Path(args.out), frame_time=args.time)
    print(f"wrote {args.out} (space={res.space.value})")
    return 0


def _detections_to_frames(paths: list[Path]) -> dict:
    """Chunk detection JSONs -> single timeline on the source clock."""
    frames_players, frames_ball = [], []
    fps_eff = None
    w = h = None
    for p, chunk in paths:
        d = json.loads(p.read_text())
        if fps_eff is None:
            fps_eff = d["fps"] / d.get("stride", 1)
            w, h = d["width"], d["height"]
        frames_players += d["players"]
        frames_ball += d["ball"]
    return {"players": frames_players, "ball": frames_ball, "fps_eff": fps_eff, "w": w, "h": h}


def _signals(video, det: dict, res, pitch, cfg: Config, remote: bool = False) -> tuple[dict, dict]:
    from .audio import audio_envelope, audio_spikes
    from .ball import ball_signals, link_ball
    from .players import PlayerSigCfg, player_signals

    fps_eff = det["fps_eff"]
    w, h = det["w"], det["h"]
    diag: dict = {}

    track = link_ball(det["ball"], fps_eff, w, cfg.ball_max_jump_frac, cfg.ball_min_tracklet,
                      cfg.ball_interp_gap_s, cfg.ball_smooth_window)
    v_shot = cfg.v_shot_pitch if res.space is Space.PITCH else cfg.v_shot_pixel
    sig = ball_signals(track, res.zones, res.space, cfg.bin_s, v_shot, cal=res.cal, lost_s=cfg.ball_lost_s)
    diag["ball_seen"] = float(sig["ball_seen"].mean())

    pcfg = PlayerSigCfg(
        v_run=cfg.v_run_pitch if res.space is Space.PITCH else cfg.v_run_pixel,
        cluster_radius=cfg.cluster_radius_pitch if res.space is Space.PITCH else cfg.cluster_radius_pixel,
        cluster_min_players=cfg.cluster_min_players,
        cluster_slow=cfg.cluster_slow_pitch if res.space is Space.PITCH else cfg.cluster_slow_pixel,
        cluster_min_dur_s=cfg.cluster_min_dur_s,
        restart_half_width=cfg.restart_half_width,
        restart_centre_r=cfg.restart_centre_r)
    sig.update(player_signals(det["players"], fps_eff, res.cal, res.space, res.zones, w, h, pitch, pcfg, cfg.bin_s))

    if remote:
        from .modal_app import audio_envelope_remote
        env = audio_envelope_remote.remote(video, cfg.audio_hop_s)
        t_a, db = np.array(env["t"]), np.array(env["rms_db"])
    else:
        t_a, db = audio_envelope(Path(video), cfg.audio_hop_s)
    sig["audio"] = audio_spikes(t_a, db, cfg.bin_s, cfg.audio_baseline_win_s,
                                cfg.audio_thresh_db, cfg.audio_min_dur_s)
    n_bins = max(len(v) for v in sig.values())
    for k, v in list(sig.items()):
        if len(v) < n_bins:
            sig[k] = np.pad(v, (0, n_bins - len(v)))
    return sig, diag


def cmd_run(args) -> int:
    from pitchworld.pitch import PitchModel
    from pitchworld.sync import probe

    from .calib import _manual_entry, calibrate_frame
    from .chunks import extract_clip, split
    from .combine import combine
    from .modal_app import (
        detect_chunk_path,
        extract_clips_remote,
        probe_remote,
        read_frame_remote,
        run_detection,
        split_remote,
    )
    from .report import write_review_html

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(Path(args.config) if args.config else None)
    pitch = PitchModel.load(Path(args.pitch)) if args.pitch else PitchModel.standard()
    warnings: list[str] = []

    video, remote = _parse_source(args.video, cfg)
    if remote and args.local:
        raise ValueError("--local is only for local files")
    info = probe_remote.remote(video) if remote else probe(Path(video))
    duration = info["duration"]
    if args.max_minutes:
        duration = min(duration, args.max_minutes * 60)

    if remote:
        frame_jpg = out / "calib_frame.jpg"
        if not frame_jpg.exists():
            frame_jpg.write_bytes(read_frame_remote.remote(video, args.frame_time))
        import cv2
        frame = cv2.imdecode(np.frombuffer(frame_jpg.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        res = calibrate_frame(frame, pitch, _manual_entry(Path(args.calib)) if args.calib else None,
                              Path(args.goal_zones_px) if args.goal_zones_px else None)
    else:
        res = calibrate(Path(video), pitch, Path(args.calib) if args.calib else None,
                        frame_time=args.frame_time,
                        goal_zones_px=Path(args.goal_zones_px) if args.goal_zones_px else None)
    warnings += res.warnings
    if res.space is Space.PIXEL:
        print("WARNING: calibration failed -> PIXEL space heuristics in use", file=sys.stderr)
        if not res.zones:
            _warn("no goal zones (need --goal-zones-px in pixel space); ball/attack signals will be zero", warnings)

    det_dir = out / "detections"
    det_dir.mkdir(exist_ok=True)
    if args.reuse and list(det_dir.glob("chunk_*.json")):
        det_paths = sorted(det_dir.glob("chunk_*.json"))
    elif remote:
        chunks = split_remote.remote(video, cfg.chunk_s)
        det_paths = [det_dir / f"chunk_{c['idx']:03d}.json" for c in chunks]
        todo = [(c, p) for c, p in zip(chunks, det_paths) if not p.exists()]
        if todo:
            results = detect_chunk_path.map([c["path"] for c, _ in todo], [cfg.to_dict()] * len(todo))
            for (_, p), r in zip(todo, results):
                p.write_text(json.dumps(r))
    else:
        chunks = split(Path(video), out / "chunks", cfg.chunk_s)
        det_paths = run_detection(chunks, cfg, det_dir, local=args.local)

    det = _detections_to_frames([(p, None) for p in det_paths])
    sig, diag = _signals(video, det, res, pitch, cfg, remote=remote)
    np.savez(out / "signals.npz", **{k: np.asarray(v) for k, v in sig.items()})
    (out / "signal_meta.json").write_text(json.dumps({"bin_s": cfg.bin_s, **diag}))

    cands = combine(sig, cfg.bin_s, cfg, duration)
    if not args.no_clips:
        if remote:
            clips = [{"id": c.id, "start": c.start, "end": min(c.end, duration)} for c in cands]
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
                extract_clip(Path(video), c.start, min(c.end, duration), out / "clips" / name)
                c.clip = name

    write_review_html(out, cands)
    payload = {"video": args.video, "duration_s": duration,
               "calibration": ({"method": res.cal.method, "confidence": res.cal.confidence,
                                "warnings": warnings} if res.cal else None),
               "space": res.space.value, "config": cfg.to_dict(), "diagnostics": diag,
               "warnings": warnings, "candidates": [c.to_dict() for c in cands]}
    (out / "candidates.json").write_text(json.dumps(payload, indent=2))
    if diag["ball_seen"] < 0.2:
        print(f"WARNING: ball_seen={diag['ball_seen']:.2f} (<0.2) — ball model unreliable on this footage",
              file=sys.stderr)
    print(f"space={res.space.value}  ball_seen={diag['ball_seen']:.2f}  candidates={len(cands)}")
    print(f"{'rank':<5}{'type':<8}{'conf':<7}{'time':<8}{'goal':<5}signals")
    for c in cands:
        mm, ss = int(c.t_event // 60), int(c.t_event % 60)
        print(f"{c.rank:<5}{c.type:<8}{c.confidence:<7.2f}{mm}:{ss:02d}    {c.goal:<5}"
              + " ".join(f"{k}={v:.2f}" for k, v in c.signals.items()))
    print(f"-> {out}/candidates.json, review.html, clips/")
    return 0


def cmd_recombine(args) -> int:
    from .combine import combine
    from .report import write_review_html

    out = Path(args.out)
    cfg = Config.load(Path(args.config) if args.config else None)
    payload = json.loads((out / "candidates.json").read_text())
    video = Path(payload["video"])
    sig = dict(np.load(out / "signals.npz"))
    duration = payload["duration_s"]
    cands = combine(sig, cfg.bin_s, cfg, duration)
    if not args.no_clips:
        from .chunks import extract_clip
        for c in cands:
            mm, ss = int(c.t_event // 60), int(c.t_event % 60)
            name = f"{c.id}_{c.type}_{mm}m{ss:02d}s.mp4"
            extract_clip(video, c.start, min(c.end, duration), out / "clips" / name)
            c.clip = name
    write_review_html(out, cands)
    payload["candidates"] = [c.to_dict() for c in cands]
    payload["config"] = cfg.to_dict()
    (out / "candidates.json").write_text(json.dumps(payload, indent=2))
    print(f"{len(cands)} candidates -> {out}/candidates.json")
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
    p.add_argument("--max-minutes", type=float)
    p.add_argument("--frame-time", type=float, default=1.0)
    p.add_argument("--no-clips", action="store_true")
    p.set_defaults(f=cmd_run)

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
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_check_calib)

    p = sub.add_parser("recombine")
    p.add_argument("--out", required=True)
    p.add_argument("--pitch")
    p.add_argument("--config")
    p.add_argument("--no-clips", action="store_true")
    p.set_defaults(f=cmd_recombine)

    args = ap.parse_args()
    t0 = time.time()
    rc = args.f(args)
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
