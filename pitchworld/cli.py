"""pitchworld — multi-camera football clips -> synced clips + unified pitch-coordinate tracking JSON.

    pitchworld run cam0.mp4 cam1.mp4 [cam2.mp4 cam3.mp4] --out out/ --calib calib.json [--pitch pitch.json]
    pitchworld sync  cam0.mp4 cam1.mp4 ... --out out/            # sync only, writes sync.json + contact sheet
    pitchworld calibrate cam0.mp4 --camera 0 --out calib.json    # click landmarks (GUI)
    pitchworld check-calib cam0.mp4 --camera 0 --calib calib.json --out check.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .calibrate import CameraCalibration, calibrate_camera, load_calibration_file, read_frame
from .pitch import PitchModel
from .sync import probe, render_synced, sync_clips


def _pitch_from_args(args) -> PitchModel:
    if args.pitch:
        return PitchModel.load(Path(args.pitch))
    return PitchModel.standard()


def _load_manual_points(calib_path: str | None, n: int) -> list[dict | None]:
    """Per-camera constraint dicts ({points, lines, arcs}) from a calibration JSON, or None."""
    if not calib_path:
        return [None] * n
    data = load_calibration_file(Path(calib_path))
    cams = data.get("cameras", {})
    out = []
    for i in range(n):
        e = cams.get(str(i)) or {}
        e = {k: e.get(k) for k in ("points", "lines", "arcs", "parallels")}
        out.append(e if any(e.values()) else None)
    return out


def cmd_sync(args) -> int:
    clips = [Path(p) for p in args.clips]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = sync_clips(clips, manual_offsets=args.offsets, min_conf=args.min_sync_conf, max_lag_s=args.max_lag)
    res.to_json(out / "sync.json")
    _print_sync(res)
    return 0


def _print_sync(res) -> None:
    print("SYNC")
    for i, (o, c, m) in enumerate(zip(res.offsets_s, res.confidences, res.method)):
        flag = "" if (c >= 0.5 or m != "audio_xcorr") else "   <-- LOW CONFIDENCE"
        print(f"  clip {i}: offset {o:+.3f}s  confidence {c:.2f}  ({m}){flag}")
    print(f"  common window: {res.common_duration_s:.2f}s")
    for n in res.notes:
        print("  !", n)
    if res.low_confidence:
        print("  !! Audio sync confidence is low. Verify with the contact sheet; if wrong, re-run with "
              "--offsets 0 <off1> <off2> ... (seconds; time in clip i that matches t=0 of clip 0)")


def cmd_calibrate(args) -> int:
    from .calib_tool import run_click_tool

    pitch = _pitch_from_args(args)
    landmarks = args.landmarks.split(",") if args.landmarks else None
    run_click_tool(Path(args.clip), args.camera, pitch, Path(args.out), landmarks, frame_time=args.time)
    return 0


def cmd_check_calib(args) -> int:
    from .viz import calibration_check_image

    pitch = _pitch_from_args(args)
    pts = _load_manual_points(args.calib, args.camera + 1)[args.camera]
    cal, warnings = calibrate_camera(Path(args.clip), pitch, pts, frame_time=args.time)
    for w in warnings:
        print("  !", w)
    calibration_check_image(read_frame(Path(args.clip), args.time), pitch, cal, Path(args.out))
    print(f"camera {args.camera}: {cal.method} err={cal.reproj_error_px:.1f}px/{cal.reproj_error_m:.2f}m "
          f"conf={cal.confidence:.2f} -> {args.out}")
    return 0


def cmd_run(args) -> int:
    from .merge import fuse, summarize
    from .postprocess import (
        assign_teams,
        filter_static_tracks,
        render_jersey_sheet,
        render_team_snapshot,
        smooth_tracks,
    )
    from .viz import calibration_check_image, render_overlay, render_pitch_map, sync_contact_sheet

    t_start = time.time()
    clips = [Path(p) for p in args.clips]
    if not 2 <= len(clips) <= 4:
        print("need 2-4 clips", file=sys.stderr)
        return 2
    for c in clips:
        if not c.exists():
            print(f"missing: {c}", file=sys.stderr)
            return 2
    out = Path(args.out)
    (out / "synced").mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    pitch = _pitch_from_args(args)
    warnings: list[str] = []

    # 1. SYNC -----------------------------------------------------------
    print(f"[1/4] syncing {len(clips)} clips ...")
    sync = sync_clips(clips, manual_offsets=args.offsets, min_conf=args.min_sync_conf, max_lag_s=args.max_lag)
    sync.to_json(out / "sync.json")
    _print_sync(sync)
    warnings += sync.notes
    fps = args.fps or min(probe(c)["fps"] for c in clips)
    fps = float(round(fps))
    synced = []
    for i, c in enumerate(clips):
        dst = out / "synced" / f"cam{i}.mp4"
        if not dst.exists() or not args.reuse:
            render_synced(c, dst, sync.common_start_s[i], sync.common_duration_s, fps, height=args.height)
        synced.append(dst)
    sync_contact_sheet(synced, [sync.common_duration_s * f for f in (0.1, 0.5, 0.9)], out / "debug" / "sync_check.jpg")
    print(f"  synced clips -> {out/'synced'} @ {fps:.0f} fps, {sync.common_duration_s:.2f}s; check {out/'debug'/'sync_check.jpg'}")

    # 2. CALIBRATE -----------------------------------------------------
    print("[2/4] calibrating cameras ...")
    manual = _load_manual_points(args.calib, len(clips))
    cals: list[CameraCalibration] = []
    for i, (c, pts) in enumerate(zip(clips, manual)):
        cal, w = calibrate_camera(c, pitch, pts, frame_time=args.calib_time)
        for msg in w:
            print(f"  ! cam{i}: {msg}")
            warnings.append(f"cam{i}: {msg}")
        flag = "" if cal.confidence >= 0.5 else "   <-- LOW CONFIDENCE"
        print(f"  cam{i}: {cal.method} {cal.n_points} pts, err {cal.reproj_error_px:.1f}px / {cal.reproj_error_m:.2f}m, "
              f"confidence {cal.confidence:.2f}{flag}")
        # calibration was done on the original clip; synced clips may be rescaled -> adjust H
        info_src, info_dst = probe(c), probe(synced[i])
        s = info_dst["width"] / info_src["width"]
        if abs(s - 1) > 1e-3:
            S = np.diag([1 / s, 1 / s, 1.0])
            cal.H = (cal.matrix @ S).tolist()
            for p in cal.points:
                p["pixel"] = [round(v * s, 1) for v in p["pixel"]]
            for tr in cal.lines + cal.arcs + cal.parallels:
                tr["pixels"] = [[round(v * s, 1) for v in uv] for uv in tr["pixels"]]
        calibration_check_image(read_frame(synced[i], 0.5), pitch, cal, out / "debug" / f"calib_cam{i}.jpg")
        cals.append(cal)

    # 3. TRACK ---------------------------------------------------------
    print(f"[3/4] tracking with {args.model} on {'local CPU' if args.local else 'Modal GPU'} ...")
    per_cam = []
    cache = out / "debug" / "raw_tracks.json"
    if args.reuse and cache.exists():
        per_cam = json.loads(cache.read_text())
        print("  reusing cached detections")
    else:
        if args.local:
            from .modal_app import track_local

            for s in synced:
                per_cam.append(track_local(str(s), model_name=args.model, conf=args.conf, imgsz=args.imgsz,
                                           classes=(0, 32) if args.ball else (0,),
                                           ball_imgsz=args.ball_imgsz)["frames"])
        else:
            from .modal_app import app, track_video

            payloads = [s.read_bytes() for s in synced]
            with app.run():
                results = list(track_video.map(payloads, kwargs={"model_name": args.model, "conf": args.conf,
                                                                "imgsz": args.imgsz,
                                                                "classes": (0, 32) if args.ball else (0,),
                                                                "ball_imgsz": args.ball_imgsz}))
            per_cam = [r["frames"] for r in results]
        cache.write_text(json.dumps(per_cam))
    for i, fr in enumerate(per_cam):
        n = sum(len(f) for f in fr)
        print(f"  cam{i}: {len(fr)} frames, {n} person detections ({n/max(1,len(fr)):.1f}/frame)")

    # 4. FUSE + OUTPUT -------------------------------------------------
    print("[4/4] projecting to pitch coordinates and merging cameras ...")
    import math

    from .jointfit import apply_pose, cross_camera_consistency, refine_joint

    sizes = [(probe(s)["width"], probe(s)["height"]) for s in synced]
    consistency = cross_camera_consistency(cals, per_cam)
    if args.joint_refine and all(c.pose for c in cals):
        print("  joint multi-camera refinement (players as shared ground points) ...")
        jr = refine_joint(cals, pitch, per_cam, sizes, free_radius=pitch.d_radius > 0)
        pitch = jr.pitch
        for i, cal in enumerate(cals):
            med = [v for k, v in jr.pair_median_m.items() if str(i) in k.split("-") and not math.isnan(v)]
            cals[i] = apply_pose(cal, jr.poses[i], sizes[i], pitch, float(np.median(med)) if med else float("nan"))
            calibration_check_image(read_frame(synced[i], 0.5), pitch, cals[i], out / "debug" / f"calib_cam{i}.jpg")
        warnings += jr.notes
        consistency = jr.pair_median_m
    worst = max([v for v in consistency.values() if not math.isnan(v)], default=float("nan"))
    print("  cross-camera player-position disagreement (median m): "
          + ", ".join(f"cam{k}: {v:.2f}" for k, v in consistency.items()))
    if math.isnan(worst):
        warnings.append("could not measure cross-camera consistency (no shared tracks)")
    elif worst > 1.0:
        warnings.append(f"cameras disagree on player positions by up to {worst:.2f} m (median): the calibrations "
                        "do not define one shared pitch frame; merged positions/IDs are LOW confidence")
    timeline = fuse(per_cam, cals, pitch, fps, merge_dist=args.merge_dist, margin=args.pitch_margin,
                    min_box_h=args.min_box_h, min_conf=args.min_det_conf, ball=args.ball)
    team_info = None
    if not args.no_static_filter:
        timeline, dropped = filter_static_tracks(timeline, pitch, fps, min_disp_m=args.static_min_disp)
        print(f"  static filter dropped {len(dropped)} ids: {dropped}")
    if not args.no_smooth:
        timeline = smooth_tracks(timeline, fps, window=args.smooth_window, max_gap=args.max_gap,
                                 min_frames=args.min_track_frames)
    if not args.no_team:
        timeline, team_info = assign_teams(timeline, synced, pitch, n_samples=args.team_samples)
        print(f"  team sizes: {team_info['team_sizes']}")
    stats = summarize(timeline)
    low_sync = sync.low_confidence
    low_cal = any(c.confidence < 0.5 for c in cals) or math.isnan(worst) or worst > 1.0
    result = {
        "version": __version__,
        "pitch": pitch.to_dict(),
        "fps": fps,
        "num_frames": len(timeline),
        "duration_s": round(len(timeline) / fps, 3),
        "cameras": [{
            "index": i, "source": str(c.resolve()), "synced_clip": str(s.resolve()),
            "sync_offset_s": sync.offsets_s[i], "sync_confidence": sync.confidences[i], "sync_method": sync.method[i],
            "trim_start_s": sync.common_start_s[i],
            "homography_px_to_m": cal.H, "calibration": {k: v for k, v in cal.to_dict().items() if k != "H"},
            "frame_size": [probe(s)["width"], probe(s)["height"]],
        } for i, (c, s, cal) in enumerate(zip(clips, synced, cals))],
        "quality": {"sync_low_confidence": low_sync, "calibration_low_confidence": low_cal,
                    "cross_camera_disagreement_m": consistency, "warnings": warnings, "stats": stats},
        "frames": timeline,
    }
    if team_info is not None:
        result["quality"]["teams"] = team_info
    (out / "tracking.json").write_text(json.dumps(result))
    print(f"  tracking.json: {len(timeline)} frames, {stats}")
    render_team_snapshot(timeline, pitch, len(timeline) // 2, out / "debug" / "teams_pitch.png",
                         team_info["team_colours_bgr"] if team_info else [[0, 0, 255], [255, 0, 0]])
    render_jersey_sheet(synced[0], timeline, 0, len(timeline) // 2, out / "debug" / "jerseys_cam0.jpg")

    if not args.no_viz:
        render_pitch_map(timeline, pitch, fps, out / "debug" / "pitch_map.mp4")
        for i, s in enumerate(synced):
            render_overlay(s, timeline, i, out / "debug" / f"overlay_cam{i}.mp4", pitch, cals[i])
        print(f"  debug videos -> {out/'debug'}")

    print(f"done in {time.time()-t_start:.0f}s -> {out}")
    if low_sync or low_cal:
        print("\n!!! QUALITY WARNING !!!")
        if low_sync:
            print("  - audio sync confidence is LOW; verify debug/sync_check.jpg or pass --offsets")
        if low_cal:
            print("  - camera calibration is LOW confidence (per-camera fit and/or cross-camera disagreement); "
                  "inspect debug/calib_cam*.jpg and add clicked landmarks (pitchworld calibrate)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pitchworld", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_pitch(p):
        p.add_argument("--pitch", help="pitch model JSON (default: standard 105x68). See examples/")

    def common_sync(p):
        p.add_argument("--offsets", type=float, nargs="+", help="manual offsets (s), one per clip; first is 0")
        p.add_argument("--min-sync-conf", type=float, default=0.5)
        p.add_argument("--max-lag", type=float, default=None, help="limit |offset| search (s)")

    p = sub.add_parser("run", help="full pipeline")
    p.add_argument("clips", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--calib", help="calibration JSON with clicked landmarks per camera")
    p.add_argument("--calib-time", type=float, default=1.0, help="time (s) of the frame used for calibration")
    p.add_argument("--fps", type=float, default=None, help="output fps (default: min source fps, rounded)")
    p.add_argument("--height", type=int, default=None, help="rescale synced clips to this height")
    p.add_argument("--model", default="yolov8m.pt", help="Ultralytics weights, e.g. yolov8m.pt, yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--merge-dist", type=float, default=2.0, help="max pitch distance (m) to merge cross-camera detections")
    p.add_argument("--ball", action="store_true")
    p.add_argument("--ball-imgsz", type=int, default=1920)
    p.add_argument("--min-box-h", type=int, default=12)
    p.add_argument("--min-det-conf", type=float, default=0.0)
    p.add_argument("--pitch-margin", type=float, default=3.0)
    p.add_argument("--no-team", action="store_true")
    p.add_argument("--no-static-filter", action="store_true")
    p.add_argument("--static-min-disp", type=float, default=1.5)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--max-gap", type=int, default=10)
    p.add_argument("--min-track-frames", type=int, default=15)
    p.add_argument("--team-samples", type=int, default=40)
    p.add_argument("--joint-refine", action="store_true",
                   help="experimental: jointly refine pose-fitted calibrations using players seen by >=2 cameras")
    p.add_argument("--local", action="store_true", help="run YOLO locally on CPU instead of Modal")
    p.add_argument("--reuse", action="store_true", help="reuse synced clips / cached detections in --out")
    p.add_argument("--no-viz", action="store_true")
    common_pitch(p)
    common_sync(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sync", help="audio sync only")
    p.add_argument("clips", nargs="+")
    p.add_argument("--out", required=True)
    common_sync(p)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("calibrate", help="click landmarks for one camera")
    p.add_argument("clip")
    p.add_argument("--camera", type=int, required=True)
    p.add_argument("--out", required=True, help="calibration JSON to create/update")
    p.add_argument("--landmarks", help="comma-separated landmark names to click, in order")
    p.add_argument("--time", type=float, default=1.0)
    common_pitch(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("check-calib", help="render pitch lines through a calibration onto a frame")
    p.add_argument("clip")
    p.add_argument("--camera", type=int, required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time", type=float, default=1.0)
    common_pitch(p)
    p.set_defaults(func=cmd_check_calib)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
