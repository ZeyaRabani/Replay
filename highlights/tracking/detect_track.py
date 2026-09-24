"""Detect + ByteTrack players and estimate frame->reference homography.

Decodes frames sequentially at a fixed rate via an ffmpeg pipe, runs YOLO
person detection, ByteTrack, and ORB-based stabilisation against a reference
frame. Saves detections and per-frame homographies to a .npz plus meta.json.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def probe_size(video):
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", video,
        ]
    ).decode().strip().split(",")
    return int(out[0]), int(out[1])


def frame_reader(video, width, height, fps):
    frame_bytes = width * height * 3
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        stdout=subprocess.PIPE,
        bufsize=frame_bytes * 8,
    )

    def gen():
        while True:
            buf = b""
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    return
                buf += chunk
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

    return proc, gen()


def orb_homography(orb, matcher, gray, ref_kp, ref_des, top_band, person_boxes):
    """Compute homography gray -> reference using keypoints in the top band
    outside person boxes. Returns (H, n_inliers)."""
    kp, des = orb.detectAndCompute(gray, None)
    if not kp or des is None or len(kp) < 10:
        return None, 0
    pts = np.array([k.pt for k in kp])
    keep = pts[:, 1] < top_band
    if person_boxes is not None and len(person_boxes):
        x1 = np.minimum(person_boxes[:, 0], person_boxes[:, 2])
        x2 = np.maximum(person_boxes[:, 0], person_boxes[:, 2])
        y1 = np.minimum(person_boxes[:, 1], person_boxes[:, 3])
        y2 = np.maximum(person_boxes[:, 1], person_boxes[:, 3])
        inside = (
            (pts[:, 0, None] >= x1) & (pts[:, 0, None] <= x2)
            & (pts[:, 1, None] >= y1) & (pts[:, 1, None] <= y2)
        ).any(axis=1)
        keep &= ~inside
    idx = np.where(keep)[0]
    if len(idx) < 10:
        return None, 0
    kp_f = [kp[i] for i in idx]
    des_f = des[idx]
    matches = matcher.knnMatch(des_f, ref_des, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m in pair[:1] if m.distance < 0.75 * pair[1].distance]
    if len(good) < 10:
        return None, 0
    src = np.float32([kp_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0
    n_inliers = int(mask.sum())
    det = np.linalg.det(H)
    degenerate = (
        not np.isfinite(det) or not (0.5 < abs(det) < 2)
        or abs(H[0, 2]) > 300 or abs(H[1, 2]) > 300
    )
    if degenerate:
        return None, n_inliers
    return H, n_inliers


def torso_hsv(frame, box):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = int(np.clip(x1 + 0.25 * bw, 0, w_img))
    tx2 = int(np.clip(x1 + 0.75 * bw, 0, w_img))
    ty1 = int(np.clip(y1 + 0.20 * bh, 0, h_img))
    ty2 = int(np.clip(y1 + 0.55 * bh, 0, h_img))
    if tx2 <= tx1 or ty2 <= ty1:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    return [float(v) for v in hsv.reshape(-1, 3).mean(axis=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--ref-time", type=float, default=600)
    ap.add_argument("--top-band", type=int, default=260)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    video = args.video
    width, height = probe_size(video)
    fps = args.fps
    ref_idx = round(args.ref_time * fps)

    model = YOLO(args.model)
    tracker = sv.ByteTrack(
        frame_rate=round(fps),
        lost_track_buffer=int(3 * fps),
        minimum_matching_threshold=0.8,
        track_activation_threshold=args.conf,
    )
    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    proc, frames = frame_reader(video, width, height, fps)
    t0 = time.time()

    det_rows = []
    frame_rows = []
    ref_frame = None
    ref_kp = ref_des = None
    prev_H = None
    n_frames = 0
    n_stab_fail = 0

    for frame_idx, frame in enumerate(frames):
        t = frame_idx / fps
        if args.max_seconds is not None and t > args.max_seconds:
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf,
                        classes=[0], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        dets = tracker.update_with_detections(dets)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = dets.xyxy if len(dets) else np.empty((0, 4))

        if frame_idx == ref_idx:
            ref_frame = frame.copy()
            ref_kp_all, ref_des_all = orb.detectAndCompute(gray, None)
            pts = np.array([k.pt for k in ref_kp_all])
            keep = np.where(pts[:, 1] < args.top_band)[0]
            ref_kp = [ref_kp_all[i] for i in keep]
            ref_des = ref_des_all[keep]

        stab_ok = 1.0
        if ref_kp is None:
            H = np.eye(3)
            n_inliers = 0
            stab_ok = 0.0
        elif frame_idx == ref_idx:
            H = np.eye(3)
            n_inliers = len(ref_kp)
        else:
            H, n_inliers = orb_homography(
                orb, matcher, gray, ref_kp, ref_des, args.top_band, boxes)
            if H is None or n_inliers < 25:
                H = prev_H if prev_H is not None else np.eye(3)
                stab_ok = 0.0
                n_stab_fail += 1
        prev_H = H

        track_ids = (dets.tracker_id if dets.tracker_id is not None
                     else np.full(len(dets), -1))
        for i in range(len(dets)):
            x1, y1, x2, y2 = boxes[i]
            h, s, v = torso_hsv(frame, boxes[i])
            det_rows.append([frame_idx, t, float(track_ids[i]),
                             x1, y1, x2, y2,
                             float(dets.confidence[i]), h, s, v])
        frame_rows.append([frame_idx, t, float(n_inliers), stab_ok]
                          + H.reshape(-1).tolist())
        n_frames += 1

        if n_frames % 200 == 0:
            elapsed = time.time() - t0
            print(f"[{n_frames} frames, t={t:.0f}s] "
                  f"{n_frames / elapsed:.2f} frames/s, "
                  f"{len(det_rows)} dets, {n_stab_fail} stab fails",
                  flush=True)

    proc.stdout.close()
    proc.wait()

    runtime = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    det = np.asarray(det_rows, dtype=np.float32).reshape(-1, 11)
    fr = np.asarray(frame_rows, dtype=np.float32).reshape(-1, 13)
    np.savez(out, det=det, frames=fr)

    if ref_frame is not None:
        cv2.imwrite(str(out.with_name("ref_frame.png")), ref_frame)

    meta = {
        "fps": fps, "imgsz": args.imgsz, "model": args.model,
        "video": str(video), "ref_time": args.ref_time,
        "runtime_s": runtime, "n_frames": n_frames,
        "n_detections": len(det), "n_stab_fail": n_stab_fail,
    }
    with open(out.with_name("meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {n_frames} frames in {runtime:.1f}s "
          f"({n_frames / runtime:.2f} f/s), {len(det)} dets -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
