# Multi-view hi-res player tracking ("players v2")

Goal: replace the single-angle 480p players pass with a fused, pitch-space track set
built from all angles at 1080p, so the radar and per-player stats are meaningfully better.
CPU only (Oracle: 4 ARM cores, 23 GB). Must run behind the job lock like every other pass.

## 1. Per-angle calibration (UI + API)

Replace the 4-corner radar picker with a per-angle **pitch landmark picker**:
- For each angle i, show a still (existing /analysis/frame endpoint, angle=i, t=frame_t).
- The user picks from a fixed landmark list (dropdown/chips) and clicks its position on the
  padded canvas (reuse the padded/draggable canvas from RadarReplay). Landmarks and pitch
  coordinates (metres, pitch 100×64 default; use project pitch_len_m if present, width = len*0.64):
  corners (4), halfway×touchline (2), centre spot, penalty-box corners (8: each box has 2 on the
  goal line and 2 inside), six-yard-box corners (8), penalty spots (2), goal posts (4).
- Need ≥4 non-collinear landmarks per angle (UI recommends 8–12); show reprojection RMS (metres) after solving.
- Store `multiangle/calib.json`: `{ "angles": { "0": {"pts": [{"name","fx","fy"}...], "H": [[..]] , "rms_m": x}, ... }, "pitch": {"len_m","wid_m"} }`.
- Backend: `GET/PUT /analysis/calib` (validate names ∈ list, fx,fy ∈ [-1,2]); solve H with
  cv2.findHomography(RANSAC off, LMEDS off — plain least squares; ≥4 pts) in the backend and
  return rms. Frontend no longer needs homography.ts for radar (keep file if still used).

## 2. Hi-res detection per angle (`highlights/analysis/detect_hr.py`)

- Input: original source video for angle i (not the 480p proxy), window = shared window
  (existing ctx offsets), fps = 2.
- Decode with the existing ffmpeg pipe reader at native size. Detect with YOLO (yolov8n.pt,
  `--model` override) on a 2×2 tile grid with 10 % overlap (each tile ~1056×594 → imgsz=1088),
  `conf=0.15`, `classes=[0]`, then merge tile boxes with NMS (IoU 0.5) in full-frame coords.
  Support `--backend onnx`: export yolov8n to ONNX once (`model.export(format="onnx", imgsz=1088)`)
  and run through onnxruntime (aarch64 wheel; add to Docker deps) — benchmark both on Oracle later
  and keep whichever is faster (do NOT benchmark now; Oracle is busy).
- Output `analysis/players_v2/det_a{i}.npz`: arrays `t` (match time, s), `x1,y1,x2,y2` (px),
  `conf`, plus `meta.json` (w,h,fps,imgsz,model). Resume-safe: write every 5 min; on restart
  skip finished seconds.
- Footpoint = (x centre, y2). Also save a 32×96 crop feature per detection? No — team colour:
  reuse `players.assign_team` on the torso crop; store `team` per detection ('A'/'B'/None).
- Time budget: measure s/frame on Oracle on the first 20 frames and log ETA; abort with a
  PipelineError if projected > 8 h for the angle.

## 3. Fusion into pitch space (`highlights/analysis/fuse_tracks.py`)

For each 0.5 s step t:
1. Project every angle's footpoints through its H → pitch (x,y) m. Drop points outside the
   pitch by > 3 m. Drop detections whose box height < 12 px.
2. Cross-view merge: greedy agglomerate points from different angles within 2.0 m into one
   observation (mean position, weighted by conf; team = majority vote; keep per-angle box refs).
3. Track in pitch space: constant-velocity Kalman (per player), Hungarian assignment with
   gate 3.0 m (grow to 4.5 m after misses), max 6 s of misses before a track ends, min track
   length 4 s to keep. Team of a track = majority over its observations; never assign an
   observation whose team conflicts with the track's team when both known.
4. Output `analysis/players_v2/tracks.json`: `{ "step": 0.5, "t0": .., "tracks": [
   {"id", "team", "start", "end", "xy": [[x,y]...] (NaN for gaps ≤ 2 s → linearly filled),
   "dist_m", "sprints", "crops": [paths...] } ] }` and `summary.json` (n tracks, mean track
   length, players-visible-per-step histogram — sanity: median should be ~ number of players
   on the pitch; report it).
   Crops: for each track, save up to 6 crops from the angle with the largest box at that time
   (1080p source → much cleaner than today's 480p crops).
5. Ball: keep the existing per-angle `features_1s` ball_xy projected through H, merged by
   confidence, as `ball` in tracks.json.

## 4. Wire-up

- `players_run.py --v2`: runs detect_hr for each angle sequentially then fuse; status stages
  `detect a0/a1/a2`, `fuse`; needs calib.json for every angle (PipelineError otherwise).
- API: `POST /analysis/players/v2/run`, `GET /analysis/players/v2/tracks` (strip crops list to
  urls), reuse job lock; `GET /analysis/players/paths` returns v2 tracks when present (already in
  pitch coords: add `"space": "pitch"` so the radar skips the homography).
- Radar (labs only for now): when `space == "pitch"` plot directly; show players-visible count.
- Groups UI (labs): groups now = v2 tracks (far fewer, far longer); keep name/merge/hide.

## 5. Tests (pytest, no video)

- calib: solving H from 6 synthetic landmarks recovers a known H (rms < 1e-6 m); 3 pts → 422.
- fuse: two synthetic angles seeing the same 6 moving players with 0.5 m noise + 1 angle
  missing every 3rd step → fusion yields 6 tracks, each ≥ 95 % of steps, positions within 1 m.
- det_hr: frame-reader + resume logic with a fake model (mock predict) writes/reads npz.

Gates: `pytest highlights -q`, `ruff check highlights`, `npx tsc --noEmit`, `npm run build`.
Do NOT deploy or run on Oracle without my go — re-cuts are running there.
