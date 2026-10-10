# Track benchmark — where does tracking time go?

- Machine: **8 cores** (this box, /home/ubuntu), torch 2.14.0 CPU, ultralytics 8.4.161, yolov8n, ffmpeg 4.4.2
- Source: `/home/ubuntu/ma_proj/angles/a0/match.mp4` — **vp9, 2400×1080**, 5337 s (YouTube download)
- Window: 120 s at t=600–720, `fps=1,scale=960:-2` (the exact `_frame_reader` command)
- Keyframe interval: **~4.3 s avg, irregular (2–8 s)** — 28 key packets in 120 s → `-skip_frame nokey` yields only ~0.23 fps, below the required 1 fps

## Seconds per 120 s of video

| Variant | Wall time | Notes |
|---|---|---|
| (a) decode-only, current cmd | **9.3 s** | baseline |
| (b) `-skip_frame nokey` | 10.8 s | **slower than baseline** and only ~28 usable frames/120 s — dead end on VP9 |
| (c1) `-threads 4` | 10.8 s | no gain (VP9 decoder already saturates) |
| (c2) `-hwaccel auto` | 9.2 s | no CUDA/VDPAU on this box — falls back to software, same speed |
| (d) full `compute_rows` imgsz 960 | **13.5 s** | 120 frames, ball_rate **0.175**, mean players **7.4** |
| (e) `compute_rows` imgsz 640 | 11.0 s | ball_rate **0.017** (−90%), mean players 7.1 — ball detection collapses |
| (f) `compute_rows` fps 0.5 | 11.0 s | 60 frames, ball_rate 0.167, mean players 7.5 — accuracy holds |

## Read on the numbers
- Decode ≈ **9.3 s of the 13.5 s ≈ 69%** of full tracking; YOLO on 960-wide frames is only ~4 s/120 s (~35 ms/frame).
- Decode-bound: ffmpeg `-vf fps=1` still **decodes every frame** (30 fps in → drop 29/30). The real lever is decoding fewer frames, but `-skip_frame nokey` can't (keyframes every ~4.3 s ≫ 1 s, and it was slower anyway — likely VP9 alt-ref overhead). `-threads`/`-hwaccel` do nothing.
- imgsz 640 saves ~2.5 s but destroys ball detection (0.175 → 0.017 rate): yolov8n needs the 960 input for the small ball.
- fps 0.5 halves frame count yet only saves 2.5 s — because decode dominates; temporal coverage is halved (worse linger semantics, sparser director input).
- Effective speedups available: process-level parallelism across angles (already 2-way), or splitting each angle's decode+detect across N time-chunks of the *same* video in parallel processes (ffmpeg segment seeks are cheap — `-ss` before `-i` is a keyframe seek). ~8 cores: 2 angles × ~2 threads of VP9 decode each is roughly saturated already; CPU is the wall.

## Analysis-proxy experiment (same 120 s window, t=600 of a0)

- yt-dlp on this box: metadata OK (480p = format 244, 854×384 vp9; 720p = format 247, 1280×576 vp9) but **media downloads 403** (PoToken/JS-runtime — same class of failure as the server bot-check). Fallback used: ffmpeg transcode of the local 1080p.
- Transcode cost (libx264 veryfast crf 23, 120 s): **480p 11.3 s, 720p 12.2 s** — comparable to one decode pass; a dedicated proxy encode would pay for itself if tracking runs more than ~1.2× on the same file.

| Variant | Decode-only | Full compute_rows (fps=1, imgsz 960) | ball_rate | mean players |
|---|---|---|---|---|
| 1080p vp9 baseline | 9.3 s | **13.5 s** | 0.175 | 7.4 |
| 480p proxy (854×384, h264) | 0.75 s | **4.5 s** | **0.175** | 7.2 |
| 720p proxy (1280×576, h264) | 1.9 s | **5.0 s** | 0.100 | 7.4 |

- File size per hour (this clip, crf 23): 480p ≈ **0.33 GB/h**, 720p ≈ **1.2 GB/h**, vs the 1080p vp9 source ≈ 3.9 GB/h.
- 480p → **3.0×** end-to-end speedup (13.5 → 4.5 s); decode share collapses from 69% to ~17% — remaining time is YOLO + pipe reads.
- Accuracy: 480p *matched* the 1080p ball rate on this window (0.175, players 7.2) — the 960-wide upscale fed to YOLO apparently preserves small-ball detectability; 720p's lower ball_rate (0.100 vs 0.175) is likely detection noise on this single 120 s sample — one window is thin evidence; validate on a labeled stretch before shipping.
- Caveat: proxy is re-encoded h264 (I-frame cadence normal) — no sparse-keyframe issue; also cheaper per GB stored. If the download path is fixed (fresh PoToken), YouTube's own 480p vp9 stream would likely behave similarly.
