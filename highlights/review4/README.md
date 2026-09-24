# review4 — manual visual ground-truthing, 4050 s → end of file

Human-style review of the last ~21.5 minutes of the match video (4050–5337 s) using
timestamped contact sheets. No detector; every event here was seen by eye.

## Setup

```bash
sudo apt-get install -y ffmpeg   # ffmpeg + ffprobe
pip install -U yt-dlp
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
  --merge-output-format mp4 -o ~/match/match.mp4 https://youtu.be/5qj_nsQSzvQ
```

## Run

```bash
cd highlights/review4
# coarse pass: 1 frame / 2 s, 5x4 tiles (40 s per sheet), timestamp burned in
python make_sheets.py coarse ~/match/match.mp4 ~/sheets/coarse --start 4050
# fine pass over an interesting window (2 fps, 10 s per sheet)
python make_sheets.py fine ~/match/match.mp4 ~/sheets/fine --start 4542 --end 4562 --fps 2
# zoom on the near goal (right part of the 1280x576 frame) at 4 fps
python make_sheets.py fine ~/match/match.mp4 ~/sheets/zoom --start 4544 --end 4554 --fps 4 \
  --crop 760:430:520:60 --tag _near
# zoom on the far goal (top-left) at 1 frame / 2 s
python make_sheets.py coarse ~/match/match.mp4 ~/sheets/far --start 4050 --end 5050 \
  --crop 800:260:0:20 --tag _far
```

Then view every sheet and write down what you see (see `outputs/review4_log.md`).

## Outputs

- `outputs/review4_events.json` — events in the shared schema (`source: "review4"`).
- `outputs/review4_log.md` — per-sheet notes from the coarse and fine passes.
- `outputs/evidence/*.jpg` — fine-pass sheets backing each claimed event (JPEG re-encodes of the PNG sheets).

## Runtime (8-core CPU, no GPU)

- Download: ~3 min (fast datacenter link).
- Coarse sheets (33 sheets, full frame): 46 s wall.
- Fine/zoom sheets (~50 sheets): ~2–3 min total.
- Viewing + note taking: ~1 h.

## Result summary

- **0 goals observed** in 4050–5337 s.
- Two big near-goal chances with a keeper dive: **~4548 s** (1-v-1, keeper beaten, blocked on the line — conf 0.7 shot, 0.15 goal) and **~4979.5 s** (diving save / wide, immediately followed by full time — conf 0.6 shot, 0.2 goal).
- Several lower-confidence near-goal chances (4089, 4240, 4297, 4314, 4396, 4741, 4790, 4885).
- Match ends ~4986 s; pitch empty from ~5050 s; camera picked up ~5326 s.

## Known limitations

- The ball is rarely visible at 640 px tile width; most "shot" calls are inferred from keeper dives, players retrieving the ball behind the goal and the restart type (goal kick / corner), so the strike second is ±1–2 s.
- The far goal is ~20 px tall; attempts at the far end (lime attacking) cannot be verified. Only one low-confidence far-end "chance" is recorded; a far-end goal would only have been detectable via a kickoff formation, none was seen.
- Coarse pass samples every 2 s, so a very quick far-end event between samples could be missed.
- `t` for chances without a visible strike is the moment the ball went dead / the peak of the action, not the kick.
