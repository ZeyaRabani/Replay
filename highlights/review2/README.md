# review2: visual ground truth, 1350–2700 s

Independent human-style review of the middle 22.5 minutes of the supplied match. This is not a detector: every coarse sheet was viewed, candidates were reviewed at higher temporal resolution, and event labels were written manually.

## Setup

```bash
python3 -m pip install -U yt-dlp
sudo apt-get install -y ffmpeg imagemagick  # omit if already installed
mkdir -p ~/match
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
  --merge-output-format mp4 -o ~/match/match.mp4 \
  https://youtu.be/5qj_nsQSzvQ
```

`make_sheets.py` itself only requires Python 3 and ffmpeg.

## Run

Full-frame coarse pass (one frame every 2 s, 40 s per sheet):

```bash
python3 make_sheets.py ~/match/match.mp4 ~/match/review2-coarse \
  --start 1350 --end 2700 --step 2 --cols 5 --rows 4
```

Far-goal zoom pass:

```bash
python3 make_sheets.py ~/match/match.mp4 ~/match/review2-far \
  --start 1350 --end 2700 --step 2 --crop 640:220:0:30 \
  --width 800 --cols 4 --rows 5 --prefix far
```

Fine pass for a candidate window (4 fps):

```bash
python3 make_sheets.py ~/match/match.mp4 ~/match/review2-fine \
  --start 1483 --end 1488 --step 0.25 --crop 900:500:380:30 \
  --width 640 --cols 5 --rows 4 --prefix event
```

The committed result is `outputs/events.json`; fine-pass proof sheets are under `outputs/evidence/`. `review2_log.md` records the coarse review.

## Measured runtime

On this CPU-only 8-core machine:

- Download and merge: about 2 minutes (network was unusually fast, roughly 10–70 MB/s).
- 34 full-frame coarse sheets: 45.8 seconds.
- 34 far-goal crop sheets: about 1 minute.
- Fine-sheet extraction: about 3 minutes total.
- Manual viewing and annotation: about 65 minutes.
- End-to-end active runtime: about 72 minutes.

## Known limitations

- The fixed ground-level corner camera makes the near goal clear but the far goal extremely small. Far-goal ball flight is often impossible to follow even after cropping; only one far event is included, with reduced confidence.
- Player occlusion is severe in crowded near-goal sequences. Ambiguous attempts are labelled `chance` or assigned lower confidence rather than promoted to shots/goals.
- No goal was directly observed in 1350–2700 s. There was no convincing celebration plus centre-circle restart in the segment.
- Event timestamps are visual estimates from 4 fps sheets, generally within about 0.5–1.0 s.
