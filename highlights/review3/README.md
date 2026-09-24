# review3: independent visual review (2700–4050 s)

Human-style ground-truth review of the requested video segment. This track intentionally uses no event detector.

## Setup

```bash
pip install -U yt-dlp
sudo apt-get install ffmpeg
```

Download the source:

```bash
mkdir -p ~/match
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
  --merge-output-format mp4 -o ~/match/match.mp4 \
  https://youtu.be/5qj_nsQSzvQ
```

## Generate review sheets

Coarse full-frame pass (12 frames / sheet, 24 s / sheet):

```bash
python highlights/review3/make_sheets.py ~/match/match.mp4 \
  --start 2700 --end 4050 --step 2 --cols 4 --rows 3 \
  --out ~/match/sheets/coarse
```

Far-goal crop pass:

```bash
python highlights/review3/make_sheets.py ~/match/match.mp4 \
  --start 2700 --end 4050 --step 2 --cols 4 --rows 3 \
  --crop 'iw*0.45:ih:0:0' --out ~/match/sheets/far
```

Fine pass for a candidate window (4 fps, 5 s / sheet):

```bash
python highlights/review3/make_sheets.py ~/match/match.mp4 \
  --start 3438 --end 3464 --fps 4 --cols 5 --rows 4 \
  --tile-w 400 --out ~/match/sheets/fine_3438
```

## Outputs

- `outputs/review3_events.json`: events in the shared schema.
- `outputs/evidence/*.png`: timestamped fine-pass sheet for every claimed event.
- `review3_log.md`: per-sheet coarse-pass log and fine-pass conclusions.

## Measured runtime

On the provided 8-core CPU machine:

- Video download and merge: approximately 2 minutes (network was unusually fast).
- 57 full-frame coarse sheets: 20 seconds.
- 57 far-goal crop sheets: 20 seconds.
- Candidate fine sheets: approximately 2 minutes total.
- Systematic visual review, fine review, logging, and packaging: approximately 55 minutes wall-clock.

## Known limitations

- The fixed ground-level camera gives a clear near goal but an extremely small far goal. A far-end strike or ball crossing cannot usually be resolved directly.
- Event time precision is about ±0.5 s for near-goal fine-pass events and ±1–2 s for far-goal events.
- The low-confidence far-goal event at 2851.8 s is inferred from the attack sequence and player reaction, not direct ball visibility.
- No goals were directly observed in this segment. Near-goal attempts were checked for follow-up celebration and centre restart before ruling them non-goals.
