# review1: independent visual ground-truthing (0–1350 s)

This track manually reviews video seconds 0–1350. It does not run an event detector. Timestamp convention is seconds from the start of the video file.

## Setup

```bash
pip install -U yt-dlp
sudo apt-get install -y ffmpeg fonts-dejavu-core  # if ffmpeg/fonts are absent
mkdir -p ~/match
yt-dlp -f 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]' \
  --merge-output-format mp4 -o ~/match/match.mp4 \
  https://youtu.be/5qj_nsQSzvQ
```

## Reproduce contact sheets

```bash
# 34 coarse sheets, 40 seconds per sheet
python highlights/review1/make_sheets.py coarse ~/match/match.mp4 ~/match/coarse \
  --start 0 --end 1350 --step 2

# Fine pass around a candidate window (4 fps)
python highlights/review1/make_sheets.py fine ~/match/match.mp4 ~/match/fine \
  --start 902 --end 912 --fps 4

# Fine pass crop-zooming the far-goal area
python highlights/review1/make_sheets.py far ~/match/match.mp4 ~/match/far \
  --start 1100 --end 1350 --fps 1 --crop 640:220:0:0
```

Review `review1_log.md`, `outputs/review1_events.json`, and the timestamped PNG evidence under `outputs/evidence/`.

## Measured runtime

On the provided CPU machine, generating all 34 coarse sheets took **55.6 s**. Candidate fine sheets took about **2 min** total. Downloading the 720p video took about **2 min** on this run. The systematic coarse and fine visual review took about **35 min**.

## Findings and limitations

The actual lime/yellow-vs-orange match begins at approximately **1104.8 s**; preceding activity is setup and warm-up. Three representative, directly observed near-goal warm-up attempts are labeled `other` with `signals.kind = "warmup"`. No match goal, shot, or big chance was directly observed from kickoff through 1350 s.

The fixed corner camera makes near-goal action clear but renders far-goal players tiny. A separate full-resolution far-region crop was inspected for every match second at 1 fps, plus 4 fps around candidate clusters. Subtle far-goal attempts without a visible strike, reaction, celebration, or restart could still be missed. Shot outcomes in the warm-up are often obscured by the near net and goal frame, so they are intentionally not labeled as goals/saves.
