# hl3 / audio — crowd & player audio excitement (secondary signal)

Detects moments of elevated audio activity (shouting, ball strikes, keeper/crowd
reaction) and referee whistles from the match audio. CPU only, no cloud APIs.
All times are **seconds from the start of the video file**.

## Setup

```bash
sudo apt-get install -y ffmpeg            # if missing
pip install -U yt-dlp librosa soundfile scipy numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install panns-inference               # PANNs CNN14 AudioSet tagger (weights ~300 MB, auto-downloaded to ~/panns_data on first run)
```

## Run on a video

```bash
cd highlights/audio
V=~/match/match.mp4
python extract_audio.py $V ~/match/match16k.wav                                          # mono 16 kHz WAV
python features.py ~/match/match16k.wav outputs/features_1s.json --whistles outputs/whistles.json
python features.py ~/match/match16k.wav outputs/features_0.5s.json --step 0.5             # optional finer grid
python tagger.py   ~/match/match16k.wav outputs/panns_1s.json                             # PANNs CNN14, 2 s windows / 1 s hop
python events.py   outputs/features_1s.json outputs/events.json \
                   --panns outputs/panns_1s.json --whistles outputs/whistles.json --match-start 1020
python plot.py     outputs/features_1s.json outputs/events.json outputs/energy_zscore.png --panns outputs/panns_1s.json
python clips.py    $V outputs/events.json ~/match/clips -n 8                              # 6 s clips + 3x2 frame sheets for review
```

`--match-start` (default 1020 s for this video) marks the kick-off: audio peaks
before it are emitted as `type:"other", signals.kind:"warmup"` so a downstream
merger can exclude the warm-up shooting drills; whistles before it get
`signals.warmup=true`.

## Measured runtime (8-core CPU, 31 GB RAM, 89-min video)

| step | time |
|---|---|
| ffmpeg audio extraction | 4.5 s |
| `features.py` (1 s grid + frame-level whistles) | ~45 s |
| `features.py` 0.5 s grid | ~25 s |
| `tagger.py` PANNs CNN14 (5336 windows, 8 threads) | 57 s (+ one-off weight download) |
| `events.py` + `plot.py` | < 5 s |
| **total** | **~2.5 min** |

## Method

`features.py` (STFT 1024 / hop 10 ms, aggregated to the grid):

| column | meaning |
|---|---|
| `rms_db` | overall level |
| `low_db`, `speech_db`, `high_db` | band power 0–300 Hz (wind/handling), 300–3000 Hz (voices, shouts, claps, ball strikes), 3–8 kHz |
| `speech_ratio` | share of power in 300–3000 Hz |
| `flux`, `onset_density` | spectral flux (mean), librosa onsets per second |
| `centroid_hz`, `flatness` | spectral centroid / flatness |
| `whistle_max`, `whistle_frac`, `whistle_hz` | tonality of the 2–4.5 kHz band (peak bin / band mean, gated by band share), fraction of frames whistle-like, median peak freq |
| `z60_rms`, `z300_rms` | z-score of `rms_db` vs rolling 60 s / 300 s window |
| `z300_speech`, `z300_speech_s5` | z-score of `speech_db` vs rolling 300 s, and its 5 s moving average (the excitement score) |

Whistles (`detect_whistles`): frame-level runs ≥ 0.25 s where the 2.2–4.5 kHz
band is strongly tonal with a stable peak frequency; ≤ 0.15 s dropouts are
bridged so trills count once. Confidence from duration, tonality and
frequency stability.

Excitement events (`events.py`): peaks of `z300_speech_s5 + 2·PANNs(cheer/shout/applause…)`
above 1.5 with ≥ 20 s separation; the supporting window is the contiguous region
where z stays above half the peak. Confidence = f(z, duration, tagger support,
whistle within ±5 s). `signals` carries all the numbers plus the max PANNs
probabilities for Cheering/Applause/Crowd/Shout/Speech/Whistle/Wind in the window.

Half-time: the longest ≥ 2 min stretch with both level and onset density in the
bottom quartile (60 s smoothed). **None found** in this video — the half-time
break appears to have been cut from the recording (or play was continuous).

## Outputs (`outputs/`)

- `events.json` — shared hl3 schema. 239 events: 35 `excitement`, 4 `other/warmup` peaks, 200 `other/whistle`.
- `features_1s.json`, `features_0.5s.json` — dense feature tables (`step_s`, `columns`, `rows`).
- `panns_1s.json` — PANNs CNN14 class probabilities per second (17 AudioSet classes).
- `whistles.json` — frame-level whistle segments.
- `energy_zscore.png` — levels, z-score with peaks (red = excitement, grey = warm-up, green = whistles), PANNs probabilities.
- `top8_frames/` — 3×2 frame sheets (1 fps) for the 6 s clips around the top-8 excitement peaks.

## Honest assessment — how informative is audio here?

**There is essentially no crowd.** PANNs CNN14 gives `Cheering` ≤ 0.035 and
`Applause` ≤ 0.015 over the whole match (means ≈ 0.000); the dominant classes
are `Speech` (mean 0.68 — players calling for the ball) and `Wind`. The tagger
therefore adds almost nothing on this footage; the hand-crafted speech-band
z-score is what carries the signal.

What the z-score actually measures is *how loud the players are near the
microphone*. Because the camera sits ~10 m from the near goal, near-goal
attacks (keeper shouting, defenders, ball hitting net/post, celebrations) are
loud and far-goal attacks are barely audible. Review of the top-8 peaks
(frames in `outputs/top8_frames/`):

| rank | t (s) | what the frames show |
|---|---|---|
| 1 | 1354 (22:34) | ball high in the air over the near box, keeper reacting, players on the arc — near-goal shot/clearance; whistle at 1353 & 1359 |
| 2 | 4747 (79:07) | 6+ players packed in the near goal mouth, orange player walks right past the camera — near-goal set-piece/scramble (20 s sustained) |
| 3 | 3057 (50:57) | attack on the near goal, keeper off his line, players in the box |
| 4 | 4550 (75:50) | scramble on the near goal line, camera gets knocked (frame tilts), orange player exits past the lens — likely goal / goal-line incident |
| 5 | 3524 (58:44) | near-goal attack, keeper gathers |
| 6 | 1443 (24:03) | play at the far end, near keeper idle — loud calling only (possible far-goal chance; cannot confirm from audio) |
| 7 | 4416 (73:36) | play at the far end, near keeper idle — same as above |
| 8 | 1710 (28:30) | cluster of 5 players tussling at the near post |

So ~6/8 top peaks coincide with near-goal action; the other two are loud
talk with play far away. Audio alone cannot tell a goal from a scramble or a
foul, and it is strongly biased toward the near goal. It is useful as a
**secondary/confirming signal** (boost visual candidates that coincide with a
z ≥ 1.5 peak, especially when a whistle follows within ~5 s), not as a primary
detector.

Whistles: 200 segments detected (many are the referee's short blasts; some are
from adjacent pitches — the sustained 2.9 kHz tone at 180–191 s during warm-up
is one such). Peak frequencies cluster at 2.3–3.8 kHz. Precision was not
measured against ground truth; treat `confidence < 0.6` whistles as weak.

## Known limitations

- Baseline is a rolling 300 s window, so a burst right after a long quiet stretch scores higher than an equally loud burst during a noisy spell.
- Wind gusts (PANNs `Wind` up to 0.47) leak into `rms_db`; the speech-band z-score is mostly robust to this but not fully.
- Speech near the camera (bystanders, substitutes) produces false excitement peaks.
- Whistle detector is a tonality heuristic; birds, squeaky shoes on artificial turf and neighbouring-pitch whistles can trigger it.
- `--match-start` is a manual constant for this video; there is no automatic kick-off detection.
