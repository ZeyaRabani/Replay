#!/usr/bin/env bash
# Re-run projection + fusion + post-processing on the real footage using cached detections.
# Put the three synced clips (cam0.mp4 cam1.mp4 cam2.mp4, 2316x1080 @ 30 fps, ~24 s) in $OUT/synced/.
# Detections are reused from debug/raw_tracks.json so no Modal/GPU is needed; drop --reuse to re-detect.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-/tmp/pitchworld_real}"
mkdir -p "$OUT/debug" "$OUT/synced"
cp "$HERE/debug/raw_tracks.json" "$OUT/debug/raw_tracks.json"
for i in 0 1 2; do [ -f "$OUT/synced/cam$i.mp4" ] || { echo "missing $OUT/synced/cam$i.mp4"; exit 1; }; done
pitchworld run "$OUT/synced/cam0.mp4" "$OUT/synced/cam1.mp4" "$OUT/synced/cam2.mp4" \
  --out "$OUT" --offsets 0 0 0 --reuse \
  --calib "$ROOT/examples/real_footage_calib.json" \
  --pitch "$ROOT/examples/pitch_small_sided_50x30.json" "$@"
