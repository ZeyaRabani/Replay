#!/usr/bin/env bash
# Generate a 120 s 640x360 25fps test video with AAC audio.
set -euo pipefail
out="${1:-$(dirname "$0")/sample_120s.mp4}"
ffmpeg -y \
  -f lavfi -i "testsrc=duration=120:size=640x360:rate=25" \
  -f lavfi -i "sine=frequency=440:duration=120" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 96k \
  -movflags +faststart "$out"
echo "wrote $out"
