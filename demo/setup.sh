#!/usr/bin/env bash
# One-shot setup for the Replay demo-video toolchain. Idempotent. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system tools"
command -v ffmpeg >/dev/null || { echo "ffmpeg missing (apt install ffmpeg)"; exit 1; }
command -v node >/dev/null || { echo "node missing (nvm install 24)"; exit 1; }

echo "== VEED open-edit skill (agent skill + CLI)"
# Installs .agents/skills/open-edit/SKILL.md (+ 20 other agent dirs). Renders only on macOS arm64 / Windows x64.
npx -y skills add veedstudio/open-edit --skill open-edit -y >/dev/null 2>&1 || npx -y skills add veedstudio/open-edit --skill open-edit
# Preflight: on Linux this prints "unsupported platform linux/x64" and exits 1 -> we fall back to ffmpeg.
if npx --yes @veedstudio/openedit-cli init --dry --workspace "$PWD"; then
  echo "open-edit renderer available: npx --yes @veedstudio/openedit-cli init --workspace $PWD"
else
  echo "open-edit renderer NOT available on this host; make_demo.py will use ffmpeg (and still emit a .wv composition)."
fi

echo "== WhisperX (CPU) for narration captions"
if [ ! -x .venv-whisperx/bin/whisperx ]; then
  python3 -m venv .venv-whisperx
  .venv-whisperx/bin/pip install -U pip
  .venv-whisperx/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  .venv-whisperx/bin/pip install whisperx          # note: re-resolves torch 2.8 from PyPI (~2 min, several GB)
fi
.venv-whisperx/bin/whisperx --help >/dev/null && echo "whisperx OK"

echo "== recorder deps (Playwright over Chrome CDP)"
python3 -m pip install -q playwright

echo "done. next: see demo/README.md"
