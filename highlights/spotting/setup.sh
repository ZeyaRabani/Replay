#!/usr/bin/env bash
# Install CPU deps and fetch the released E2E-Spot SoccerNet-v2 checkpoint (~50 MB).
set -euo pipefail
cd "$(dirname "$0")"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q timm numpy pillow tqdm
mkdir -p third_party
if [ ! -d third_party/e2e-spot-models ]; then
  git clone -q --depth 1 https://github.com/jhong93/e2e-spot-models.git third_party/e2e-spot-models
fi
ls third_party/e2e-spot-models/soccer_challenge_rny008gsm_gru_rgb/
