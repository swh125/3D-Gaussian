#!/usr/bin/env bash
set -euo pipefail

# Purpose: Train 3D Gaussian affinity (contrastive) features.
# Usage:
#   chmod +x scripts/train_affinity.sh
#   ./scripts/train_affinity.sh

# --------- USER CONFIG REQUIRED ----------
MODEL_PATH="${MODEL_PATH:-}"         # REQUIRED: trained 3DGS model directory
ITERATIONS="${ITERATIONS:-10000}"
NUM_SAMPLED_RAYS="${NUM_SAMPLED_RAYS:-1000}"
# -----------------------------------------

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: Please set MODEL_PATH to your trained 3DGS model directory."
  exit 1
fi

echo "[SAGA] Training contrastive features..."
python train_contrastive_feature.py \
  -m "${MODEL_PATH}" \
  --iterations "${ITERATIONS}" \
  --num_sampled_rays "${NUM_SAMPLED_RAYS}"

echo "Done."




