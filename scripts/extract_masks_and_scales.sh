#!/usr/bin/env bash
set -euo pipefail

# Purpose: Extract SAM masks and compute mask scales from a trained 3DGS model.
# Usage:
#   chmod +x scripts/extract_masks_and_scales.sh
#   ./scripts/extract_masks_and_scales.sh
#
# Requires: SAM checkpoint downloaded to third_party/segment-anything/sam_ckpt.

# --------- USER CONFIG REQUIRED ----------
IMAGE_ROOT="${IMAGE_ROOT:-/home/bygpu/data/book_scene}"    # processed scene root
MODEL_PATH="${MODEL_PATH:-}"                   # REQUIRED: trained 3DGS model directory (output of train_scene.py)
SAM_CKPT="${SAM_CKPT:-third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth}"
DOWNSAMPLE="${DOWNSAMPLE:-2}"                  # 1/2/4/8
# -----------------------------------------

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: Please set MODEL_PATH to your trained 3DGS model directory."
  exit 1
fi

echo "[SAGA] Extracting SAM masks..."
python extract_segment_everything_masks.py \
  --image_root "${IMAGE_ROOT}" \
  --sam_checkpoint_path "${SAM_CKPT}" \
  --downsample "${DOWNSAMPLE}"

echo "[SAGA] Computing mask scales..."
python get_scale.py \
  --image_root "${IMAGE_ROOT}" \
  --model_path "${MODEL_PATH}"

echo "Done. Masks in ${IMAGE_ROOT}/sam_masks, scales in ${IMAGE_ROOT}/mask_scales"




