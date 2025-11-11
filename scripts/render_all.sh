#!/usr/bin/env bash
set -euo pipefail

# Purpose: Render baseline, segmented scene, and 2D masks.
# Usage:
#   chmod +x scripts/render_all.sh
#   ./scripts/render_all.sh
#
# Provide MODEL_PATH and optional PRECOMPUTED_MASK (saved by GUI or notebook).

# --------- USER CONFIG (MODEL_PATH 需在训练完成后填写) ----------
MODEL_PATH="${MODEL_PATH:-}"                         # 训练输出目录 (train_baseline.sh 运行后终端会打印)
PRECOMPUTED_MASK="${PRECOMPUTED_MASK:-}"             # 例如 ./segmentation_res/book_object.pt
# -------------------------------------------------------------

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: Please set MODEL_PATH to your trained 3DGS model directory."
  exit 1
fi

echo "[Render] Baseline scene..."
python render.py -m "${MODEL_PATH}" --target scene

if [[ -n "${PRECOMPUTED_MASK}" ]]; then
  echo "[Render] Scene with segmentation (foreground only)..."
  python render.py -m "${MODEL_PATH}" --precomputed_mask "${PRECOMPUTED_MASK}" --target scene --segment

  echo "[Render] 2D masks..."
  python render.py -m "${MODEL_PATH}" --precomputed_mask "${PRECOMPUTED_MASK}" --target seg
else
  echo "Note: PRECOMPUTED_MASK not set; skipping segmented renders."
fi

echo "Done."



