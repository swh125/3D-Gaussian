#!/usr/bin/env bash
# Optimized 3DGS training pipeline (render + metrics)
# Usage: bash scripts/train_optimized_scene.sh <DATA_ROOT> <MODEL_PATH>
# Example:
#   bash scripts/train_optimized_scene.sh /home/user/data/video_scene ./output/video_scene_opt_$(date +%Y%m%d_%H%M%S)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/train_optimized_scene.sh <DATA_ROOT> <MODEL_PATH>"
    echo "  <DATA_ROOT>: directory containing COLMAP outputs (images/ + sparse/)"
    echo "  <MODEL_PATH>: destination directory for optimized model checkpoints"
    exit 1
fi

DATA_ROOT="$1"
MODEL_PATH="$2"

ITERATIONS_OPT="${ITERATIONS_OPT:-35000}"
TEST_LAST="${TEST_LAST:-40}"

echo "==============================================="
echo "Optimized 3DGS Training"
echo "==============================================="
echo "Data root        : ${DATA_ROOT}"
echo "Model output     : ${MODEL_PATH}"
echo "Iterations       : ${ITERATIONS_OPT}"
echo "Test tail frames : ${TEST_LAST}"
echo ""

python train_scene.py \
  -s "${DATA_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --iterations "${ITERATIONS_OPT}" \
  --position_lr_init 0.00012 \
  --position_lr_final 0.000001 \
  --densify_until_iter 28000 \
  --densify_grad_threshold 0.00015 \
  --opacity_reset_interval 2500 \
  --lambda_dssim 0.15 \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "==============================================="
echo "Rendering optimized model (train + test)"
echo "==============================================="
python render.py \
  -m "${MODEL_PATH}" \
  -s "${DATA_ROOT}" \
  --target scene \
  --iteration "${ITERATIONS_OPT}"

echo ""
echo "==============================================="
echo "Computing metrics"
echo "==============================================="
python scripts/compute_metrics.py \
  --model_path "${MODEL_PATH}" \
  --set train \
  --iteration "${ITERATIONS_OPT}"

python scripts/compute_metrics.py \
  --model_path "${MODEL_PATH}" \
  --set test \
  --iteration "${ITERATIONS_OPT}"

echo ""
echo "Done. Outputs in ${MODEL_PATH}"

