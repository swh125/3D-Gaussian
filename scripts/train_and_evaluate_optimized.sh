#!/usr/bin/env bash
# Train optimized 3DGS with 320 train + 15 test split, then compute metrics.
# Usage:
#   bash scripts/train_and_evaluate_optimized.sh <DATA_ROOT> <OUTPUT_MODEL>
# Example:
#   bash scripts/train_and_evaluate_optimized.sh /home/bygpu/data/video_scene ./output/video_scene_optimized_320train

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/train_and_evaluate_optimized.sh <DATA_ROOT> <OUTPUT_MODEL>"
    echo "  <DATA_ROOT>    : ns-process-data output (contains images/ + sparse/)"
    echo "  <OUTPUT_MODEL> : destination directory for trained model"
    echo ""
    echo "Optional environment variables:"
    echo "  ITERATIONS           (default: 30000)"
    echo "  TEST_LAST            (default: 15, for 320 train + 15 test)"
    echo "  POSITION_LR_INIT     (default: 0.00016)"
    echo "  POSITION_LR_FINAL    (default: 0.0000016)"
    echo "  LAMBDA_DSSIM         (default: 0.2)"
    echo "  DENSIFY_FROM_ITER    (default: 500)"
    echo "  DENSIFY_UNTIL_ITER   (default: 15000)"
    exit 1
fi

DATA_ROOT="$1"
MODEL_OUTPUT="$2"

# Default hyperparameters (can be tuned for optimization)
ITERATIONS="${ITERATIONS:-30000}"
TEST_LAST="${TEST_LAST:-15}"  # 320 train + 15 test
POSITION_LR_INIT="${POSITION_LR_INIT:-0.00016}"
POSITION_LR_FINAL="${POSITION_LR_FINAL:-0.0000016}"
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0.2}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-500}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"

echo "==============================================="
echo "Optimized 3DGS Training & Evaluation"
echo "==============================================="
echo "Data root        : ${DATA_ROOT}"
echo "Model output     : ${MODEL_OUTPUT}"
echo "Iterations       : ${ITERATIONS}"
echo "Train/Test split : 320 train + ${TEST_LAST} test"
echo "Position LR init : ${POSITION_LR_INIT}"
echo "Position LR final: ${POSITION_LR_FINAL}"
echo "Lambda DSSIM     : ${LAMBDA_DSSIM}"
echo "Densify range    : ${DENSIFY_FROM_ITER} - ${DENSIFY_UNTIL_ITER}"
echo ""

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
    exit 1
fi
if [[ -e "${MODEL_OUTPUT}" && -n "$(ls -A "${MODEL_OUTPUT}" 2>/dev/null)" ]]; then
    echo "WARNING: MODEL_OUTPUT already exists: ${MODEL_OUTPUT}"
    read -p "Continue and overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "==============================================="
echo "[Step 1/4] Training optimized 3DGS"
echo "==============================================="
python train_scene.py \
  -s "${DATA_ROOT}" \
  --model_path "${MODEL_OUTPUT}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --position_lr_init "${POSITION_LR_INIT}" \
  --position_lr_final "${POSITION_LR_FINAL}" \
  --lambda_dssim "${LAMBDA_DSSIM}" \
  --densify_from_iter "${DENSIFY_FROM_ITER}" \
  --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
  --save_iterations "${ITERATIONS}"

echo ""
echo "==============================================="
echo "[Step 2/4] Rendering train set"
echo "==============================================="
python render.py \
  -m "${MODEL_OUTPUT}" \
  -s "${DATA_ROOT}" \
  --target scene \
  --iteration "${ITERATIONS}" \
  --skip_test

echo ""
echo "==============================================="
echo "[Step 3/4] Rendering test set"
echo "==============================================="
python render.py \
  -m "${MODEL_OUTPUT}" \
  -s "${DATA_ROOT}" \
  --target scene \
  --iteration "${ITERATIONS}" \
  --skip_train

echo ""
echo "==============================================="
echo "[Step 4/4] Computing metrics"
echo "==============================================="
echo "--- Train Set Metrics ---"
python scripts/compute_metrics.py \
  --model_path "${MODEL_OUTPUT}" \
  --set train \
  --iteration "${ITERATIONS}"

echo ""
echo "--- Test Set Metrics ---"
python scripts/compute_metrics.py \
  --model_path "${MODEL_OUTPUT}" \
  --set test \
  --iteration "${ITERATIONS}"

echo ""
echo "==============================================="
echo "Training and evaluation complete!"
echo "==============================================="
echo "Model output: ${MODEL_OUTPUT}"
echo "Train set: 320 frames"
echo "Test set: ${TEST_LAST} frames"
echo ""
echo "Metrics saved in output directory."
echo "Check train/ours_${ITERATIONS}/ and test/ours_${ITERATIONS}/ for renders."

