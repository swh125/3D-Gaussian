#!/usr/bin/env bash
# Retrain both baseline and optimized models from scratch
# Usage:
#   bash scripts/retrain_baseline_and_optimized.sh <SCENE_ROOT>
# Example:
#   bash scripts/retrain_baseline_and_optimized.sh /home/bygpu/data/video_scene

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/retrain_baseline_and_optimized.sh <SCENE_ROOT>"
  echo "  <SCENE_ROOT> : Path to processed scene (contains images/ + sparse/)"
  exit 1
fi

SCENE_ROOT="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_BASELINE="./output/baseline_${TIMESTAMP}"
MODEL_OPTIMIZED="./output/optimized_${TIMESTAMP}"
ITERATIONS=30000
TEST_LAST=25  # 310 train, 25 test (same split for both)

# Optimized hyperparameters
LAMBDA_EDGE=0.1  # Edge loss weight (set to 0 to disable)

echo "==============================================="
echo "Retraining Baseline and Optimized Models"
echo "==============================================="
echo "Scene root       : ${SCENE_ROOT}"
echo "Baseline model   : ${MODEL_BASELINE}"
echo "Optimized model  : ${MODEL_OPTIMIZED}"
echo "Iterations       : ${ITERATIONS}"
echo "Train/Test split : 310 / ${TEST_LAST}"
echo "Lambda edge      : ${LAMBDA_EDGE}"
echo ""

# ============================================
# Step 1: Train Baseline
# ============================================
echo "[1/4] Training baseline model..."
python train_scene.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_BASELINE}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --save_iterations "${ITERATIONS}"

echo ""
echo "✓ Baseline training complete: ${MODEL_BASELINE}"

# ============================================
# Step 2: Train Optimized
# ============================================
echo ""
echo "[2/4] Training optimized model..."
python train_scene_optimized.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_OPTIMIZED}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --lambda_edge "${LAMBDA_EDGE}" \
  --save_iterations "${ITERATIONS}"

echo ""
echo "✓ Optimized training complete: ${MODEL_OPTIMIZED}"

# ============================================
# Step 3: Render test sets
# ============================================
echo ""
echo "[3/4] Rendering test sets..."

echo "  Rendering baseline test set..."
python render.py \
  -m "${MODEL_BASELINE}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo "  Rendering optimized test set..."
python render.py \
  -m "${MODEL_OPTIMIZED}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

# ============================================
# Step 4: Compute metrics
# ============================================
echo ""
echo "[4/4] Computing metrics..."

echo "  Computing baseline metrics..."
python scripts/compute_metrics.py \
  --model_path "${MODEL_BASELINE}" \
  --set test \
  --iteration "${ITERATIONS}"

echo "  Computing optimized metrics..."
python scripts/compute_metrics.py \
  --model_path "${MODEL_OPTIMIZED}" \
  --set test \
  --iteration "${ITERATIONS}"

echo ""
echo "==============================================="
echo "Retraining Complete!"
echo "==============================================="
echo "Baseline model  : ${MODEL_BASELINE}"
echo "Optimized model : ${MODEL_OPTIMIZED}"
echo ""
echo "Next steps:"
echo "1. Open GUI to segment objects:"
echo "   python saga_gui.py --model_path ${MODEL_BASELINE}"
echo "   python saga_gui.py --model_path ${MODEL_OPTIMIZED}"
echo ""
echo "2. Compute IoU after segmentation:"
echo "   python scripts/compute_segmentation_iou.py \\"
echo "     --pred_mask_3d ./segmentation_res/<object>_baseline.pt \\"
echo "     --gt_mask_dir <path_to_gt_masks> \\"
echo "     --model_path ${MODEL_BASELINE} \\"
echo "     --source_path ${SCENE_ROOT} \\"
echo "     --iteration ${ITERATIONS}"
echo ""




