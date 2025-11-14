#!/usr/bin/env bash
# Train baseline 3DGS model with 320 train / 15 test split
# Then render and compute metrics (PSNR/SSIM/LPIPS)

set -euo pipefail

# Configuration
SCENE_ROOT="${SCENE_ROOT:-/home/bygpu/data/video_scene}"
MODEL_BASELINE="${MODEL_BASELINE:-./output/video_scene_baseline_320_15_$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30000}"
TEST_LAST="${TEST_LAST:-15}"  # Last 15 frames for test (320 train, 15 test)

echo "==============================================="
echo "Baseline 3DGS Training & Evaluation"
echo "==============================================="
echo "Scene root      : ${SCENE_ROOT}"
echo "Model path      : ${MODEL_BASELINE}"
echo "Iterations      : ${ITERATIONS}"
echo "Train/Test split: 320 / 15"
echo ""

# Step 1: Train baseline model
echo "[1/3] Training baseline 3D Gaussian Splatting..."
python train_scene.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_BASELINE}" \
  --iterations "${ITERATIONS}" \
  --test_iterations 7000 "${ITERATIONS}" \
  --save_iterations 7000 "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "[2/3] Rendering test set..."
python render.py \
  -m "${MODEL_BASELINE}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "[3/3] Computing metrics..."
python scripts/compute_metrics.py \
  --model_path "${MODEL_BASELINE}" \
  --set test \
  --iteration "${ITERATIONS}"

echo ""
echo "==============================================="
echo "Training and evaluation complete!"
echo "==============================================="
echo "Model path: ${MODEL_BASELINE}"
echo "Metrics saved in: ${MODEL_BASELINE}/test/ours_${ITERATIONS}/"

