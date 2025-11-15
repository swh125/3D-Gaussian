#!/usr/bin/env bash
# Train optimized 3DGS model with 320 train / 15 test split
# Then render and compute metrics (PSNR/SSIM/LPIPS)

set -euo pipefail

# Configuration
MODEL_BASELINE="${MODEL_BASELINE:-./output/video_scene_20251116_012233}"
MODEL_OPT="${MODEL_OPT:-./output/video_scene_optimized_$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-30000}"
TEST_LAST="${TEST_LAST:-25}"  # Last 25 frames for test (310 train, 25 test)

# Auto-detect SCENE_ROOT from baseline model's cfg_args
if [ -f "${MODEL_BASELINE}/cfg_args" ]; then
    SCENE_ROOT=$(python3 -c "
import re
try:
    with open('${MODEL_BASELINE}/cfg_args', 'r') as f:
        content = f.read()
        match = re.search(r\"source_path\s*=\s*['\\\"]([^'\\\"]+)['\\\"]\", content)
        if match:
            print(match.group(1))
except Exception as e:
    pass
" 2>/dev/null)
    if [ -z "${SCENE_ROOT}" ] || [ ! -d "${SCENE_ROOT}" ]; then
        SCENE_ROOT="/home/bygpu/data/video_scene"
        echo "Could not auto-detect or path invalid, using: ${SCENE_ROOT}"
    else
        echo "✓ Auto-detected SCENE_ROOT from baseline: ${SCENE_ROOT}"
    fi
else
    SCENE_ROOT="${SCENE_ROOT:-/home/bygpu/data/video_scene}"
    echo "Baseline cfg_args not found, using: ${SCENE_ROOT}"
fi

# Finetune configuration (optional)
FINETUNE_FROM="${FINETUNE_FROM:-}"                          # Path to checkpoint for finetuning (empty = train from scratch)
FINETUNE_LR_SCALE="${FINETUNE_LR_SCALE:-0.1}"              # Learning rate scale for finetuning (default: 0.1)

# Optimized hyperparameters
POSITION_LR_INIT_OPT="${POSITION_LR_INIT_OPT:-0.0002}"      # Slightly higher than baseline
POSITION_LR_FINAL_OPT="${POSITION_LR_FINAL_OPT:-0.0000016}"
LAMBDA_DSSIM_OPT="${LAMBDA_DSSIM_OPT:-0.25}"                # Increased SSIM weight
LAMBDA_EDGE="${LAMBDA_EDGE:-0.1}"                           # Edge loss weight
DENSIFY_FROM_ITER_OPT="${DENSIFY_FROM_ITER_OPT:-500}"
DENSIFY_UNTIL_ITER_OPT="${DENSIFY_UNTIL_ITER_OPT:-25000}"  # Extended densification period
DENSIFY_GRAD_THRESHOLD_OPT="${DENSIFY_GRAD_THRESHOLD_OPT:-0.0002}"

echo "==============================================="
echo "Optimized 3DGS Training & Evaluation"
echo "==============================================="
echo "Scene root      : ${SCENE_ROOT}"
echo "Baseline model  : ${MODEL_BASELINE}"
echo "Optimized model : ${MODEL_OPT}"
echo "Iterations      : ${ITERATIONS}"
echo "Train/Test split: 310 / 25"
if [ -n "${FINETUNE_FROM}" ]; then
  echo "Finetune from   : ${FINETUNE_FROM} (LR scale: ${FINETUNE_LR_SCALE})"
else
  echo "Training mode   : From scratch"
fi
echo ""

# Step 1: Train optimized model
echo "[1/4] Training optimized 3D Gaussian Splatting..."
python train_scene_optimized.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_OPT}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --position_lr_init "${POSITION_LR_INIT_OPT}" \
  --position_lr_final "${POSITION_LR_FINAL_OPT}" \
  --lambda_dssim "${LAMBDA_DSSIM_OPT}" \
  --lambda_edge "${LAMBDA_EDGE}" \
  --densify_from_iter "${DENSIFY_FROM_ITER_OPT}" \
  --densify_until_iter "${DENSIFY_UNTIL_ITER_OPT}" \
  --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD_OPT}" \
  --save_iterations "${ITERATIONS}" \
  $([ -n "${FINETUNE_FROM}" ] && echo "--finetune_from ${FINETUNE_FROM} --finetune_lr_scale ${FINETUNE_LR_SCALE}" || echo "")

echo ""
echo "[2/4] Rendering test set..."
python render.py \
  -m "${MODEL_OPT}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "[3/4] Computing metrics for optimized model..."
python scripts/compute_metrics.py \
  --model_path "${MODEL_OPT}" \
  --set test \
  --iteration "${ITERATIONS}"

echo ""
echo "[4/4] Computing metrics for baseline model (if available)..."
if [ -d "${MODEL_BASELINE}/test/ours_${ITERATIONS}" ]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_BASELINE}" \
    --set test \
    --iteration "${ITERATIONS}"
else
  echo "Baseline test set not found, skipping baseline metrics."
fi

echo ""
echo "==============================================="
echo "Training and evaluation complete!"
echo "==============================================="
echo "Optimized model: ${MODEL_OPT}"
echo "Metrics saved in: ${MODEL_OPT}/test/ours_${ITERATIONS}/"



