#!/usr/bin/env bash
# 3DGS Training Optimization Ablation Study
# Tests: Baseline, Edge Loss only, SSIM only, Both

set -euo pipefail

# Configuration
MODEL_BASELINE="${MODEL_BASELINE:-./output/video_scene_20251113_005931}"

# Auto-detect SCENE_ROOT from baseline model's cfg_args
if [ -f "${MODEL_BASELINE}/cfg_args" ]; then
    # Try to extract source_path from cfg_args (Namespace format)
    SCENE_ROOT=$(python3 -c "
import re
try:
    with open('${MODEL_BASELINE}/cfg_args', 'r') as f:
        content = f.read()
        # Match source_path='...' (Namespace format)
        # Try multiple patterns
        match = re.search(r\"source_path\s*=\s*['\\\"]([^'\\\"]+)['\\\"]\", content)
        if match:
            print(match.group(1))
        else:
            # Try without quotes
            match = re.search(r\"source_path\s*=\s*([^,\\s)]+)\", content)
            if match:
                print(match.group(1).strip(\"'\\\"\"))
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
    SCENE_ROOT="/home/bygpu/data/video_scene"
    echo "Baseline cfg_args not found, using: ${SCENE_ROOT}"
fi

ITERATIONS="${ITERATIONS:-30000}"
TEST_LAST="${TEST_LAST:-40}"  # Last 40 frames for test

# Baseline hyperparameters (from train_scene.py defaults)
LAMBDA_DSSIM_BASELINE="${LAMBDA_DSSIM_BASELINE:-0.2}"  # Baseline default
LAMBDA_EDGE_BASELINE="${LAMBDA_EDGE_BASELINE:-0.0}"    # No edge loss

# Optimized hyperparameters
LAMBDA_DSSIM_OPT="${LAMBDA_DSSIM_OPT:-0.25}"            # Increased SSIM weight
LAMBDA_EDGE="${LAMBDA_EDGE:-0.1}"                       # Edge loss weight

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==============================================="
echo "3DGS Training Optimization Ablation Study"
echo "==============================================="
echo "Scene root: ${SCENE_ROOT}"
echo "Baseline model: ${MODEL_BASELINE}"
echo "Iterations: ${ITERATIONS}"
echo "Test set: Last ${TEST_LAST} frames"
echo ""

# ============================================
# Experiment 1: Edge Loss Only
# ============================================
echo "==============================================="
echo "[Experiment 1/3] Edge Loss Only"
echo "==============================================="
MODEL_EDGE_ONLY="./output/video_scene_ablation_edge_only_${TIMESTAMP}"
echo "Training with edge loss only (λ_edge=0.1, λ_dssim=baseline=0.2)..."
python train_scene_optimized.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_EDGE_ONLY}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --lambda_dssim "${LAMBDA_DSSIM_BASELINE}" \
  --lambda_edge "${LAMBDA_EDGE}" \
  --save_iterations "${ITERATIONS}"

echo ""
echo "Rendering test set..."
python render.py \
  -m "${MODEL_EDGE_ONLY}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "Computing metrics..."
if [ -f "scripts/compute_metrics.py" ]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_EDGE_ONLY}" \
    --set test \
    --iteration "${ITERATIONS}"
fi

echo "✓ Edge Loss Only complete: ${MODEL_EDGE_ONLY}"
echo ""

# ============================================
# Experiment 2: SSIM Only
# ============================================
echo "==============================================="
echo "[Experiment 2/3] SSIM Adjustment Only"
echo "==============================================="
MODEL_SSIM_ONLY="./output/video_scene_ablation_ssim_only_${TIMESTAMP}"
echo "Training with SSIM adjustment only (λ_dssim=0.25, λ_edge=0)..."
python train_scene_optimized.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_SSIM_ONLY}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --lambda_dssim "${LAMBDA_DSSIM_OPT}" \
  --lambda_edge 0.0 \
  --save_iterations "${ITERATIONS}"

echo ""
echo "Rendering test set..."
python render.py \
  -m "${MODEL_SSIM_ONLY}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "Computing metrics..."
if [ -f "scripts/compute_metrics.py" ]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_SSIM_ONLY}" \
    --set test \
    --iteration "${ITERATIONS}"
fi

echo "✓ SSIM Only complete: ${MODEL_SSIM_ONLY}"
echo ""

# ============================================
# Experiment 3: Both (Full Optimization)
# ============================================
echo "==============================================="
echo "[Experiment 3/3] Both (Full Optimization)"
echo "==============================================="
MODEL_BOTH="./output/video_scene_ablation_both_${TIMESTAMP}"
echo "Training with both optimizations (λ_dssim=0.25, λ_edge=0.1)..."
python train_scene_optimized.py \
  -s "${SCENE_ROOT}" \
  --model_path "${MODEL_BOTH}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --test_last_n "${TEST_LAST}" \
  --lambda_dssim "${LAMBDA_DSSIM_OPT}" \
  --lambda_edge "${LAMBDA_EDGE}" \
  --save_iterations "${ITERATIONS}"

echo ""
echo "Rendering test set..."
python render.py \
  -m "${MODEL_BOTH}" \
  -s "${SCENE_ROOT}" \
  --iteration "${ITERATIONS}" \
  --skip_train \
  --eval \
  --test_last_n "${TEST_LAST}"

echo ""
echo "Computing metrics..."
if [ -f "scripts/compute_metrics.py" ]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_BOTH}" \
    --set test \
    --iteration "${ITERATIONS}"
fi

echo "✓ Both complete: ${MODEL_BOTH}"
echo ""

# ============================================
# Baseline Metrics (if available)
# ============================================
echo "==============================================="
echo "[Baseline] Computing baseline metrics..."
echo "==============================================="
if [ -d "${MODEL_BASELINE}/test/ours_${ITERATIONS}" ]; then
  if [ -f "scripts/compute_metrics.py" ]; then
    python scripts/compute_metrics.py \
      --model_path "${MODEL_BASELINE}" \
      --set test \
      --iteration "${ITERATIONS}"
  fi
  echo "✓ Baseline metrics computed"
else
  echo "⚠️  Baseline test set not found, skipping baseline metrics."
  echo "   You may need to render baseline test set first:"
  echo "   python render.py -m ${MODEL_BASELINE} -s ${SCENE_ROOT} --iteration ${ITERATIONS} --skip_train --eval --test_last_n ${TEST_LAST}"
fi
echo ""

# ============================================
# Summary
# ============================================
echo "==============================================="
echo "Ablation Study Complete!"
echo "==============================================="
echo ""
echo "Results saved in:"
echo "  - Baseline: ${MODEL_BASELINE}"
echo "  - Edge Loss Only: ${MODEL_EDGE_ONLY}"
echo "  - SSIM Only: ${MODEL_SSIM_ONLY}"
echo "  - Both: ${MODEL_BOTH}"
echo ""
echo "Metrics location:"
echo "  - <model_path>/test/ours_${ITERATIONS}/metrics.txt"
echo ""
echo "To compare PSNR/SSIM/LPIPS, check the metrics files above."

