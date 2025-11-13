#!/usr/bin/env bash
# Clean full pipeline: train 3DGS on all frames, regenerate SAM masks with post-processing,
# compute mask scales, and train contrastive features.
# Usage:
#   bash scripts/run_full_pipeline_clean_masks.sh \
#       --scene-root /home/bygpu/data/video_scene \
#       --model-path ./output/video_scene_fulltrain_$(date +%Y%m%d_%H%M%S)
# All arguments are optional; see defaults below.

set -euo pipefail

# ---------- Default configuration (override via flags) ----------
SCENE_ROOT="/home/bygpu/data/video_scene"
MODEL_PATH=""
ITERATIONS=30000
SAM_ARCH="vit_h"
SAM_CKPT="third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"
SAM_EXTRA_CKPTS="third_party/segment-anything/sam_ckpt/sam_vit_l_0b3195.pth"
SAM_DOWNSAMPLE=1
SAM_MIN_PIXELS=300
SAM_MIN_COMPONENT_PIXELS=200
SAM_CLOSING_KERNEL=5
SAM_MIN_IOU=0.90
SAM_MIN_STABILITY=0.90
SAM_MIN_RATIO=0.0
SAM_MAX_MASKS=0
FEATURE_ITERS=10000
FEATURE_NUM_RAYS=500
FEATURE_SAMPLE_RATE=0.0
FEATURE_LR=0.0025
TEST_LAST=0

# ---------- Parse CLI flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-root)
      SCENE_ROOT="$2"; shift 2 ;;
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    --sam-arch)
      SAM_ARCH="$2"; shift 2 ;;
    --sam-ckpt)
      SAM_CKPT="$2"; shift 2 ;;
    --sam-extra-ckpts)
      SAM_EXTRA_CKPTS="$2"; shift 2 ;;
    --sam-downsample)
      SAM_DOWNSAMPLE="$2"; shift 2 ;;
    --sam-min-pixels)
      SAM_MIN_PIXELS="$2"; shift 2 ;;
    --sam-min-component-pixels)
      SAM_MIN_COMPONENT_PIXELS="$2"; shift 2 ;;
    --sam-closing-kernel)
      SAM_CLOSING_KERNEL="$2"; shift 2 ;;
    --sam-min-iou)
      SAM_MIN_IOU="$2"; shift 2 ;;
    --sam-min-stability)
      SAM_MIN_STABILITY="$2"; shift 2 ;;
    --sam-min-ratio)
      SAM_MIN_RATIO="$2"; shift 2 ;;
    --sam-max-masks)
      SAM_MAX_MASKS="$2"; shift 2 ;;
    --feature-iters)
      FEATURE_ITERS="$2"; shift 2 ;;
    --feature-num-rays)
      FEATURE_NUM_RAYS="$2"; shift 2 ;;
    --feature-sample-rate)
      FEATURE_SAMPLE_RATE="$2"; shift 2 ;;
    --feature-lr)
      FEATURE_LR="$2"; shift 2 ;;
    --test-last)
      TEST_LAST="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "ERROR: scene root not found: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  SCENE_NAME=$(basename "${SCENE_ROOT}")
  SCENE_NAME=${SCENE_NAME// /_}
  MODEL_PATH="./output/${SCENE_NAME}_fulltrain_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${MODEL_PATH}"

echo "==============================================="
echo "Full pipeline with clean masks"
echo "==============================================="
echo "Scene root      : ${SCENE_ROOT}"
echo "Model output    : ${MODEL_PATH}"
echo "Iterations      : ${ITERATIONS}"
echo "SAM checkpoint  : ${SAM_CKPT}"
echo "Extra checkpoints: ${SAM_EXTRA_CKPTS}" 
if [[ "${TEST_LAST}" -gt 0 ]]; then
  echo "Test tail frames : ${TEST_LAST}"
else
  echo "Test tail frames : disabled"
fi
echo ""

# ---------- Step 1: Train 3DGS on all frames ----------
TRAIN_CMD=(
  python train_scene.py
  -s "${SCENE_ROOT}"
  --model_path "${MODEL_PATH}"
  --iterations "${ITERATIONS}"
  --save_iterations "${ITERATIONS}"
)

if [[ "${TEST_LAST}" -gt 0 ]]; then
  TRAIN_CMD+=(--eval --test_last_n "${TEST_LAST}")
fi

echo "[1/6] Training 3D Gaussian Splatting..."
"${TRAIN_CMD[@]}"
echo "✓ Training complete"

# ---------- Step 2: Render outputs ----------
echo "[2/6] Rendering trained views..."
python render.py \
  -m "${MODEL_PATH}" \
  -s "${SCENE_ROOT}" \
  --target scene \
  --iteration "${ITERATIONS}"

echo "[3/6] Computing PSNR/SSIM/LPIPS (train set)"
python scripts/compute_metrics.py \
  --model_path "${MODEL_PATH}" \
  --set train \
  --iteration "${ITERATIONS}"

if [[ "${TEST_LAST}" -gt 0 ]]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_PATH}" \
    --set test \
    --iteration "${ITERATIONS}"
fi

# ---------- Step 3: Regenerate SAM masks with filtering ----------
echo "[4/6] Extracting clean SAM masks..."
python extract_segment_everything_masks.py \
  --image_root "${SCENE_ROOT}" \
  --sam_arch "${SAM_ARCH}" \
  --sam_checkpoint_path "${SAM_CKPT}" \
  --extra_checkpoints "${SAM_EXTRA_CKPTS}" \
  --downsample "${SAM_DOWNSAMPLE}" \
  --min_mask_pixels "${SAM_MIN_PIXELS}" \
  --min_component_pixels "${SAM_MIN_COMPONENT_PIXELS}" \
  --closing_kernel "${SAM_CLOSING_KERNEL}" \
  --min_predicted_iou "${SAM_MIN_IOU}" \
  --min_stability_score "${SAM_MIN_STABILITY}" \
  --min_mask_ratio "${SAM_MIN_RATIO}" \
  --max_masks_per_image "${SAM_MAX_MASKS}"

echo "[5/6] Computing mask scales..."
python get_scale.py \
  --image_root "${SCENE_ROOT}" \
  --model_path "${MODEL_PATH}"

# ---------- Step 4: Train contrastive features ----------
echo "[6/6] Training contrastive features..."
FEATURE_CMD=(
  python train_contrastive_feature.py
  -m "${MODEL_PATH}"
  -s "${SCENE_ROOT}"
  --iteration "${ITERATIONS}"
  --iterations "${FEATURE_ITERS}"
  --feature_lr "${FEATURE_LR}"
)
if [[ "${FEATURE_NUM_RAYS}" != "0" && "${FEATURE_NUM_RAYS}" != "0.0" ]]; then
  FEATURE_CMD+=(--num_sampled_rays "${FEATURE_NUM_RAYS}")
fi
if [[ "${FEATURE_SAMPLE_RATE}" != "0" && "${FEATURE_SAMPLE_RATE}" != "0.0" ]]; then
  FEATURE_CMD+=(--ray_sample_rate "${FEATURE_SAMPLE_RATE}")
fi

"${FEATURE_CMD[@]}"

echo "==============================================="
echo "Full pipeline finished!"
echo "Model path : ${MODEL_PATH}"
echo "==============================================="
