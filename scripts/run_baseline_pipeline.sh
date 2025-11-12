#!/usr/bin/env bash
set -euo pipefail

# Purpose: Complete baseline pipeline - from video to trained model ready for segmentation
# Usage:
#   chmod +x scripts/run_baseline_pipeline.sh
#   ./scripts/run_baseline_pipeline.sh
#
# This script runs all steps in sequence:
#   1. Process video/images with nerfstudio (COLMAP)
#   2. Train baseline 3D Gaussian Splatting
#   3. Extract SAM masks and compute scales
#   4. Train contrastive features for segmentation
#
# After this completes, you can run GUI segmentation:
#   python saga_gui.py --model_path <MODEL_PATH> --data_path <DATA_PATH>

# --------- USER CONFIG (修改这些路径) ----------
DATA_RAW="${DATA_RAW:-/home/bygpu/Desktop/book.mp4}"          # 原始视频或图像目录
OUTPUT_DIR="${OUTPUT_DIR:-/home/bygpu/data/book_scene}"       # ns-process-data 输出目录
INPUT_TYPE="${INPUT_TYPE:-video}"                             # video 或 images
NUM_DOWNSCALES="${NUM_DOWNSCALES:-2}"                         # 显存紧张可调大
SAM_CKPT="${SAM_CKPT:-third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth}"  # SAM checkpoint 路径
DOWNSAMPLE="${DOWNSAMPLE:-2}"                                 # SAM mask downsample (1/2/4/8)
ITERATIONS_BASELINE="${ITERATIONS_BASELINE:-30000}"           # Baseline 训练迭代次数
ITERATIONS_AFFINITY="${ITERATIONS_AFFINITY:-10000}"            # 对比特征训练迭代次数
NUM_SAMPLED_RAYS="${NUM_SAMPLED_RAYS:-1000}"                  # 对比特征训练采样光线数
FEATURE_LR="${FEATURE_LR:-0.0025}"                            # 对比特征学习率（优化参数）
# -------------------------------------------------------

echo "=========================================="
echo "Baseline Pipeline - Complete Workflow"
echo "=========================================="
echo ""

# Step 1: Process data with nerfstudio
echo "[Step 1/4] Processing data with nerfstudio..."
echo "----------------------------------------"
if [[ "${INPUT_TYPE}" == "video" ]]; then
  ns-process-data video \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue || {
    echo "Warning: hloc processing failed, trying fallback..."
    # Fallback: try manual COLMAP mapper
    if [[ -d "${OUTPUT_DIR}/colmap" ]] && [[ ! -d "${OUTPUT_DIR}/colmap/sparse/0" ]]; then
      echo "Running COLMAP mapper manually..."
      cd "${OUTPUT_DIR}/colmap"
      colmap mapper \
        --database_path database.db \
        --image_path ../images \
        --output_path sparse \
        --Mapper.ba_global_function_tolerance=1e-6 || {
        echo "Error: COLMAP mapper failed. Please check your data."
        exit 1
      }
      cd - > /dev/null
    fi
  }
else
  ns-process-data images \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue
fi

# Create sparse symlink if needed
if [[ -d "${OUTPUT_DIR}/colmap/sparse" ]] && [[ ! -d "${OUTPUT_DIR}/sparse" ]]; then
  echo "Creating sparse symlink..."
  ln -s colmap/sparse "${OUTPUT_DIR}/sparse"
fi

echo "✓ Data processing complete"
echo ""

# Step 2: Train baseline 3DGS
echo "[Step 2/4] Training baseline 3D Gaussian Splatting..."
echo "----------------------------------------"
python train_scene.py -s "${OUTPUT_DIR}" --iterations "${ITERATIONS_BASELINE}"

# Extract model path from output (usually in ./output/ directory)
MODEL_PATH=$(find ./output -type d -name "*" -newer "${OUTPUT_DIR}" 2>/dev/null | head -1)
if [[ -z "${MODEL_PATH}" ]]; then
  # Try to find the most recent output directory
  MODEL_PATH=$(ls -td ./output/*/ 2>/dev/null | head -1)
  MODEL_PATH=${MODEL_PATH%/}
fi

if [[ -z "${MODEL_PATH}" ]] || [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Could not find trained model path. Please check train_scene.py output."
  echo "Please manually set MODEL_PATH and run the remaining steps."
  exit 1
fi

echo "✓ Baseline training complete. Model path: ${MODEL_PATH}"
echo ""

# Step 3: Extract SAM masks and compute scales
echo "[Step 3/4] Extracting SAM masks and computing scales..."
echo "----------------------------------------"
if [[ ! -f "${SAM_CKPT}" ]]; then
  echo "ERROR: SAM checkpoint not found at ${SAM_CKPT}"
  echo "Please download SAM checkpoint first:"
  echo "  mkdir -p third_party/segment-anything/sam_ckpt"
  echo "  wget -O ${SAM_CKPT} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
  exit 1
fi

python extract_segment_everything_masks.py \
  --image_root "${OUTPUT_DIR}" \
  --sam_checkpoint_path "${SAM_CKPT}" \
  --downsample "${DOWNSAMPLE}"

python get_scale.py \
  --image_root "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}"

echo "✓ SAM masks and scales extracted"
echo ""

# Step 4: Train contrastive features
echo "[Step 4/4] Training contrastive features for segmentation..."
echo "----------------------------------------"
python train_contrastive_feature.py \
  -m "${MODEL_PATH}" \
  --iterations "${ITERATIONS_AFFINITY}" \
  --feature_lr "${FEATURE_LR}" \
  --num_sampled_rays "${NUM_SAMPLED_RAYS}"

echo "✓ Contrastive features trained"
echo ""

# Summary
echo "=========================================="
echo "Baseline Pipeline Complete!"
echo "=========================================="
echo ""
echo "Model path: ${MODEL_PATH}"
echo "Data path: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run GUI for interactive segmentation:"
echo "     python saga_gui.py --model_path ${MODEL_PATH} --data_path ${OUTPUT_DIR}"
echo ""
echo "  2. Or render results:"
echo "     python render.py -m ${MODEL_PATH} -s ${OUTPUT_DIR} --target scene"
echo ""


