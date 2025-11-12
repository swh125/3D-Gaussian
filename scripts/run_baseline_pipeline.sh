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
TEST_LAST="${TEST_LAST:-40}"                                  # baseline 渲染中划为测试集的尾部帧数
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
    echo "Warning: hloc processing failed, trying fallback with colmap..."
    # Fallback: use colmap instead of hloc
    ns-process-data video \
      --data "${DATA_RAW}" \
      --output-dir "${OUTPUT_DIR}" \
      --num-downscales "${NUM_DOWNSCALES}" \
      --sfm-tool colmap || {
      echo "Warning: nerfstudio colmap failed, trying manual COLMAP processing..."
      # If images were extracted, try manual COLMAP
      if [[ -d "${OUTPUT_DIR}/images" ]]; then
        echo "Images found, running COLMAP manually (without GPU parameter)..."
        # Ensure colmap directory exists
        mkdir -p "${OUTPUT_DIR}/colmap"
        cd "${OUTPUT_DIR}/colmap"
        # Create database if it doesn't exist
        if [[ ! -f database.db ]]; then
          echo "Creating COLMAP database..."
          colmap database_creator --database_path database.db || {
            echo "Error: Failed to create COLMAP database."
            cd - > /dev/null
            exit 1
          }
        fi
        # 1. 特征提取（最关键的步骤）
        colmap feature_extractor \
          --database_path database.db \
          --image_path ../images \
          --ImageReader.single_camera 1 || {
          echo "Error: COLMAP feature extraction failed."
          cd - > /dev/null
          exit 1
        }
        # 2. 特征匹配
        colmap exhaustive_matcher \
          --database_path database.db || {
          echo "Error: COLMAP matching failed."
          cd - > /dev/null
          exit 1
        }
        # 3. 映射重建
        mkdir -p sparse
        colmap mapper \
          --database_path database.db \
          --image_path ../images \
          --output_path sparse || {
          echo "Error: COLMAP mapper failed."
          cd - > /dev/null
          exit 1
        }
        cd - > /dev/null
        echo "✓ Manual COLMAP processing complete"
      else
        echo "Error: Images not found or database not created. Please check your data."
        exit 1
      fi
    }
  }
else
  ns-process-data images \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue || {
    echo "Warning: hloc processing failed, trying fallback with colmap..."
    # Fallback: use colmap instead of hloc
    ns-process-data images \
      --data "${DATA_RAW}" \
      --output-dir "${OUTPUT_DIR}" \
      --num-downscales "${NUM_DOWNSCALES}" \
      --sfm-tool colmap || {
      echo "Warning: nerfstudio colmap failed, trying manual COLMAP processing..."
      # If images exist, try manual COLMAP
      if [[ -d "${OUTPUT_DIR}/images" ]]; then
        echo "Images found, running COLMAP manually (without GPU parameter)..."
        # Ensure colmap directory exists
        mkdir -p "${OUTPUT_DIR}/colmap"
        cd "${OUTPUT_DIR}/colmap"
        # Create database if it doesn't exist
        if [[ ! -f database.db ]]; then
          echo "Creating COLMAP database..."
          colmap database_creator --database_path database.db || {
            echo "Error: Failed to create COLMAP database."
            cd - > /dev/null
            exit 1
          }
        fi
        # 1. 特征提取（最关键的步骤）
        colmap feature_extractor \
          --database_path database.db \
          --image_path ../images \
          --ImageReader.single_camera 1 || {
          echo "Error: COLMAP feature extraction failed."
          cd - > /dev/null
          exit 1
        }
        # 2. 特征匹配
        colmap exhaustive_matcher \
          --database_path database.db || {
          echo "Error: COLMAP matching failed."
          cd - > /dev/null
          exit 1
        }
        # 3. 映射重建
        mkdir -p sparse
        colmap mapper \
          --database_path database.db \
          --image_path ../images \
          --output_path sparse || {
          echo "Error: COLMAP mapper failed."
          cd - > /dev/null
          exit 1
        }
        cd - > /dev/null
        echo "✓ Manual COLMAP processing complete"
      else
        echo "Error: Images not found or database not created. Please check your data."
        exit 1
      fi
    }
  }
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
# Pre-define model output path to keep track of cfg_args
SCENE_NAME=$(basename "${OUTPUT_DIR}")
SCENE_NAME=${SCENE_NAME// /_}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_PATH="./output/${SCENE_NAME}_${TIMESTAMP}"
export MODEL_PATH
echo "Model output will be stored at: ${MODEL_PATH}"

python train_scene.py -s "${OUTPUT_DIR}" --iterations "${ITERATIONS_BASELINE}" --model_path "${MODEL_PATH}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Expected model path not found at ${MODEL_PATH}"
  echo "Please check train_scene.py output."
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

# Step 3B: Render baseline views and split tail frames into test set
echo "[Step 3B] Rendering baseline views for metrics..."
echo "----------------------------------------"
python render.py -m "${MODEL_PATH}" -s "${OUTPUT_DIR}" --target scene --skip_test
python scripts/split_train_test_tail.py \
  --model_path "${MODEL_PATH}" \
  --iteration "${ITERATIONS_BASELINE}" \
  --test_last "${TEST_LAST}"

# Step 4: Train contrastive features
echo "[Step 4/5] Training contrastive features for segmentation..."
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


