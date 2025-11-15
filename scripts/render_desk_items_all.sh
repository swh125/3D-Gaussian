#!/usr/bin/env bash
# 使用desk_items.pt渲染所有帧的2D mask和彩色场景，并打包成zip
# Usage: bash scripts/render_desk_items_all.sh

set -euo pipefail

MODEL_PATH="./output/output_scene_20251112_203112"
MASK_FILE="./segmentation_res/desk_items.pt"
ITERATION="30000"
OUTPUT_NAME="desk_items"
OUTPUT_DIR="$HOME/Desktop"

echo "=========================================="
echo "Rendering desk_items for all frames"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Mask: ${MASK_FILE}"
echo "Iteration: ${ITERATION}"
echo ""

# 检查文件是否存在
if [ ! -f "${MASK_FILE}" ]; then
  echo "ERROR: Mask file not found: ${MASK_FILE}"
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}"
  exit 1
fi

# 确保cfg_args存在（render.py需要）
if [ ! -f "${MODEL_PATH}/cfg_args" ]; then
  echo "ERROR: cfg_args not found in ${MODEL_PATH}"
  exit 1
fi

# 复制cfg_args到seg_cfg_args（如果需要）
cp "${MODEL_PATH}/cfg_args" "${MODEL_PATH}/seg_cfg_args" 2>/dev/null || true

echo "[1/4] Rendering 2D masks for train set..."
python render.py -m "${MODEL_PATH}" \
  --target seg \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_test

echo "[2/4] Rendering 2D masks for test set..."
python render.py -m "${MODEL_PATH}" \
  --target seg \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_train

echo "[3/4] Rendering colored scenes for train set..."
python render.py -m "${MODEL_PATH}" \
  --target scene \
  --segment \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_test

echo "[4/4] Rendering colored scenes for test set..."
python render.py -m "${MODEL_PATH}" \
  --target scene \
  --segment \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_train

echo ""
echo "[5/5] Packaging to zip..."

cd "${MODEL_PATH}"

# 创建临时目录
TEMP_DIR="${OUTPUT_NAME}_temp"
mkdir -p "${TEMP_DIR}"

# 复制训练集的2D mask
if [ -d "train/ours_${ITERATION}/mask" ]; then
  mkdir -p "${TEMP_DIR}/train/2d_mask"
  cp -r train/ours_${ITERATION}/mask/* "${TEMP_DIR}/train/2d_mask/" 2>/dev/null || true
  echo "✓ Copied train 2D masks ($(ls -1 train/ours_${ITERATION}/mask/*.png 2>/dev/null | wc -l) files)"
fi

# 复制训练集的彩色渲染
if [ -d "train/ours_${ITERATION}/renders" ]; then
  mkdir -p "${TEMP_DIR}/train/colored_scene"
  cp -r train/ours_${ITERATION}/renders/* "${TEMP_DIR}/train/colored_scene/" 2>/dev/null || true
  echo "✓ Copied train colored scenes ($(ls -1 train/ours_${ITERATION}/renders/*.png 2>/dev/null | wc -l) files)"
fi

# 复制测试集的2D mask
if [ -d "test/ours_${ITERATION}/mask" ]; then
  mkdir -p "${TEMP_DIR}/test/2d_mask"
  cp -r test/ours_${ITERATION}/mask/* "${TEMP_DIR}/test/2d_mask/" 2>/dev/null || true
  echo "✓ Copied test 2D masks ($(ls -1 test/ours_${ITERATION}/mask/*.png 2>/dev/null | wc -l) files)"
fi

# 复制测试集的彩色渲染
if [ -d "test/ours_${ITERATION}/renders" ]; then
  mkdir -p "${TEMP_DIR}/test/colored_scene"
  cp -r test/ours_${ITERATION}/renders/* "${TEMP_DIR}/test/colored_scene/" 2>/dev/null || true
  echo "✓ Copied test colored scenes ($(ls -1 test/ours_${ITERATION}/renders/*.png 2>/dev/null | wc -l) files)"
fi

# 打包成zip
if [ -d "${TEMP_DIR}/train" ] || [ -d "${TEMP_DIR}/test" ]; then
  cd "${TEMP_DIR}"
  zip -r "${OUTPUT_DIR}/${OUTPUT_NAME}.zip" . && echo "✓ ${OUTPUT_NAME}.zip created on Desktop"
  cd ..
  ls -lh "${OUTPUT_DIR}/${OUTPUT_NAME}.zip"
  rm -rf "${TEMP_DIR}"
else
  echo "ERROR: No files to package"
fi

# 清理原始渲染结果（可选，注释掉如果不想删除）
# rm -rf seg/train/ours_${ITERATION} seg/test/ours_${ITERATION}
# rm -rf scene/train/ours_${ITERATION} scene/test/ours_${ITERATION}

cd - > /dev/null

echo ""
echo "=========================================="
echo "Completed!"
echo "Zip file: ${OUTPUT_DIR}/${OUTPUT_NAME}.zip"
echo "=========================================="

