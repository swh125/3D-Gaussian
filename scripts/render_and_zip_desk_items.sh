#!/usr/bin/env bash
# Render and zip desk_items segmentation results
# Usage:
#   bash scripts/render_and_zip_desk_items.sh

set -euo pipefail

MODEL_PATH="./output/77e56970-f"
OBJ="desk_items"
ITERATION=30000
MASK_FILE="./segmentation_res/${OBJ}.pt"

echo "=========================================="
echo "Rendering and Zipping ${OBJ}"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Mask file: ${MASK_FILE}"
echo "Iteration: ${ITERATION}"
echo ""

# Check files exist
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${MASK_FILE}" ]]; then
  echo "ERROR: Mask file not found: ${MASK_FILE}"
  exit 1
fi

# Render 2D masks
echo "[1/2] Rendering 2D masks for ${OBJ}..."
python render.py \
  -m "${MODEL_PATH}" \
  --target seg \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_train

# Render colored scene
echo "[2/2] Rendering colored scene for ${OBJ}..."
python render.py \
  -m "${MODEL_PATH}" \
  --target scene \
  --segment \
  --precomputed_mask "${MASK_FILE}" \
  --iteration "${ITERATION}" \
  --skip_train

# Create zip files
echo "[3/4] Creating zip files..."

# Zip 2D masks (output is in test/ours_30000/mask/)
MASK_DIR="${MODEL_PATH}/test/ours_${ITERATION}/mask"
if [[ -d "${MASK_DIR}" ]]; then
  cd "${MODEL_PATH}/test/ours_${ITERATION}"
  zip -r "${OBJ}_2d_mask.zip" "mask" 2>/dev/null || true
  mv "${OBJ}_2d_mask.zip" "${MODEL_PATH}/" 2>/dev/null || true
  cd - > /dev/null
  echo "  ✓ Created: ${MODEL_PATH}/${OBJ}_2d_mask.zip"
else
  echo "  ✗ WARNING: Mask directory not found: ${MASK_DIR}"
fi

# Zip colored scene renders (output is in test/ours_30000/renders/)
RENDER_DIR="${MODEL_PATH}/test/ours_${ITERATION}/renders"
if [[ -d "${RENDER_DIR}" ]]; then
  cd "${MODEL_PATH}/test/ours_${ITERATION}"
  zip -r "${OBJ}_colored_scene.zip" "renders" 2>/dev/null || true
  mv "${OBJ}_colored_scene.zip" "${MODEL_PATH}/" 2>/dev/null || true
  cd - > /dev/null
  echo "  ✓ Created: ${MODEL_PATH}/${OBJ}_colored_scene.zip"
else
  echo "  ✗ WARNING: Render directory not found: ${RENDER_DIR}"
fi

# Move to Desktop
echo "[4/4] Moving to Desktop..."
if [[ -f "${MODEL_PATH}/${OBJ}_2d_mask.zip" ]]; then
  mv "${MODEL_PATH}/${OBJ}_2d_mask.zip" ~/Desktop/ 2>/dev/null || true
  echo "  ✓ Moved: ~/Desktop/${OBJ}_2d_mask.zip"
fi

if [[ -f "${MODEL_PATH}/${OBJ}_colored_scene.zip" ]]; then
  mv "${MODEL_PATH}/${OBJ}_colored_scene.zip" ~/Desktop/ 2>/dev/null || true
  echo "  ✓ Moved: ~/Desktop/${OBJ}_colored_scene.zip"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Files on Desktop:"
echo "  - ${OBJ}_2d_mask.zip"
echo "  - ${OBJ}_colored_scene.zip"
echo ""




