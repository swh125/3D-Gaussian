#!/usr/bin/env bash
# Render and zip segmentation results for optimized masks
# Usage:
#   bash scripts/render_and_zip_optimized.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./output/output_scene_20251112_203112}"
SEGMENTATION_RES_DIR="${SEGMENTATION_RES_DIR:-./segmentation_res}"
ITERATION="${ITERATION:-30000}"

# List of objects to process
OBJECTS=(
  "book_optimized"
  "glasses_optimized"
  "umbrella_optimized"
  "pencil_case_optimized"
  "juice_optimized"
)

echo "=========================================="
echo "Rendering and Zipping Optimized Masks"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Segmentation dir: ${SEGMENTATION_RES_DIR}"
echo "Iteration: ${ITERATION}"
echo ""

# Check model path exists
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}"
  exit 1
fi

# Process each object
for obj in "${OBJECTS[@]}"; do
  MASK_FILE="${SEGMENTATION_RES_DIR}/${obj}.pt"
  
  if [[ ! -f "${MASK_FILE}" ]]; then
    echo "WARNING: Mask file not found: ${MASK_FILE}, skipping..."
    continue
  fi
  
  echo "----------------------------------------"
  echo "Processing: ${obj}"
  echo "----------------------------------------"
  
  # Render 2D masks
  echo "[1/2] Rendering 2D masks for ${obj}..."
  python render.py \
    -m "${MODEL_PATH}" \
    --target seg \
    --precomputed_mask "${MASK_FILE}" \
    --iteration "${ITERATION}" \
    --skip_train
  
  # Render colored scene with segmentation
  echo "[2/2] Rendering colored scene with segmentation for ${obj}..."
  python render.py \
    -m "${MODEL_PATH}" \
    --target scene \
    --segment \
    --precomputed_mask "${MASK_FILE}" \
    --iteration "${ITERATION}" \
    --skip_train
  
  # Create zip files
  echo "[3/3] Creating zip files for ${obj}..."
  
  # Zip 2D masks
  SEG_DIR="${MODEL_PATH}/seg/ours_${ITERATION}"
  if [[ -d "${SEG_DIR}" ]]; then
    cd "${MODEL_PATH}"
    zip -r "${obj}_2d_mask.zip" "seg/ours_${ITERATION}" 2>/dev/null || true
    cd - > /dev/null
    echo "  ✓ Created: ${MODEL_PATH}/${obj}_2d_mask.zip"
  fi
  
  # Zip colored scene renders
  SCENE_DIR="${MODEL_PATH}/scene/ours_${ITERATION}"
  if [[ -d "${SCENE_DIR}" ]]; then
    cd "${MODEL_PATH}"
    zip -r "${obj}_colored_scene.zip" "scene/ours_${ITERATION}" 2>/dev/null || true
    cd - > /dev/null
    echo "  ✓ Created: ${MODEL_PATH}/${obj}_colored_scene.zip"
  fi
  
  echo "✓ Completed: ${obj}"
  echo ""
done

echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Zip files are in: ${MODEL_PATH}"
echo ""

