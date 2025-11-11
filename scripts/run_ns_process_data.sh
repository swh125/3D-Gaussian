#!/usr/bin/env bash
set -euo pipefail

# Purpose: Process raw video/images into a COLMAP-ready scene with nerfstudio.
# Usage:
#   chmod +x scripts/run_ns_process_data.sh
#   ./scripts/run_ns_process_data.sh
#
# Edit the variables below to match your server paths before running.

# --------- USER CONFIG (已填默认值，可按需修改) ----------
DATA_RAW="${DATA_RAW:-/home/bygpu/Desktop/book.mp4}"    # 原始视频或图像目录
OUTPUT_DIR="${OUTPUT_DIR:-/home/bygpu/data/book_scene}" # ns-process-data 输出目录
INPUT_TYPE="${INPUT_TYPE:-video}"                       # video 或 images
NUM_DOWNSCALES="${NUM_DOWNSCALES:-2}"                   # 显存紧张可调大
# -------------------------------------------------------

echo "[ns-process-data] input=${DATA_RAW}  out=${OUTPUT_DIR}  type=${INPUT_TYPE}  downscales=${NUM_DOWNSCALES}"

# Try with hloc first (avoids COLMAP SIFT GPU parameter issue)
if [[ "${INPUT_TYPE}" == "video" ]]; then
  ns-process-data video \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue || {
    echo "Warning: hloc processing failed, trying fallback..."
    exit 1
  }
elif [[ "${INPUT_TYPE}" == "images" ]]; then
  ns-process-data images \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue || {
    echo "Warning: hloc processing failed, trying fallback..."
    exit 1
  }
else
  echo "ERROR: Unsupported INPUT_TYPE='${INPUT_TYPE}'. Use 'video' or 'images'."
  exit 1
fi

# Check if mapper failed due to SuiteSparse issue and retry with DENSE solver
# This handles the case where hloc/colmap mapper fails because SuiteSparse is not available
if [[ -f "${OUTPUT_DIR}/colmap/database.db" ]] && [[ ! -d "${OUTPUT_DIR}/colmap/sparse/0" ]]; then
  echo "COLMAP sparse reconstruction missing. Retrying mapper with DENSE_SCHUR solver (avoids SuiteSparse requirement)..."
  mkdir -p "${OUTPUT_DIR}/colmap/sparse"
  colmap mapper \
    --database_path "${OUTPUT_DIR}/colmap/database.db" \
    --image_path "${OUTPUT_DIR}/images" \
    --output_path "${OUTPUT_DIR}/colmap/sparse" \
    --Mapper.ba_global_function_tolerance=1e-6 \
    --Mapper.ba_global_sparse_linear_algebra_library_type=DENSE_SCHUR || {
    echo "ERROR: COLMAP mapper failed even with DENSE_SCHUR solver."
    exit 1
  }
  echo "Successfully completed COLMAP mapping with DENSE_SCHUR solver."
fi

echo "Done. Scene prepared at: ${OUTPUT_DIR}"



