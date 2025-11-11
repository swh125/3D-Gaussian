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

if [[ "${INPUT_TYPE}" == "video" ]]; then
  ns-process-data video \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue
elif [[ "${INPUT_TYPE}" == "images" ]]; then
  ns-process-data images \
    --data "${DATA_RAW}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-downscales "${NUM_DOWNSCALES}" \
    --sfm-tool hloc \
    --feature-type superpoint \
    --matcher-type superglue
else
  echo "ERROR: Unsupported INPUT_TYPE='${INPUT_TYPE}'. Use 'video' or 'images'."
  exit 1
fi

echo "Done. Scene prepared at: ${OUTPUT_DIR}"



