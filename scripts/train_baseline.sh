#!/usr/bin/env bash
set -euo pipefail

# Purpose: Train baseline 3D Gaussian Splatting on a processed scene.
# Usage:
#   ./scripts/train_baseline.sh
#   (If permission denied, run: chmod +x scripts/train_baseline.sh)
#
# Edit SCENE_ROOT to point to the directory produced by ns-process-data.

# --------- USER CONFIG (已填默认值，可按需修改) ----------
SCENE_ROOT="${SCENE_ROOT:-/home/bygpu/data/book_scene}"   # ns-process-data 输出目录
# -------------------------------------------------------

echo "[3DGS] Training baseline on scene: ${SCENE_ROOT}"
python train_scene.py -s "${SCENE_ROOT}"
echo "Training launched. Check output folder printed by the script for model path."



