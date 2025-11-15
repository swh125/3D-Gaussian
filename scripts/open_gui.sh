#!/usr/bin/env bash
# Open GUI to view segmentation results
# Usage:
#   bash scripts/open_gui.sh [MODEL_PATH]
#   bash scripts/open_gui.sh ./output/video_scene_20251113_005931

set -euo pipefail

MODEL_PATH="${1:-./output/video_scene_20251113_005931}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}"
  exit 1
fi

echo "=========================================="
echo "Opening GUI for segmentation"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo ""
echo "Note: GUI requires X11 forwarding or VNC on server"
echo "If you're using SSH, make sure to use: ssh -X user@server"
echo ""

# Open GUI
python saga_gui.py \
  --model_path "${MODEL_PATH}" \
  --feature_iteration 10000 \
  --scene_iteration 30000

