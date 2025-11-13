#!/usr/bin/env bash
# Fine-tune a baseline 3DGS model with safer hyper-parameters.
# Usage:
#   bash scripts/train_optimized_scene.sh <DATA_ROOT> <BASELINE_MODEL> <OPT_MODEL>
# Example:
#   bash scripts/train_optimized_scene.sh /home/user/data/video_scene ./output/video_scene_20251113_005931 ./output/video_scene_20251113_005931_opt

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/train_optimized_scene.sh <DATA_ROOT> <BASELINE_MODEL> <OPT_MODEL>"
    echo "  <DATA_ROOT>       : ns-process-data output (contains images/ + sparse/)"
    echo "  <BASELINE_MODEL>  : trained baseline model directory (contains point_cloud/iteration_xxx)"
    echo "  <OPT_MODEL>       : destination directory for optimized model checkpoints"
    echo ""
    echo "Optional environment variables:"
    echo "  FINETUNE_STEPS           (default: 4000)"
    echo "  ITERATIONS_OPT           (default: BASE_ITER + FINETUNE_STEPS)"
    echo "  TEST_LAST                (default: 0)"
    echo "  POSITION_LR_INIT_OPT     (default: 0.00008)"
    echo "  POSITION_LR_FINAL_OPT    (default: 0.0000008)"
    echo "  POSITION_LR_DELAY_MULT   (default: 0.0)"
    echo "  LAMBDA_DSSIM_OPT         (default: 0.20)"
    echo "  DENSIFY_FROM_ITER_OPT    (default: 0)"
    echo "  DENSIFY_UNTIL_ITER_OPT   (default: 0)"
    exit 1
fi

DATA_ROOT="$1"
BASELINE_MODEL="$2"
MODEL_OPT="$3"

FINETUNE_STEPS="${FINETUNE_STEPS:-4000}"
TEST_LAST="${TEST_LAST:-0}"
POSITION_LR_INIT_OPT="${POSITION_LR_INIT_OPT:-0.00008}"
POSITION_LR_FINAL_OPT="${POSITION_LR_FINAL_OPT:-0.0000008}"
POSITION_LR_DELAY_MULT="${POSITION_LR_DELAY_MULT:-0.0}"
LAMBDA_DSSIM_OPT="${LAMBDA_DSSIM_OPT:-0.20}"
DENSIFY_FROM_ITER_OPT="${DENSIFY_FROM_ITER_OPT:-0}"
DENSIFY_UNTIL_ITER_OPT="${DENSIFY_UNTIL_ITER_OPT:-0}"

echo "==============================================="
echo "Optimized 3DGS Fine-tuning"
echo "==============================================="
echo "Data root        : ${DATA_ROOT}"
echo "Baseline model   : ${BASELINE_MODEL}"
echo "Optimized output : ${MODEL_OPT}"
echo "Finetune steps   : ${FINETUNE_STEPS}"
echo "Test tail frames : ${TEST_LAST}"
echo ""

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
    exit 1
fi
if [[ ! -d "${BASELINE_MODEL}" ]]; then
    echo "ERROR: BASELINE_MODEL not found: ${BASELINE_MODEL}"
    exit 1
fi
if [[ -e "${MODEL_OPT}" && -n "$(ls -A "${MODEL_OPT}" 2>/dev/null)" ]]; then
    echo "ERROR: OPT_MODEL already exists and is not empty: ${MODEL_OPT}"
    echo "Please choose a new directory or clean it manually."
    exit 1
fi

BASE_ITER=$(python - "${BASELINE_MODEL}" <<'PY'
import os
import sys

model_path = sys.argv[1]
pc_dir = os.path.join(model_path, "point_cloud")
best = -1
if os.path.isdir(pc_dir):
    for name in os.listdir(pc_dir):
        if not name.startswith("iteration_"):
            continue
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        ply = os.path.join(pc_dir, name, "scene_point_cloud.ply")
        if os.path.isfile(ply):
            best = max(best, idx)
if best < 0:
    raise SystemExit("ERROR: could not locate scene_point_cloud.ply in baseline model.")
print(best)
PY
)

if [[ -z "${BASE_ITER}" ]]; then
    echo "ERROR: failed to determine baseline iteration."
    exit 1
fi

if [[ -z "${ITERATIONS_OPT:-}" ]]; then
    ITERATIONS_OPT=$((BASE_ITER + FINETUNE_STEPS))
fi

if (( ITERATIONS_OPT <= BASE_ITER )); then
    echo "ERROR: ITERATIONS_OPT (${ITERATIONS_OPT}) must be greater than baseline iteration (${BASE_ITER})."
    exit 1
fi

echo "Baseline iteration: ${BASE_ITER}"
echo "Target iteration  : ${ITERATIONS_OPT}"
echo ""

mkdir -p "${MODEL_OPT}/point_cloud"

if [[ -f "${BASELINE_MODEL}/cfg_args" ]]; then
    cp "${BASELINE_MODEL}/cfg_args" "${MODEL_OPT}/baseline_cfg_args"
fi
if [[ -f "${BASELINE_MODEL}/cameras.json" ]]; then
    cp "${BASELINE_MODEL}/cameras.json" "${MODEL_OPT}/cameras.json"
fi
if [[ -f "${BASELINE_MODEL}/input.ply" ]]; then
    cp "${BASELINE_MODEL}/input.ply" "${MODEL_OPT}/input.ply"
fi

echo "Copying baseline point cloud iteration ${BASE_ITER}..."
cp -r "${BASELINE_MODEL}/point_cloud/iteration_${BASE_ITER}" "${MODEL_OPT}/point_cloud/"

echo ""
echo "==============================================="
echo "[Step 1/3] Fine-tuning from baseline weights"
echo "==============================================="
TRAIN_CMD=(
  "python" "train_scene.py"
  "-s" "${DATA_ROOT}"
  "--model_path" "${MODEL_OPT}"
  "--iterations" "${ITERATIONS_OPT}"
  "--load_iteration" "${BASE_ITER}"
  "--position_lr_init" "${POSITION_LR_INIT_OPT}"
  "--position_lr_final" "${POSITION_LR_FINAL_OPT}"
  "--position_lr_delay_mult" "${POSITION_LR_DELAY_MULT}"
  "--position_lr_max_steps" "${FINETUNE_STEPS}"
  "--lambda_dssim" "${LAMBDA_DSSIM_OPT}"
  "--densify_from_iter" "${DENSIFY_FROM_ITER_OPT}"
  "--densify_until_iter" "${DENSIFY_UNTIL_ITER_OPT}"
  "--save_iterations" "${ITERATIONS_OPT}"
)
if [[ "${TEST_LAST}" -gt 0 ]]; then
  TRAIN_CMD+=("--eval" "--test_last_n" "${TEST_LAST}")
fi

"${TRAIN_CMD[@]}"

echo ""
echo "==============================================="
echo "[Step 2/3] Rendering optimized model"
echo "==============================================="
python render.py \
  -m "${MODEL_OPT}" \
  -s "${DATA_ROOT}" \
  --target scene \
  --iteration "${ITERATIONS_OPT}"

echo ""
echo "==============================================="
echo "[Step 3/3] Computing metrics"
echo "==============================================="
python scripts/compute_metrics.py \
  --model_path "${MODEL_OPT}" \
  --set train \
  --iteration "${ITERATIONS_OPT}"

if [[ "${TEST_LAST}" -gt 0 ]]; then
  python scripts/compute_metrics.py \
    --model_path "${MODEL_OPT}" \
    --set test \
    --iteration "${ITERATIONS_OPT}"
else
  echo "(Skip) TEST_LAST=0, 不计算单独测试集指标"
fi

echo ""
echo "Done. Optimized outputs in ${MODEL_OPT}"


