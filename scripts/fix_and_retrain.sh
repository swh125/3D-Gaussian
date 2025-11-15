#!/usr/bin/env bash
# 修复路径问题并重新训练
# 用法: bash scripts/fix_and_retrain.sh <正确的数据路径> [模型输出路径]

CORRECT_DATA_PATH="${1:-/home/bygpu/data/book_scene}"
MODEL_OUTPUT="${2:-}"

if [[ ! -d "${CORRECT_DATA_PATH}" ]]; then
    echo "错误: 数据路径不存在: ${CORRECT_DATA_PATH}"
    exit 1
fi

echo "=========================================="
echo "修复路径并重新训练"
echo "=========================================="
echo "数据路径: ${CORRECT_DATA_PATH}"
echo ""

# 1. 检查数据
echo "[1] 检查数据..."
python scripts/check_loaded_cameras.py "${CORRECT_DATA_PATH}"
echo ""

# 2. 检查是否有足够的相机
CAMERA_COUNT=$(python scripts/check_loaded_cameras.py "${CORRECT_DATA_PATH}" 2>&1 | grep "总计:" | awk '{print $2}')
if [[ -z "${CAMERA_COUNT}" ]] || [[ "${CAMERA_COUNT}" -lt 10 ]]; then
    echo "❌ 错误: 相机数量不足 (${CAMERA_COUNT})"
    echo "请检查:"
    echo "  1. 数据路径是否正确: ${CORRECT_DATA_PATH}"
    echo "  2. sparse/0 目录是否存在且完整"
    echo "  3. 是否需要重新运行 ns-process-data"
    exit 1
fi

# 3. 重新训练
echo "[2] 开始重新训练..."
if [[ -z "${MODEL_OUTPUT}" ]]; then
    SCENE_NAME=$(basename "${CORRECT_DATA_PATH}")
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MODEL_OUTPUT="./output/${SCENE_NAME}_${TIMESTAMP}"
fi

echo "模型输出路径: ${MODEL_OUTPUT}"
echo ""

python train_scene.py -s "${CORRECT_DATA_PATH}" --iterations 30000 --model_path "${MODEL_OUTPUT}"

if [[ ! -d "${MODEL_OUTPUT}" ]]; then
    echo "❌ 错误: 训练失败，模型目录不存在"
    exit 1
fi

echo ""
echo "✓ 训练完成！模型路径: ${MODEL_OUTPUT}"
echo ""
echo "下一步:"
echo "  1. 提取 SAM 掩码:"
echo "     python extract_segment_everything_masks.py --image_root ${CORRECT_DATA_PATH} --sam_checkpoint_path third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth --downsample 2"
echo ""
echo "  2. 计算 mask scales:"
echo "     python get_scale.py --image_root ${CORRECT_DATA_PATH} --model_path ${MODEL_OUTPUT}"
echo ""
echo "  3. 训练对比特征:"
echo "     python train_contrastive_feature.py -m ${MODEL_OUTPUT} --iterations 10000 --num_sampled_rays 1000"
echo ""












