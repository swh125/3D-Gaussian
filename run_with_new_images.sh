#!/usr/bin/env bash
# 一键运行脚本 - 新照片路径配置
# 使用方法：修改下面的路径，然后运行: bash run_with_new_images.sh

# ========== 请在这里填写你的照片路径 ==========
PHOTOS_PATH="/home/bygpu/Desktop/video.mp4"  # 你的照片文件夹路径（或视频文件路径）
OUTPUT_DIR="/home/bygpu/data/book_scene"     # 输出目录（ns-process-data 的输出目录，必须是绝对路径！）
INPUT_TYPE="video"                            # "images" 或 "video"
FEATURE_LR="0.0025"                           # 对比特征学习率（优化参数，可选）
TEST_LAST="40"                                # baseline 渲染尾部划为测试集的帧数
# =============================================

echo "=========================================="
echo "一键运行 - 新照片训练"
echo "=========================================="
echo ""

# 检查路径是否存在
if [[ ! -d "${PHOTOS_PATH}" ]] && [[ ! -f "${PHOTOS_PATH}" ]]; then
    echo "错误: 照片路径不存在: ${PHOTOS_PATH}"
    echo "请修改脚本中的 PHOTOS_PATH 变量"
    exit 1
fi

echo "照片路径: ${PHOTOS_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "输入类型: ${INPUT_TYPE}"
echo ""

# 设置环境变量并运行主脚本
export DATA_RAW="${PHOTOS_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export INPUT_TYPE="${INPUT_TYPE}"
export FEATURE_LR="${FEATURE_LR}"
export TEST_LAST="${TEST_LAST}"

echo "正在运行一键脚本..."
echo ""

bash scripts/run_baseline_pipeline.sh

