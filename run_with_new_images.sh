#!/usr/bin/env bash
# 一键运行脚本 - 新照片路径配置
# 使用方法：修改下面的路径，然后运行: bash run_with_new_images.sh

# ========== 请在这里填写你的照片路径 ==========
PHOTOS_PATH="/home/bygpu/Desktop/video.mp4"  # 你的照片文件夹路径（或视频文件路径）
INPUT_TYPE="video"                            # "images" 或 "video"
FEATURE_LR="0.0025"                           # 对比特征学习率（优化参数，可选）
TEST_LAST="40"                               # 测试集帧数（后N个作为测试集）
                                             # 例如：总共335张，TEST_LAST=40 → 前295张训练，后40张测试
                                             # 图片会按文件名排序（通常对应拍摄顺序）
AUTO_OPEN_GUI="1"                             # 完成后自动打开GUI (1=是, 0=否)

# OUTPUT_DIR 会根据视频文件名自动生成（如果未设置环境变量）
# 例如：如果视频是 video.mp4，输出目录会是 /home/bygpu/data/video_scene
# 你也可以手动设置：export OUTPUT_DIR="/home/bygpu/data/your_custom_scene"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    VIDEO_NAME=$(basename "${PHOTOS_PATH}" .mp4)
    OUTPUT_DIR="/home/bygpu/data/${VIDEO_NAME}_scene"
fi
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
echo "测试集数量: ${TEST_LAST} (后${TEST_LAST}张作为测试集)"
if [[ "${TEST_LAST}" -gt 0 ]]; then
    echo "  ✅ 保证: 图片按文件名排序，保持原视频顺序"
    echo "  ✅ 保证: 最后 ${TEST_LAST} 张作为测试集（原视频的最后 ${TEST_LAST} 帧）"
    echo "  ✅ 保证: 所有处理过程（mask提取、渲染等）都保持这个顺序"
    echo "  ✅ 保证: 2D mask和彩色渲染的输出顺序与原视频一致"
fi
echo "自动打开GUI: $([ "${AUTO_OPEN_GUI}" == "1" ] && echo "是" || echo "否")"
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

# 检查pipeline是否成功完成
if [[ $? -ne 0 ]]; then
    echo "错误: Pipeline执行失败，无法打开GUI"
    exit 1
fi

# 尝试从输出中提取MODEL_PATH，或使用默认逻辑
# run_baseline_pipeline.sh会在最后输出MODEL_PATH
# 我们尝试从最新的output目录中找到模型
SCENE_NAME=$(basename "${OUTPUT_DIR}")
SCENE_NAME=${SCENE_NAME// /_}

# 查找最新的模型目录
LATEST_MODEL=$(ls -td ./output/${SCENE_NAME}_* 2>/dev/null | head -1)

if [[ -z "${LATEST_MODEL}" ]] || [[ ! -d "${LATEST_MODEL}" ]]; then
    echo "警告: 无法自动找到模型路径"
    echo "请手动运行GUI:"
    echo "  python saga_gui.py --model_path <MODEL_PATH> --data_path ${OUTPUT_DIR}"
    exit 0
fi

MODEL_PATH="${LATEST_MODEL}"
echo ""
echo "=========================================="
echo "Pipeline完成！"
echo "=========================================="
echo "模型路径: ${MODEL_PATH}"
echo "数据路径: ${OUTPUT_DIR}"
echo ""

# 如果设置了自动打开GUI
if [[ "${AUTO_OPEN_GUI}" == "1" ]]; then
    echo "正在打开GUI进行交互式分割..."
    echo ""
    echo "GUI使用说明:"
    echo "  - 左键拖拽: 旋转视角"
    echo "  - 中键拖拽: 平移"
    echo "  - 右键点击: 输入点提示（需要先选择分割模式）"
    echo "  - 选择目标后，点击 'segment3D' 进行3D分割"
    echo "  - 使用 'save as' 保存分割结果"
    echo ""
    echo "按 Ctrl+C 退出GUI"
    echo ""
    
    python saga_gui.py --model_path "${MODEL_PATH}" --data_path "${OUTPUT_DIR}"
else
    echo "要打开GUI进行交互式分割，请运行:"
    echo "  python saga_gui.py --model_path ${MODEL_PATH} --data_path ${OUTPUT_DIR}"
    echo ""
fi

