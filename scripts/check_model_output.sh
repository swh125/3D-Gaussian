#!/usr/bin/env bash
# 检查模型输出目录的状态（训练集/测试集数量、配置等）
# 用法: bash scripts/check_model_output.sh <MODEL_PATH>

MODEL_PATH="${1:-./output/output_scene_20251112_225845}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "错误: 模型目录不存在: ${MODEL_PATH}"
    exit 1
fi

echo "=========================================="
echo "检查模型输出目录: ${MODEL_PATH}"
echo "=========================================="
echo ""

cd "${MODEL_PATH}" || exit 1

# 检查目录整体结构
echo "=== 目录结构 ==="
ls -la
echo ""

# 检查训练集和测试集的图片数量
echo "=== 训练集渲染图片数量 ==="
TRAIN_RENDERS=$(ls -1 train/ours_30000/renders/*.png 2>/dev/null | wc -l)
echo "${TRAIN_RENDERS}"

echo "=== 训练集GT图片数量 ==="
TRAIN_GT=$(ls -1 train/ours_30000/gt/*.png 2>/dev/null | wc -l)
echo "${TRAIN_GT}"

echo "=== 测试集渲染图片数量 ==="
TEST_RENDERS=$(ls -1 test/ours_30000/renders/*.png 2>/dev/null | wc -l)
echo "${TEST_RENDERS}"

echo "=== 测试集GT图片数量 ==="
TEST_GT=$(ls -1 test/ours_30000/gt/*.png 2>/dev/null | wc -l)
echo "${TEST_GT}"
echo ""

# 检查数据划分是否合理
TOTAL=$((TRAIN_RENDERS + TEST_RENDERS))
if [[ ${TOTAL} -gt 0 ]]; then
    TRAIN_PCT=$((TRAIN_RENDERS * 100 / TOTAL))
    TEST_PCT=$((TEST_RENDERS * 100 / TOTAL))
    echo "=== 数据划分统计 ==="
    echo "总图片数: ${TOTAL}"
    echo "训练集: ${TRAIN_RENDERS} (${TRAIN_PCT}%)"
    echo "测试集: ${TEST_RENDERS} (${TEST_PCT}%)"
    echo ""
    
    if [[ ${TEST_RENDERS} -lt 4 ]]; then
        echo "⚠️  警告: 测试集图片数量过少 (${TEST_RENDERS} < 4)，指标可能不准确"
    fi
    if [[ ${TRAIN_RENDERS} -lt 4 ]]; then
        echo "⚠️  警告: 训练集图片数量过少 (${TRAIN_RENDERS} < 4)"
    fi
    echo ""
fi

# 查看配置文件
echo "=== 配置参数 ==="
if [[ -f cfg_args ]]; then
    cat cfg_args
else
    echo "cfg_args 文件不存在"
fi
echo ""

# 查看点云文件
echo "=== 点云文件 ==="
if ls -1 *.ply 2>/dev/null | head -1 > /dev/null; then
    ls -lh *.ply
else
    echo "没有找到 .ply 文件"
fi
echo ""

# 检查是否有渲染结果
echo "=== 渲染结果检查 ==="
if [[ -d "train/ours_30000" ]]; then
    echo "✓ 训练集渲染目录存在"
else
    echo "✗ 训练集渲染目录不存在"
fi

if [[ -d "test/ours_30000" ]]; then
    echo "✓ 测试集渲染目录存在"
else
    echo "✗ 测试集渲染目录不存在"
fi
echo ""

cd - > /dev/null

