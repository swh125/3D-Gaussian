#!/usr/bin/env bash
# 完整的数据流程诊断脚本
# 用法: bash scripts/diagnose_data_pipeline.sh <DATA_PATH> [MODEL_PATH]

DATA_PATH="${1:-/home/bygpu/data/book_scene}"
MODEL_PATH="${2:-}"

echo "=========================================="
echo "数据流程完整诊断"
echo "=========================================="
echo ""

# 1. 检查原始数据
echo "=== [1] 原始数据检查 ==="
if [[ -f "${DATA_PATH}" ]] || [[ -d "${DATA_PATH}" ]]; then
    if [[ -f "${DATA_PATH}" ]]; then
        echo "输入是视频文件: ${DATA_PATH}"
        echo "文件大小: $(du -h "${DATA_PATH}" | cut -f1)"
    else
        echo "输入是目录: ${DATA_PATH}"
        IMG_COUNT=$(find "${DATA_PATH}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) 2>/dev/null | wc -l)
        echo "图片数量: ${IMG_COUNT}"
    fi
else
    echo "⚠️  警告: 数据路径不存在: ${DATA_PATH}"
fi
echo ""

# 2. 检查 COLMAP 处理后的数据
echo "=== [2] COLMAP 处理后数据检查 ==="
if [[ -d "${DATA_PATH}/images" ]]; then
    COLMAP_IMG_COUNT=$(ls -1 "${DATA_PATH}/images"/*.jpg "${DATA_PATH}/images"/*.png 2>/dev/null | wc -l)
    echo "COLMAP images 目录图片数: ${COLMAP_IMG_COUNT}"
else
    echo "⚠️  警告: ${DATA_PATH}/images 不存在"
fi

if [[ -d "${DATA_PATH}/colmap/sparse/0" ]] || [[ -d "${DATA_PATH}/sparse/0" ]]; then
    SPARSE_DIR="${DATA_PATH}/colmap/sparse/0"
    if [[ ! -d "${SPARSE_DIR}" ]]; then
        SPARSE_DIR="${DATA_PATH}/sparse/0"
    fi
    
    if [[ -f "${SPARSE_DIR}/cameras.bin" ]]; then
        echo "COLMAP cameras.bin 存在"
        # 尝试读取相机数量（需要 Python）
        python3 -c "
import struct
import sys
try:
    with open('${SPARSE_DIR}/cameras.bin', 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        print(f'COLMAP 相机数量: {num_cameras}')
except Exception as e:
    print(f'无法读取 cameras.bin: {e}')
" 2>/dev/null || echo "需要 Python 来读取 cameras.bin"
    fi
    
    if [[ -f "${SPARSE_DIR}/images.bin" ]]; then
        echo "COLMAP images.bin 存在"
        python3 -c "
import struct
import sys
try:
    with open('${SPARSE_DIR}/images.bin', 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        print(f'COLMAP 图像数量: {num_images}')
except Exception as e:
    print(f'无法读取 images.bin: {e}')
" 2>/dev/null || echo "需要 Python 来读取 images.bin"
    fi
else
    echo "⚠️  警告: COLMAP sparse 目录不存在"
fi
echo ""

# 3. 检查训练时的数据加载
if [[ -n "${MODEL_PATH}" ]] && [[ -d "${MODEL_PATH}" ]]; then
    echo "=== [3] 训练模型数据检查 ==="
    
    # 检查 cfg_args
    if [[ -f "${MODEL_PATH}/cfg_args" ]]; then
        echo "配置参数:"
        cat "${MODEL_PATH}/cfg_args"
        echo ""
    fi
    
    # 检查训练集和测试集渲染结果
    if [[ -d "${MODEL_PATH}/train/ours_30000" ]]; then
        TRAIN_RENDERS=$(ls -1 "${MODEL_PATH}/train/ours_30000/renders"/*.png 2>/dev/null | wc -l)
        TRAIN_GT=$(ls -1 "${MODEL_PATH}/train/ours_30000/gt"/*.png 2>/dev/null | wc -l)
        echo "训练集渲染: ${TRAIN_RENDERS} 张"
        echo "训练集GT: ${TRAIN_GT} 张"
    fi
    
    if [[ -d "${MODEL_PATH}/test/ours_30000" ]]; then
        TEST_RENDERS=$(ls -1 "${MODEL_PATH}/test/ours_30000/renders"/*.png 2>/dev/null | wc -l)
        TEST_GT=$(ls -1 "${MODEL_PATH}/test/ours_30000/gt"/*.png 2>/dev/null | wc -l)
        echo "测试集渲染: ${TEST_RENDERS} 张"
        echo "测试集GT: ${TEST_GT} 张"
    fi
    
    TOTAL=$((TRAIN_RENDERS + TEST_RENDERS))
    if [[ ${TOTAL} -gt 0 ]]; then
        echo ""
        echo "总计: ${TOTAL} 张"
        if [[ ${TOTAL} -lt 10 ]]; then
            echo "⚠️  严重警告: 总图片数过少 (${TOTAL} < 10)，可能是数据预处理问题！"
        fi
    fi
    echo ""
fi

# 4. 诊断建议
echo "=== [4] 诊断建议 ==="
if [[ ${COLMAP_IMG_COUNT:-0} -lt 10 ]]; then
    echo "❌ 问题: COLMAP 处理后的图片数量过少 (${COLMAP_IMG_COUNT})"
    echo "   建议:"
    echo "   1. 检查原始视频/图片是否足够多（建议 200+ 帧）"
    echo "   2. 检查 ns-process-data 是否成功处理了所有帧"
    echo "   3. 检查 COLMAP 是否成功重建了所有相机"
fi

if [[ ${TOTAL:-0} -lt 10 ]] && [[ -n "${MODEL_PATH}" ]]; then
    echo "❌ 问题: 渲染的图片总数过少 (${TOTAL})"
    echo "   建议:"
    echo "   1. 检查 Scene 加载时是否正确读取了所有相机"
    echo "   2. 检查 train/test 划分是否合理"
    echo "   3. 重新运行数据预处理流程"
fi

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="












