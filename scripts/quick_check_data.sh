#!/usr/bin/env bash
# 快速检查处理后的数据：训练集和测试集数量
# 用法: bash scripts/quick_check_data.sh <OUTPUT_DIR>
# 例如: bash scripts/quick_check_data.sh /home/bygpu/data/video_scene

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "用法: bash scripts/quick_check_data.sh <OUTPUT_DIR>"
    echo "例如: bash scripts/quick_check_data.sh /home/bygpu/data/video_scene"
    exit 1
fi

OUTPUT_DIR="$1"

echo "=========================================="
echo "快速检查数据统计"
echo "=========================================="
echo ""

# 检查 COLMAP 输出
if [[ -d "${OUTPUT_DIR}/sparse/0" ]]; then
    echo "✓ 找到 COLMAP sparse 输出"
    
    # 检查 images.bin 或 images.txt
    if [[ -f "${OUTPUT_DIR}/sparse/0/images.bin" ]]; then
        echo "  使用二进制格式 (images.bin)"
        # 使用 Python 读取二进制文件统计数量
        IMAGE_COUNT=$(python3 -c "
import sys
sys.path.insert(0, '.')
from scene.colmap_loader import read_extrinsics_binary
try:
    images = read_extrinsics_binary('${OUTPUT_DIR}/sparse/0/images.bin')
    print(len(images))
except:
    print('0')
" 2>/dev/null || echo "0")
    elif [[ -f "${OUTPUT_DIR}/sparse/0/images.txt" ]]; then
        echo "  使用文本格式 (images.txt)"
        # 统计 images.txt 中的图像数量（跳过注释行）
        IMAGE_COUNT=$(grep -v "^#" "${OUTPUT_DIR}/sparse/0/images.txt" | grep -v "^$" | wc -l || echo "0")
    else
        IMAGE_COUNT="0"
    fi
    
    echo "  COLMAP 重建的图像数: ${IMAGE_COUNT}"
else
    echo "❌ 未找到 COLMAP sparse 输出"
    IMAGE_COUNT="0"
fi

echo ""

# 检查 images 目录
if [[ -d "${OUTPUT_DIR}/images" ]]; then
    IMAGE_DIR_COUNT=$(ls -1 "${OUTPUT_DIR}/images"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l || echo "0")
    echo "✓ images 目录中的图片数: ${IMAGE_DIR_COUNT}"
else
    echo "⚠️  未找到 images 目录"
    IMAGE_DIR_COUNT="0"
fi

echo ""

# 使用 Python 脚本检查实际加载的训练/测试集
echo "=== 检查训练/测试集划分 ==="
python3 scripts/check_loaded_cameras.py "${OUTPUT_DIR}" 2>/dev/null || {
    echo "⚠️  无法加载 Scene 数据，可能路径不正确或数据不完整"
}

echo ""
echo "=========================================="
echo "检查完成"
echo "=========================================="











