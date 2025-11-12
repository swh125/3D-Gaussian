#!/usr/bin/env bash
# 安装 hloc 工具包（用于 nerfstudio 数据处理）
# 使用方法: bash scripts/install_hloc.sh

echo "=========================================="
echo "Installing hloc toolbox for nerfstudio"
echo "=========================================="
echo ""

# 检查是否在正确的环境中
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: Not in a conda environment. Make sure you're in the correct environment."
fi

echo "Installing pycolmap (required for hloc)..."
pip install pycolmap

echo ""
echo "Installing hloc dependencies..."
pip install torch torchvision
pip install opencv-python matplotlib tqdm h5py scipy

echo ""
echo "Installing superpoint and superglue (for feature extraction)..."
# SuperPoint and SuperGlue are usually included with hloc
# But we can install them explicitly if needed
pip install git+https://github.com/cvg/Hierarchical-Localization.git

echo ""
echo "=========================================="
echo "hloc installation complete!"
echo "=========================================="
echo ""
echo "You can now use hloc with nerfstudio:"
echo "  ns-process-data video --data <video> --output-dir <output> --sfm-tool hloc --feature-type superpoint --matcher-type superglue"
echo ""

