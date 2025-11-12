# 安装 hloc 工具包

hloc (Hierarchical Localization) 是 nerfstudio 推荐使用的 SFM 工具，比直接使用 COLMAP 更可靠。

## 快速安装

```bash
# 在 gaussian_splatting 环境中运行
bash scripts/install_hloc.sh
```

## 手动安装

如果自动安装脚本失败，可以手动安装：

```bash
# 1. 安装 pycolmap（hloc 的核心依赖）
pip install pycolmap

# 2. 安装其他依赖
pip install torch torchvision
pip install opencv-python matplotlib tqdm h5py scipy

# 3. 安装 hloc
pip install git+https://github.com/cvg/Hierarchical-Localization.git
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import hloc; print('hloc installed successfully')"
```

如果看到 "hloc installed successfully"，说明安装成功。

## 使用

安装 hloc 后，`run_baseline_pipeline.sh` 脚本会自动使用 hloc 进行数据处理，无需额外配置。

## 常见问题

### 1. 安装失败：找不到 pycolmap
```bash
# 尝试从 conda-forge 安装
conda install -c conda-forge pycolmap
```

### 2. 安装失败：GitHub 连接问题
```bash
# 如果无法访问 GitHub，可以尝试使用镜像或手动下载
# 或者使用代理
export https_proxy=your_proxy
pip install git+https://github.com/cvg/Hierarchical-Localization.git
```

### 3. 版本冲突
如果遇到版本冲突，建议在单独的 conda 环境中安装 nerfstudio 和 hloc：
```bash
conda create -n nsdp python=3.9
conda activate nsdp
pip install nerfstudio
pip install pycolmap
pip install git+https://github.com/cvg/Hierarchical-Localization.git
```

