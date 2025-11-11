# Reproducible Pipeline (Baseline)

本文件提供“可直接提交/复现”的最小命令集合，所有命令已封装为脚本，助教无需手敲指令即可复现。

## 🚀 一键运行（推荐）

**最简单的方式：运行一键脚本，自动完成所有步骤**

```bash
# 1. 修改脚本中的路径（可选，已有默认值）
# 编辑 scripts/run_baseline_pipeline.sh，修改以下变量：
#   - DATA_RAW: 原始视频/图像路径
#   - OUTPUT_DIR: 输出目录
#   - SAM_CKPT: SAM checkpoint 路径

# 2. 运行一键脚本
chmod +x scripts/run_baseline_pipeline.sh
./scripts/run_baseline_pipeline.sh
```

脚本会自动完成：
1. ✅ 数据处理（ns-process-data + COLMAP）
2. ✅ 基线训练（3D Gaussian Splatting）
3. ✅ SAM 掩码提取和尺度计算
4. ✅ 对比特征训练

完成后，运行 GUI 分割：
```bash
python saga_gui.py --model_path <MODEL_PATH> --data_path <OUTPUT_DIR>
```

---

## 📝 分步运行（如需调试）

如果你想分步运行或调试，可以按以下步骤：

## 0. 环境要求
- 已有可用 GPU + PyTorch + CUDA（能编译/加载本仓库 CUDA 扩展）
- 已安装本仓库依赖（或直接使用 `environment.yml` 创建 `gaussian_splatting` 环境）
- 已安装 nerfstudio + COLMAP（建议单独环境进行数据处理）
- 下载 SAM ViT-H 权重至：
  - `third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth`

## 1. 数据处理（ns-process-data）
脚本：`scripts/run_ns_process_data.sh`

脚本默认已经写入以下示例路径，可按需修改：
- `DATA_RAW=/home/bygpu/Desktop/book.mp4`
- `OUTPUT_DIR=/home/bygpu/data/book_scene`
- `INPUT_TYPE`：`video` 或 `images`
- `NUM_DOWNSCALES`：下采样次数（2~4 建议）

运行：
```bash
chmod +x scripts/run_ns_process_data.sh
./scripts/run_ns_process_data.sh
```

## 2. 基线训练（3D Gaussian Splatting）
脚本：`scripts/train_baseline.sh`

脚本默认已经写入：
- `SCENE_ROOT=/home/bygpu/data/book_scene`

运行：
```bash
chmod +x scripts/train_baseline.sh
./scripts/train_baseline.sh
```
训练脚本会在控制台打印模型输出目录（`MODEL_PATH`），后续步骤需要用到。

## 3. 掩码与尺度（SAM + scale）
脚本：`scripts/extract_masks_and_scales.sh`

修改脚本中的：
- `IMAGE_ROOT`：同 `SCENE_ROOT`
- `MODEL_PATH`：上一步训练得到的模型目录（必填，默认留空需手动填）
- `SAM_CKPT`：SAM 权重路径（默认放在本仓库第三方目录）
- `DOWNSAMPLE`：1/2/4/8（显存紧张用 2 或 4）

运行：
```bash
chmod +x scripts/extract_masks_and_scales.sh
./scripts/extract_masks_and_scales.sh
```

（可选）开放词汇分割需要先抽取 CLIP 特征：
```bash
python get_clip_features.py --image_root $HOME/data/book
```

## 4. 对比学习特征（Affinity / Contrastive）
脚本：`scripts/train_affinity.sh`

修改脚本中的：
- `MODEL_PATH`（默认留空，训练后填写）
- `ITERATIONS`、`NUM_SAMPLED_RAYS`（可保持默认）

运行：
```bash
chmod +x scripts/train_affinity.sh
./scripts/train_affinity.sh
```

## 5. 交互式 3D 分割（GUI）
```bash
python saga_gui.py --model_path <MODEL_PATH>
```
在 GUI 中使用点击模式选择目标，`segment3D` 后用 `save as` 保存，例如：
`./segmentation_res/book_object.pt`

## 6. 渲染（基线 / 分割结果 / 2D masks）
脚本：`scripts/render_all.sh`

修改脚本中的：
- `MODEL_PATH`
- `PRECOMPUTED_MASK`（从 GUI 保存的 pt 文件，如有）

运行：
```bash
chmod +x scripts/render_all.sh
./scripts/render_all.sh
```

## 7. 提交物建议
- `scripts/` 目录（本文件用到的所有脚本）
- `README_RUN.md`（本文件）
- 你的改动源码（如有）
- `results/`（保存渲染图片/视频、日志、`summary.md`，便于快速浏览）
- `REPORT.pdf`

> 体积过大数据/权重可不提交，提供下载链接或说明。



