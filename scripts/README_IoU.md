# 计算分割 IoU 的完整流程

## 什么是 GT Mask？

**GT (Ground Truth) Mask** 是真实的分割结果，用于评估模型预测的准确性。

## ⚠️ 重要：IoU 应该在测试集上计算

**IoU 评估应该在测试集（test set）上进行**，原因：
- 测试集用于评估模型的泛化性能
- 训练集上的 IoU 可能过拟合，不能真实反映模型性能
- 作业报告应该报告测试集的 IoU 指标

对于你的作业，需要：
1. **为测试集的图像**生成/标注 GT mask（真实的分割掩码）
2. 用训练好的模型渲染出**测试集**的预测 mask
3. 计算**测试集**上预测 mask 和 GT mask 之间的 IoU

## 步骤 1: 生成 GT Mask

### 方法 A: 使用 SAM 自动生成（推荐）

使用 SAM 为测试集图像生成高质量的 mask 作为 GT：

```bash
cd ~/Desktop/3D-Gaussian

# 假设测试集是最后 15 帧（索引 320-334，总共 335 帧）
python scripts/generate_gt_masks.py \
  --image_dir /home/bygpu/data/video_scene/images \
  --output_dir /home/bygpu/data/video_scene/gt_masks/test \
  --sam_checkpoint_path /path/to/sam_vit_h.pth \
  --test_indices 320-334 \
  --model_type vit_h
```

这会为测试集的每张图像生成一个 GT mask 文件（`*_gt_mask.png`）。

### 方法 B: 手动标注（更准确但耗时）

1. 打开测试集的图像
2. 使用图像编辑工具（如 GIMP、Photoshop）或标注工具（如 LabelMe）手动标注每个物体
3. 保存为二值 mask（白色=物体，黑色=背景）

### 方法 C: 使用 GUI 手动选择（中等质量）

1. 启动 SAGA GUI
2. 对测试集的每个视角，手动选择物体并保存 mask
3. 将这些 mask 作为 GT

## 步骤 2: 渲染预测 Mask

### 方法 A: 从 3D Mask 渲染（如果已有 GUI 保存的 mask）

```bash
MODEL=./output/video_scene_optimized_xxx
MASK=./segmentation_res/book_baseline.pt

# 渲染测试集的 mask
python render.py \
  -m "$MODEL" \
  -s /home/bygpu/data/video_scene \
  --target seg \
  --precomputed_mask "$MASK" \
  --iteration 30000 \
  --skip_train \
  --eval \
  --test_last_n 15
```

这会在 `$MODEL/test/ours_30000/mask/` 生成预测的 mask。

### 方法 B: 从模型直接渲染（如果模型已经训练了分割）

如果模型已经训练了分割功能，直接渲染即可。

## 步骤 3: 计算 IoU（测试集）

### 方法 A: 从目录计算（推荐）

**注意：这里使用的是 `test/ours_30000/mask`，即测试集的预测 mask**

```bash
python scripts/compute_segmentation_iou.py \
  --pred_mask_dir ./output/video_scene_optimized_xxx/test/ours_30000/mask \
  --gt_mask_dir /home/bygpu/data/video_scene/gt_masks/test \
  --verbose
```

### 如果想在训练集上也计算（可选，用于调试）

```bash
# 渲染训练集的 mask
python render.py \
  -m "$MODEL" \
  --target seg \
  --precomputed_mask ./segmentation_res/book_baseline.pt \
  --iteration 30000 \
  --skip_test \
  --eval \
  --test_last_n 15

# 计算训练集 IoU（需要先为训练集生成 GT mask）
python scripts/compute_segmentation_iou.py \
  --pred_mask_dir ./output/video_scene_optimized_xxx/train/ours_30000/mask \
  --gt_mask_dir /home/bygpu/data/video_scene/gt_masks/train \
  --verbose
```

**但作业报告里应该主要报告测试集的 IoU。**

### 方法 B: 从 3D Mask 计算

```bash
python scripts/compute_segmentation_iou.py \
  --pred_mask_3d ./segmentation_res/book_baseline.pt \
  --gt_mask_dir /home/bygpu/data/video_scene/gt_masks/test \
  --model_path ./output/video_scene_optimized_xxx \
  --source_path /home/bygpu/data/video_scene \
  --iteration 30000
```

## 输出示例

```
=== IoU Summary ===
GT mask dir  : /home/bygpu/data/video_scene/gt_masks/test
Pred mask dir : ./output/video_scene_optimized_xxx/test/ours_30000/mask
Mean IoU     : 0.8234 ± 0.0456
```

## 注意事项

1. **GT Mask 的质量很重要**：如果 GT mask 不准确，IoU 评估就没有意义
2. **Mask 尺寸要匹配**：预测 mask 和 GT mask 的尺寸要一致（脚本会自动处理）
3. **命名要对应**：GT mask 的文件名应该和预测 mask 的文件名对应（如 `00000.png` 对应 `00000_gt_mask.png`）

## 完整工作流示例

```bash
# 1. 生成 GT masks（测试集：最后 15 帧）
python scripts/generate_gt_masks.py \
  --image_dir /home/bygpu/data/video_scene/images \
  --output_dir /home/bygpu/data/video_scene/gt_masks/test \
  --sam_checkpoint_path /path/to/sam_vit_h.pth \
  --test_indices 320-334

# 2. 训练优化模型（320 train / 15 test）
bash scripts/train_and_eval_optimized.sh

# 3. 渲染预测 mask（假设用 GUI 保存的 book_baseline.pt）
MODEL=./output/video_scene_optimized_xxx
python render.py \
  -m "$MODEL" \
  --target seg \
  --precomputed_mask ./segmentation_res/book_baseline.pt \
  --iteration 30000 \
  --skip_train \
  --eval \
  --test_last_n 15

# 4. 计算 IoU
python scripts/compute_segmentation_iou.py \
  --pred_mask_dir "$MODEL/test/ours_30000/mask" \
  --gt_mask_dir /home/bygpu/data/video_scene/gt_masks/test \
  --verbose
```

