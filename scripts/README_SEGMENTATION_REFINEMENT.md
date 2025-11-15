# 分割Mask边缘优化指南

## 问题描述

分割mask的边缘经常出现：
- **光晕（halo）**：边缘模糊、不清晰
- **锯齿状边缘**：不平滑
- **小噪点**：边缘附近的小碎片
- **不连续**：边缘有断裂

## 优化方法

### 1. 形态学细化（推荐，最简单有效）

**原理**：
- **开运算**：先腐蚀后膨胀，去除边缘小噪点和光晕
- **闭运算**：先膨胀后腐蚀，填补小洞
- **连通域分析**：保留最大连通域，移除小碎片

**使用方法**：
```bash
python scripts/refine_segmentation_masks.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --method morphological \
  --opening_kernel 3 \
  --closing_kernel 5 \
  --remove_small 100
```

**参数说明**：
- `opening_kernel`: 开运算核大小（3-5，越大去除光晕越强）
- `closing_kernel`: 闭运算核大小（5-7，填补小洞）
- `remove_small`: 移除小于此像素数的连通域

### 2. 边缘感知细化（需要原图）

**原理**：利用原图的边缘信息，在边缘附近保持mask，远离边缘的地方进行形态学操作

**使用方法**：
```bash
python scripts/refine_segmentation_masks.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --image_dir ./data/train_dataset/custom_dataset/images \
  --method edge_aware
```

### 3. 多尺度细化

**原理**：在不同尺度下细化mask，然后投票融合

**使用方法**：
```bash
python scripts/refine_segmentation_masks.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --method multi_scale
```

### 4. 其他方法

- **高斯模糊**：`--method gaussian_blur`
- **形态学梯度**：`--method morphological_gradient`
- **距离变换**：`--method distance_transform`
- **双边滤波**：`--method bilateral_filter`

## 在GUI中使用优化后的Mask

1. **生成优化后的mask**：
   ```bash
   python scripts/refine_segmentation_masks.py \
     --mask_dir ./segmentation_res \
     --output_dir ./segmentation_res_refined \
     --method morphological
   ```

2. **在训练或渲染时使用**：
   ```bash
   # 使用优化后的mask进行训练
   python train_scene_optimized.py \
     -s ./data/train_dataset/custom_dataset \
     --precomputed_mask ./segmentation_res_refined/your_object.pt
   ```

## 参数调优建议

### 针对光晕问题
- 增大 `opening_kernel`（3 → 5）
- 减小 `closing_kernel`（5 → 3）

### 针对边缘不清晰
- 使用 `edge_aware` 方法（需要原图）
- 或使用 `multi_scale` 方法

### 针对小噪点
- 增大 `remove_small`（100 → 500）

## 完整工作流示例

```bash
# 1. 在GUI中分割物体，保存到 ./segmentation_res/object.pt

# 2. 优化分割mask
python scripts/refine_segmentation_masks.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --method morphological \
  --opening_kernel 3 \
  --closing_kernel 5

# 3. 使用优化后的mask训练
python train_scene_optimized.py \
  -s ./data/train_dataset/custom_dataset \
  --precomputed_mask ./segmentation_res_refined/object.pt
```

## 效果对比

优化前：
- 边缘有光晕
- 有小噪点
- 边缘不清晰

优化后：
- 边缘清晰
- 无噪点
- 边缘平滑

## 注意事项

1. **不要过度优化**：过大的kernel可能会丢失细节
2. **保留原mask备份**：优化前先备份
3. **根据物体大小调整参数**：小物体用小的kernel，大物体用大的kernel
4. **多次尝试**：不同物体可能需要不同的参数


