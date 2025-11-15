# Mask边缘优化 + IoU检查

## 说明

这个脚本在优化边缘的同时，**确保2D mask的IoU不降低（或提升）**：

1. 加载3D mask
2. 渲染成2D masks
3. 优化2D masks（简单的形态学操作）
4. 计算优化前后的IoU（如果有GT masks）
5. 如果IoU提升了，说明优化有效
6. 如果IoU降低了，返回原始mask

## 使用方法

### 如果有GT masks（推荐）

```bash
python scripts/refine_masks_with_iou_check.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --model_path ./output/your_model \
  --source_path ./data/your_scene \
  --gt_mask_dir ./data/your_scene/gt_masks/test \
  --iteration 30000
```

### 如果没有GT masks

```bash
# 没有GT时，会跳过IoU检查，直接返回原始mask
python scripts/refine_masks_with_iou_check.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --model_path ./output/your_model \
  --source_path ./data/your_scene \
  --iteration 30000
```

## 参数说明

- `--mask_dir`: 输入3D mask目录
- `--output_dir`: 输出目录
- `--model_path`: 训练好的模型路径
- `--source_path`: 场景数据路径
- `--gt_mask_dir`: GT masks目录（可选，用于IoU检查）
- `--iteration`: 模型迭代次数（默认30000）
- `--opening_kernel`: 开运算核大小（默认2）
- `--closing_kernel`: 闭运算核大小（默认3）

## 工作原理

1. **渲染2D masks**：从3D mask渲染出所有测试集的2D masks
2. **优化2D masks**：对每个2D mask进行简单的形态学优化（去除光晕）
3. **计算IoU**：
   - 原始2D masks vs GT masks → 原始IoU
   - 优化后2D masks vs GT masks → 优化后IoU
4. **决策**：
   - 如果优化后IoU ≥ 原始IoU → 优化有效（但无法直接映射回3D，返回原始3D mask）
   - 如果优化后IoU < 原始IoU → 优化无效，返回原始mask

## 注意事项

- **2D优化无法直接映射回3D**：如果2D优化提升了IoU，说明优化有效，但无法直接反向投影到3D mask
- **建议**：如果2D优化提升了IoU，可以在渲染时对2D masks进行后处理
- **如果没有GT**：无法检查IoU，会直接返回原始mask

## 简单方法（如果不需要IoU检查）

如果不需要IoU检查，可以直接使用：

```bash
python scripts/refine_masks_simple_safe.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined
```

这个脚本更简单，只做基本的边缘优化，不会报错。

