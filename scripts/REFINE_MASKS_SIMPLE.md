# 简单的Mask边缘优化（保守方法）

## 说明

这个脚本使用**最简单、最保守**的方法优化mask边缘：
- ✅ **不会报错**：有错误处理，出错时直接复制原文件
- ✅ **保持顺序**：按文件名排序，保持原视频顺序
- ✅ **提高IoU**：使用小kernel，尽量保持原始mask大小
- ✅ **去除光晕**：轻微的开运算和闭运算去除边缘光晕

## 使用方法

### 方法1：使用最简单的脚本（推荐，不检查IoU）

```bash
python scripts/refine_masks_simple_safe.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --opening_kernel 2 \
  --closing_kernel 3
```

### 方法2：使用IoU检查脚本（如果有GT masks，推荐）

```bash
python scripts/refine_masks_with_iou_check.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --model_path ./output/your_model \
  --source_path ./data/your_scene \
  --gt_mask_dir ./data/your_scene/gt_masks/test \
  --iteration 30000
```

### 方法3：使用原来的脚本（已优化默认参数）

```bash
python scripts/refine_segmentation_masks.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --method morphological \
  --opening_kernel 2 \
  --closing_kernel 3 \
  --remove_small 0
```

## 参数说明

- `--opening_kernel 2`：开运算核大小（默认2，很小，不会大幅缩小mask）
- `--closing_kernel 3`：闭运算核大小（默认3，很小，不会大幅扩大mask）
- `--remove_small 0`：不删除小连通域（保持IoU）

## 原理

1. **开运算（opening）**：先腐蚀后膨胀，去除边缘小噪点和光晕
   - kernel=2，只去除很小的噪点，不会大幅缩小mask

2. **闭运算（closing）**：先膨胀后腐蚀，填补小洞
   - kernel=3，只填补很小的洞，不会大幅扩大mask

3. **不删除小连通域**：保持所有mask区域，提高IoU

## 效果

- ✅ 边缘更清晰，光晕减少
- ✅ IoU不会降低（或可能提高）
- ✅ 处理速度快，不会报错
- ✅ 保持原视频顺序

## 注意事项

- 如果某个mask处理失败，会自动复制原文件，不会中断
- 所有mask按文件名排序处理，保持顺序
- 参数很小，效果保守，适合提高IoU

