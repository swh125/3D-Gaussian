# Mask优化说明

## 什么是"原始mask"？

**原始mask** = 你从GUI保存的3D mask文件（比如 `book.pt`）
- 这是你在GUI中手动分割物体后保存的
- 保存在 `./segmentation_res/` 目录下
- 是一个3D的二进制mask（标记哪些3D高斯点属于这个物体）

## 优化过程

1. **加载原始mask**：从 `./segmentation_res/book.pt` 加载
2. **渲染成2D masks**：从3D mask渲染出所有视角的2D masks
3. **优化2D masks**：对每个2D mask进行形态学操作（去除光晕）
4. **检查IoU**：
   - 原始2D masks vs GT masks → 原始IoU
   - 优化后2D masks vs GT masks → 优化后IoU
5. **决策**：
   - 如果优化后IoU ≥ 原始IoU → 优化有效 ✓
   - 如果优化后IoU < 原始IoU → 优化无效 ✗

## "返回原始mask"是什么意思？

**"返回原始mask"** = 保存你原来的mask文件，不做任何修改

- **如果IoU提升了**：
  - 说明2D优化方法有效 ✓
  - 但2D优化无法直接映射回3D mask
  - 所以还是保存原始3D mask
  - **优化可以在渲染2D mask时应用**（后处理）

- **如果IoU降低了**：
  - 说明优化方法无效 ✗
  - 不应该使用优化
  - 保存原始mask（不应用优化）

## 实际效果

这个脚本的主要作用是：
1. **验证优化方法是否有效**（通过IoU检查）
2. **如果有效**：你可以在渲染时对2D masks进行后处理优化
3. **如果无效**：不使用优化，保持原始mask

## 更简单的方法

如果你不需要检查IoU，可以直接使用简单的优化脚本：

```bash
python scripts/refine_masks_simple_safe.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined
```

这个脚本会：
- 直接优化mask（不做IoU检查）
- 使用保守的参数（尽量保持IoU）
- 不会报错

## 总结

- **原始mask** = 你从GUI保存的mask文件
- **"返回原始mask"** = 保存原来的mask，不做修改
- **优化检查** = 验证优化方法是否有效（通过IoU）
- **如果有效** = 可以在渲染时应用优化
- **如果无效** = 不使用优化




