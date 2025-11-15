# IoU计算和GT Mask说明

## IoU是怎么算的？

### IoU公式

```
IoU = Intersection / Union
    = (预测mask和GT mask重叠的部分) / (预测mask和GT mask合并的部分)
```

### 代码实现

```python
def compute_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """计算IoU"""
    pred_bool = pred_mask.bool()  # 预测mask（0或1）
    gt_bool = gt_mask.bool()      # GT mask（0或1）
    
    # 交集：两个mask都是1的地方
    intersection = torch.logical_and(pred_bool, gt_bool).sum().item()
    
    # 并集：两个mask至少有一个是1的地方
    union = torch.logical_or(pred_bool, gt_bool).sum().item()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union
```

### 例子

假设有一个 4x4 的mask：

**预测mask:**
```
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0
```

**GT mask:**
```
0 0 0 0
0 1 1 1
0 1 1 0
0 0 0 0
```

**交集（intersection）**：两个都是1的地方
```
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0
```
交集 = 4个像素

**并集（union）**：至少一个是1的地方
```
0 0 0 0
0 1 1 1
0 1 1 0
0 0 0 0
```
并集 = 5个像素

**IoU = 4 / 5 = 0.8**

### 在代码中的使用

1. **渲染2D masks**：从3D mask渲染出所有测试集的2D masks
2. **加载GT masks**：从GT目录加载对应的GT mask图片
3. **逐帧计算IoU**：对每一帧计算预测mask和GT mask的IoU
4. **平均IoU**：所有帧的IoU取平均值

```python
# 对每一帧
for idx in range(len(masks_2d)):
    pred_mask = masks_2d[idx]  # 从3D mask渲染的2D mask
    gt_mask = load_mask_image(gt_files[idx])  # GT mask图片
    
    iou = compute_mask_iou(pred_mask, gt_mask)
    iou_values.append(iou)

# 平均IoU
mean_iou = np.mean(iou_values)
```

## GT Mask是怎么来的？

### 方法1：使用SAM自动生成（推荐）

使用 `scripts/generate_gt_masks.py` 脚本：

```bash
python scripts/generate_gt_masks.py \
  --image_dir /path/to/data/images \
  --output_dir /path/to/data/gt_masks/test \
  --sam_checkpoint_path /path/to/sam_vit_h_4b8939.pth \
  --test_indices 295-334  # 测试集的帧索引（最后40帧）
```

**工作原理**：
1. 加载SAM模型
2. 对测试集的每张图片，使用SAM自动生成所有物体的mask
3. 将所有mask合并（union）
4. 保存为PNG图片（白色=物体，黑色=背景）

**优点**：
- 自动生成，速度快
- SAM质量高，通常很准确

**缺点**：
- 可能包含一些不想要的物体
- 需要手动筛选或后处理

### 方法2：手动标注（最准确）

1. 打开测试集的图片
2. 使用图像编辑工具（GIMP、Photoshop）或标注工具（LabelMe）手动标注
3. 保存为二值mask（白色=物体，黑色=背景）

**优点**：
- 最准确，完全符合你的需求

**缺点**：
- 耗时，需要一张一张标注

### 方法3：使用GUI手动选择（中等质量）

1. 启动SAGA GUI
2. 对测试集的每个视角，手动选择物体并保存mask
3. 将这些mask作为GT

### GT Mask的格式

- **文件格式**：PNG图片
- **颜色**：
  - 白色（255）= 物体
  - 黑色（0）= 背景
- **命名**：与测试集图片对应（按顺序）
- **位置**：`./data/your_scene/gt_masks/test/`

### 目录结构示例

```
./data/your_scene/
  ├── images/              # 原始图片
  │   ├── frame_0001.jpg
  │   ├── frame_0002.jpg
  │   └── ...
  └── gt_masks/
      └── test/            # 测试集的GT masks
          ├── 00000.png    # 对应测试集第1帧
          ├── 00001.png    # 对应测试集第2帧
          └── ...
```

## 完整流程

### 1. 生成GT Masks

```bash
# 使用SAM生成（假设测试集是最后40帧，索引295-334）
python scripts/generate_gt_masks.py \
  --image_dir ./data/your_scene/images \
  --output_dir ./data/your_scene/gt_masks/test \
  --sam_checkpoint_path third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth \
  --test_indices 295-334
```

### 2. 渲染预测Masks

```bash
# 从3D mask渲染2D masks（测试集）
python render.py \
  -m ./output/your_model \
  -s ./data/your_scene \
  --target seg \
  --precomputed_mask ./segmentation_res/your_object.pt \
  --iteration 30000 \
  --skip_train  # 只渲染测试集
```

### 3. 计算IoU

```bash
# 计算IoU
python scripts/compute_segmentation_iou.py \
  --pred_mask_dir ./output/your_model/test/ours_30000/mask \
  --gt_mask_dir ./data/your_scene/gt_masks/test
```

### 4. 优化Mask并检查IoU

```bash
# 优化mask并检查IoU是否提升
python scripts/refine_masks_with_iou_check.py \
  --mask_dir ./segmentation_res \
  --output_dir ./segmentation_res_refined \
  --model_path ./output/your_model \
  --source_path ./data/your_scene \
  --gt_mask_dir ./data/your_scene/gt_masks/test \
  --iteration 30000
```

## 总结

- **IoU计算**：`IoU = 交集 / 并集`
- **GT Mask来源**：
  1. SAM自动生成（推荐，快速）
  2. 手动标注（最准确，耗时）
  3. GUI手动选择（中等质量）
- **GT Mask格式**：PNG图片，白色=物体，黑色=背景
- **GT Mask位置**：`./data/your_scene/gt_masks/test/`




