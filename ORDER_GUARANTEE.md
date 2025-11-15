# 图片顺序保证说明

## ✅ 顺序保证

本代码库**严格保证**整个流程中图片顺序与原视频一致，具体如下：

### 1. 图片排序方式

所有图片处理都使用**按文件名字符串排序**，这确保了：
- nerfstudio处理视频时，图片通常命名为 `frame_0001.jpg`, `frame_0002.jpg`, ... `frame_0335.jpg`
- 按字符串排序会保持这个时间顺序
- **整个流程中，所有地方都使用相同的排序方式**

### 2. 训练/测试集划分

**重要：测试集是原视频的最后N个帧**

- 如果设置 `TEST_LAST=40`，且总共有335张图片
- **训练集**：前295张（第1张 ~ 第295张）
- **测试集**：后40张（第296张 ~ 第335张）✅ **这是原视频的最后40帧**

代码位置：`scene/dataset_readers.py` 第211-212行
```python
train_cam_infos = cam_infos[:-effective_test_n]  # 前N张
test_cam_infos = cam_infos[-effective_test_n:]   # 最后N张
```

### 3. 顺序保持的关键位置

#### 3.1 数据加载 (`scene/dataset_readers.py`)
- 第199行：按 `image_name` 排序
- 第211-212行：划分训练/测试集（前N训练，后N测试）

#### 3.2 Mask提取 (`extract_segment_everything_masks.py`)
- 第189行：`sorted([f for f in os.listdir(image_dir)])`
- **保证**：mask提取顺序与原视频一致

#### 3.3 Scale计算 (`get_scale.py`)
- 第100行：`sorted(os.listdir(...))`
- **保证**：scale计算顺序与原视频一致

#### 3.4 渲染 (`render.py`)
- 第42行：遍历 `views`，views的顺序来自 `scene.getTrainCameras()` 和 `scene.getTestCameras()`
- 这些camera列表已经按排序后的顺序
- **保证**：渲染输出（RGB、2D mask）的顺序与原视频一致

### 4. 验证工具

运行以下命令验证顺序是否正确：

```bash
# 验证图片顺序和训练/测试集划分
python scripts/verify_temporal_order.py \
  --data_path <数据路径> \
  --test_last_n 40
```

这会检查：
- ✅ 图片是否按文件名排序
- ✅ 测试集是否是最后N张
- ✅ 训练集和测试集是否连续（无间隔）
- ✅ 所有顺序是否一致

### 5. 输出文件顺序

所有输出文件都使用索引编号（`00000.png`, `00001.png`, ...），这些索引对应：
- **训练集**：原视频的第1帧 ~ 第N帧
- **测试集**：原视频的最后M帧

例如，如果总共335张，`TEST_LAST=40`：
- `train/ours_30000/renders/00000.png` ~ `00294.png` = 原视频第1~295帧
- `test/ours_30000/renders/00000.png` ~ `00039.png` = 原视频第296~335帧（最后40帧）

### 6. 重要提醒

⚠️ **不要手动修改图片文件名**，这会破坏顺序！

如果图片文件名不包含数字（如 `image_001.jpg`），字符串排序可能无法保持时间顺序。建议：
1. 使用nerfstudio处理视频（会自动生成有序文件名）
2. 或确保图片文件名包含时间戳或序号

### 7. 代码保证

所有关键位置都有注释说明顺序保证：
- `scene/dataset_readers.py`: 排序和划分逻辑
- `extract_segment_everything_masks.py`: mask提取顺序
- `get_scale.py`: scale计算顺序
- `render.py`: 渲染输出顺序

## 总结

✅ **图片顺序**：按文件名排序，保持原视频顺序  
✅ **测试集**：原视频的最后N个帧  
✅ **所有处理**：mask提取、渲染等都保持相同顺序  
✅ **输出文件**：索引编号对应原视频帧序号  

**你可以放心使用，顺序保证不会改变！**

