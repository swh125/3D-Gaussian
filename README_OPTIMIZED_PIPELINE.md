# 优化版本 Pipeline 使用说明

## 快速开始

```bash
# 1. 编辑脚本，设置你的视频路径
nano run_optimized_pipeline.sh
# 修改 PHOTOS_PATH 为你的视频路径

# 2. 运行优化pipeline
bash run_optimized_pipeline.sh
```

## 优化内容

### 相比 Baseline 的改进

1. **边缘优化（自动应用）**
   - 2D mask渲染时自动应用形态学操作
   - 开运算（kernel=2）：去除边缘光晕
   - 闭运算（kernel=3）：平滑边缘，去除锯齿
   - **无需额外步骤，渲染时自动优化**

2. **顺序保证**
   - 所有处理过程按文件名排序
   - 保持原视频的时序顺序
   - 2D mask和渲染输出顺序与原视频一致

3. **训练/测试集划分**
   - 支持指定最后N帧作为测试集
   - 例如：TEST_LAST=40 → 前295张训练，后40张测试

## 配置说明

在 `run_optimized_pipeline.sh` 中修改以下变量：

```bash
PHOTOS_PATH="/home/bygpu/Desktop/video.mp4"  # 你的视频路径
INPUT_TYPE="video"                            # "images" 或 "video"
FEATURE_LR="0.0025"                           # 对比特征学习率
TEST_LAST="40"                               # 测试集帧数（后N个）
AUTO_OPEN_GUI="1"                             # 完成后自动打开GUI
```

## Pipeline 步骤

1. **数据处理**：使用 nerfstudio 处理视频/图片（COLMAP）
2. **Baseline训练**：训练3D Gaussian Splatting模型
3. **SAM掩码提取**：提取Segment Anything掩码
4. **对比特征训练**：训练用于分割的对比特征
5. **自动打开GUI**：进行交互式3D分割

## 渲染优化后的Mask

优化后的mask会在渲染时自动应用形态学操作：

```bash
# 渲染2D mask（自动应用边缘优化）
python render.py -m <MODEL_PATH> -s <DATA_PATH> --target seg
```

## 与 Baseline 版本的区别

| 特性 | Baseline | 优化版本 |
|------|----------|----------|
| 边缘优化 | ❌ 无 | ✅ 自动应用 |
| 去除光晕 | ❌ 无 | ✅ 开运算（kernel=2）|
| 平滑边缘 | ❌ 无 | ✅ 闭运算（kernel=3）|
| 顺序保证 | ✅ 是 | ✅ 是 |
| 训练/测试划分 | ✅ 是 | ✅ 是 |

## 注意事项

1. **路径配置**：确保 `PHOTOS_PATH` 指向正确的视频文件
2. **GPU显存**：如果显存不足，可以调整 `NUM_DOWNSCALES` 和 `DOWNSAMPLE`
3. **形态学操作**：已集成到 `render.py`，无需额外配置
4. **IoU影响**：使用保守参数（kernel=2,3），对IoU影响很小

## 常见问题

**Q: 优化会影响IoU吗？**  
A: 使用保守参数，对IoU影响很小，通常能提升视觉质量。

**Q: 可以关闭边缘优化吗？**  
A: 边缘优化已集成到 `render.py`，如需关闭需要修改代码。

**Q: 和baseline版本有什么区别？**  
A: 主要区别是2D mask渲染时自动应用形态学优化，其他步骤相同。

## 输出说明

- **模型路径**：`./output/<scene_name>_<timestamp>/`
- **数据路径**：`/home/bygpu/data/<video_name>_scene/`
- **渲染结果**：`<model_path>/train/ours_<iteration>/renders/`
- **Mask结果**：`<model_path>/train/ours_<iteration>/mask/`（已优化）


