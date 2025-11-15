# 使用 run_with_new_images.sh 运行完整流程

## 概述

`run_with_new_images.sh` 是一个一键运行脚本，可以自动完成从数据采集到GUI分割的所有步骤。

## 功能

这个脚本会自动执行以下步骤：

1. ✅ **数据处理**：使用 nerfstudio 处理视频/图像，运行 COLMAP 进行相机姿态估计
2. ✅ **基线训练**：训练 baseline 3D Gaussian Splatting 模型
3. ✅ **SAM掩码提取**：提取 Segment Anything 掩码并计算尺度
4. ✅ **对比特征训练**：训练用于分割的对比特征
5. ✅ **自动打开GUI**（可选）：完成后自动打开交互式分割GUI

## 使用方法

### 1. 配置参数

编辑 `run_with_new_images.sh`，修改以下参数：

```bash
PHOTOS_PATH="/path/to/your/video.mp4"  # 你的视频或图像文件夹路径
INPUT_TYPE="video"                      # "video" 或 "images"
FEATURE_LR="0.0025"                     # 对比特征学习率（可选）
TEST_LAST="40"                         # 测试集帧数（后N个作为测试集）
                                        # 例如：总共335张，TEST_LAST=40 → 前295张训练，后40张测试
                                        # 图片会按文件名排序（通常对应拍摄顺序）
AUTO_OPEN_GUI="1"                      # 自动打开GUI (1=是, 0=否)
```

#### 训练/测试集划分说明

- **划分方式**：按时间顺序划分（前N张训练，后M张测试）
- **图片排序**：图片按文件名字符串排序，通常对应拍摄顺序
- **示例**：
  - 总共335张图片，`TEST_LAST="40"` → 前295张训练，后40张测试
  - 总共200张图片，`TEST_LAST="20"` → 前180张训练，后20张测试
- **验证划分**：运行 `python scripts/check_train_test_split.py --data_path <数据路径> --test_last_n 40` 查看划分结果

### 2. 运行脚本

```bash
bash run_with_new_images.sh
```

### 3. 等待完成

脚本会自动完成所有步骤。如果设置了 `AUTO_OPEN_GUI="1"`，完成后会自动打开GUI。

## GUI使用说明

当GUI打开后，你可以进行交互式3D分割：

### 视角控制
- **左键拖拽**：旋转视角
- **中键拖拽**：平移
- **右键点击**：输入点提示（需要先选择分割模式）

### 分割步骤

1. **选择分割模式**：
   - `click mode`：单点模式（推荐用于作业）
   - `multi-click mode`：多点模式

2. **调整参数**：
   - `scale`：3D尺度（用于分割和聚类）
   - `score thresh`：分割相似度阈值

3. **选择渲染模式**：
   - `RGB`：显示原始RGB
   - `PCA`：显示PCA分解的特征
   - `SIMILARITY`：显示点提示的相似度图
   - `3D CLUSTER`：显示3D聚类结果

4. **进行分割**：
   - 在目标物体上右键点击选择点
   - 点击 `segment3D` 进行3D分割
   - 如果结果不满意，可以点击 `roll back` 撤销，或继续添加点提示

5. **保存结果**：
   - 点击 `save as` 保存分割结果
   - 保存的文件会在 `./segmentation_res/your_name.pt`

## 输出文件结构

运行完成后，你会得到以下结构：

```
./output/
  └── {scene_name}_{timestamp}/     # 训练好的模型
      ├── point_cloud/
      │   ├── iteration_30000/
      │   │   └── scene_point_cloud.ply
      │   └── iteration_10000/
      │       ├── contrastive_feature_point_cloud.ply
      │       └── scale_gate.pt
      └── ...

./data/
  └── {scene_name}_scene/           # 处理后的数据
      ├── images/                   # 图像
      ├── colmap/                   # COLMAP结果
      ├── sam_masks/                # SAM掩码
      └── mask_scales/              # 掩码尺度

./segmentation_res/                 # GUI保存的分割结果
  └── your_object.pt               # 3D高斯点的二进制掩码
```

## 验证工具

### 检查图片顺序

在训练前，可以检查图片是否按拍摄顺序排列：

```bash
python scripts/check_image_order.py --data_path <数据路径>
```

这会显示：
- 图片总数
- 前10张和后10张图片名称
- 文件名模式分析
- 顺序检查结果

### 检查训练/测试集划分

训练完成后，可以验证划分是否正确：

```bash
python scripts/check_train_test_split.py \
  --data_path <数据路径> \
  --test_last_n 40
```

这会显示：
- 总图片数
- 训练集和测试集的数量
- 划分范围（第X张到第Y张）
- 示例文件名

## 常见问题

### Q: 如何确认图片是按拍摄顺序排列的？

A: 运行 `python scripts/check_image_order.py --data_path <数据路径>` 检查图片顺序。nerfstudio处理后的图片通常有顺序编号（如 `frame_0001.jpg`），按字符串排序会保持拍摄顺序。

### Q: 如何设置训练集295个，测试集40个？

A: 如果总共有335张图片，只需设置 `TEST_LAST="40"`。系统会自动：
- 前295张作为训练集（335-40=295）
- 后40张作为测试集

### Q: 如果不想自动打开GUI怎么办？

A: 设置 `AUTO_OPEN_GUI="0"`，脚本完成后会显示手动运行GUI的命令。

### Q: 如何手动打开GUI？

A: 运行：
```bash
python saga_gui.py --model_path <MODEL_PATH> --data_path <DATA_PATH>
```

其中 `<MODEL_PATH>` 是 `./output/` 下的模型目录，`<DATA_PATH>` 是数据目录。

### Q: 脚本运行失败怎么办？

A: 
1. 检查 `PHOTOS_PATH` 是否正确
2. 确保已安装所有依赖（见 `environment.yml`）
3. 确保有足够的GPU内存
4. 检查 SAM checkpoint 是否存在：`third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth`

### Q: 数据量要求？

A: 建议至少 50+ 张图片（或视频帧），200+ 帧效果更好。图像尺寸建议 512×512 或 640×480。

## 下一步：优化模型

完成baseline后，你可以：

1. **优化分割质量**：使用 `scripts/refine_segmentation_masks.py` 优化mask边缘
2. **优化训练过程**：修改 `train_scene_optimized.py` 添加损失函数或调整超参数
3. **先分割后训练**：先用GUI分割出物体，然后只训练该物体的GS模型

## 参考

- 详细文档：`README_RUN.md`
- 分割优化：`scripts/README_SEGMENTATION_REFINEMENT.md`
- 优化思路：`OPTIMIZATION_IDEAS.md`

