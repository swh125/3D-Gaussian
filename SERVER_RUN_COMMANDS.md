# 服务器运行命令

## 1. 更新代码

```bash
# 进入项目目录
cd /path/to/SegAnyGAussians-2

# 拉取最新代码
git pull origin main
```

## 2. 配置并运行一键脚本

### 2.1 编辑配置文件

```bash
# 编辑 run_with_new_images.sh，修改以下变量：
nano run_with_new_images.sh
# 或
vim run_with_new_images.sh
```

需要修改的变量：
```bash
PHOTOS_PATH="/path/to/your/video.mp4"  # 你的视频路径
INPUT_TYPE="video"                      # "video" 或 "images"
TEST_LAST="40"                         # 测试集数量（后40张）
AUTO_OPEN_GUI="1"                      # 自动打开GUI (1=是, 0=否)
```

### 2.2 运行一键脚本

```bash
# 给脚本添加执行权限（如果还没有）
chmod +x run_with_new_images.sh

# 运行脚本
bash run_with_new_images.sh
```

## 3. 验证顺序（可选，但推荐）

### 3.1 检查图片顺序

```bash
# 在数据处理完成后，检查图片顺序
python scripts/check_image_order.py \
  --data_path /path/to/data/your_scene
```

### 3.2 验证训练/测试集划分

```bash
# 验证划分是否正确（最后40张作为测试集）
python scripts/verify_temporal_order.py \
  --data_path /path/to/data/your_scene \
  --test_last_n 40
```

### 3.3 检查划分结果

```bash
# 查看训练/测试集划分详情
python scripts/check_train_test_split.py \
  --data_path /path/to/data/your_scene \
  --test_last_n 40
```

## 4. 完整示例命令

假设你的视频在 `/home/user/videos/my_video.mp4`，数据输出到 `/home/user/data/my_video_scene`：

```bash
# 1. 更新代码
cd /path/to/SegAnyGAussians-2
git pull origin main

# 2. 编辑配置
nano run_with_new_images.sh
# 修改：
#   PHOTOS_PATH="/home/user/videos/my_video.mp4"
#   TEST_LAST="40"
#   AUTO_OPEN_GUI="1"

# 3. 运行（这会自动完成所有步骤，包括打开GUI）
bash run_with_new_images.sh

# 4. 验证（在数据处理完成后）
python scripts/verify_temporal_order.py \
  --data_path /home/user/data/my_video_scene \
  --test_last_n 40
```

## 5. 如果不想自动打开GUI

如果设置 `AUTO_OPEN_GUI="0"`，脚本完成后会显示手动运行GUI的命令：

```bash
# 手动打开GUI
python saga_gui.py \
  --model_path ./output/my_video_scene_20241113_123456 \
  --data_path /home/user/data/my_video_scene
```

## 6. 检查运行状态

### 查看模型输出路径

脚本运行时会显示模型路径，例如：
```
Model path: ./output/my_video_scene_20241113_123456
```

### 查看训练进度

训练过程中会显示迭代进度和损失值。

### 查看输出文件

```bash
# 查看渲染结果
ls -lh ./output/my_video_scene_*/train/ours_30000/renders/

# 查看测试集结果
ls -lh ./output/my_video_scene_*/test/ours_30000/renders/
```

## 7. 常见问题

### 如果git pull失败

```bash
# 如果有本地修改，先保存或丢弃
git stash  # 保存本地修改
git pull origin main
git stash pop  # 恢复本地修改（如果需要）
```

### 如果路径不存在

确保：
- 视频文件路径正确
- 有读取权限
- 输出目录有写入权限

### 如果GPU内存不足

在 `run_with_new_images.sh` 中调整：
```bash
NUM_DOWNSCALES="4"  # 增大下采样（减少显存使用）
DOWNSAMPLE="4"      # SAM mask下采样
```

## 8. 后台运行（可选）

如果需要后台运行：

```bash
# 使用nohup后台运行
nohup bash run_with_new_images.sh > run.log 2>&1 &

# 查看日志
tail -f run.log

# 查看进程
ps aux | grep run_with_new_images
```

## 9. 停止运行

```bash
# 如果在前台运行，按 Ctrl+C

# 如果在后台运行，找到进程ID并kill
ps aux | grep run_with_new_images
kill <PID>
```

