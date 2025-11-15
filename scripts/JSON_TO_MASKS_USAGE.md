# 从JSON标注生成2D Mask使用说明

## 你的情况

- JSON文件在桌面：`frame_00303.json`, `frame_00311.json`, `frame_00319.json`, `frame_00327.json`, `frame_00335.json`
- 每个JSON包含多个物体：book, glasses, juice, pencil_case, umbrella

## 使用方法

### 1. 为frame_00303生成所有物体的mask

```bash
python scripts/json_to_masks.py \
  --json_file ~/Desktop/frame_00303.json \
  --output_dir ./gt_masks/frame_00303 \
  --image_size 640x480
```

如果不确定图片尺寸，可以先不指定，脚本会尝试从JSON读取：
```bash
python scripts/json_to_masks.py \
  --json_file ~/Desktop/frame_00303.json \
  --output_dir ./gt_masks/frame_00303
```

### 2. 只提取特定物体

```bash
# 只提取book和glasses
python scripts/json_to_masks.py \
  --json_file ~/Desktop/frame_00303.json \
  --output_dir ./gt_masks/frame_00303 \
  --objects book,glasses
```

### 3. 批量处理所有JSON文件

```bash
# 处理所有JSON文件
for json_file in ~/Desktop/frame_*.json; do
    frame_name=$(basename "$json_file" .json)
    python scripts/json_to_masks.py \
      --json_file "$json_file" \
      --output_dir "./gt_masks/${frame_name}"
done
```

## 输出

每个物体的mask会保存在：
```
./gt_masks/frame_00303/
  ├── book.png
  ├── glasses.png
  ├── juice.png
  ├── pencil_case.png
  └── umbrella.png
```

## JSON格式要求

脚本支持LabelMe格式的JSON：
- `shapes`: 标注列表
- 每个shape有：
  - `label`: 物体名称（如 "book"）
  - `shape_type`: "polygon", "rectangle", "circle"
  - `points`: 坐标点列表

## 如果JSON格式不同

如果JSON格式不同，告诉我格式，我可以修改脚本。

