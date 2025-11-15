#!/usr/bin/env python3
"""
从JSON标注文件生成每个物体的2D mask
简单脚本，不复杂
"""

import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import argparse


def load_json_annotation(json_path: Path):
    """加载JSON标注文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_mask_from_shapes(shapes, image_size, label_name=None):
    """
    从shapes创建mask
    
    Args:
        shapes: 标注的shapes列表
        image_size: (width, height)
        label_name: 如果指定，只提取这个label的shapes
    
    Returns:
        mask: numpy array (H, W), 0=背景, 255=物体
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in shapes:
        label = shape.get('label', '')
        
        # 如果指定了label_name，只处理这个label
        if label_name and label != label_name:
            continue
        
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])
        
        if shape_type == 'polygon' and len(points) >= 3:
            # 多边形
            polygon = [tuple(p) for p in points]
            draw.polygon(polygon, fill=255)
        elif shape_type == 'rectangle' and len(points) == 2:
            # 矩形
            x1, y1 = points[0]
            x2, y2 = points[1]
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == 'circle' and len(points) == 2:
            # 圆形（用椭圆近似）
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 计算半径
            r = ((x2-x1)**2 + (y2-y1)**2)**0.5
            draw.ellipse([x1-r, y1-r, x1+r, y1+r], fill=255)
    
    return np.array(mask)


def extract_image_size_from_json(json_data):
    """从JSON数据中提取图片尺寸"""
    if 'imageWidth' in json_data and 'imageHeight' in json_data:
        return (json_data['imageWidth'], json_data['imageHeight'])
    elif 'imageData' in json_data:
        # 如果有imageData，可以解码获取尺寸
        import base64
        from io import BytesIO
        try:
            img_data = base64.b64decode(json_data['imageData'])
            img = Image.open(BytesIO(img_data))
            return img.size
        except:
            pass
    
    # 默认尺寸（如果无法获取，需要用户提供）
    return None


def get_all_labels(json_data):
    """获取JSON中所有的label"""
    labels = set()
    shapes = json_data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label', '')
        if label:
            labels.add(label)
    return sorted(list(labels))


def main():
    parser = argparse.ArgumentParser(description="从JSON标注生成每个物体的2D mask")
    parser.add_argument("--json_file", type=str, required=True,
                       help="JSON标注文件路径（如 frame_00303.json）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（每个物体的mask会保存在这里）")
    parser.add_argument("--image_size", type=str, default=None,
                       help="图片尺寸，格式：WxH（如 640x480），如果不提供会尝试从JSON读取")
    parser.add_argument("--objects", type=str, default=None,
                       help="要提取的物体列表，用逗号分隔（如 book,glasses,juice），如果不提供会提取所有")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"错误: JSON文件不存在: {json_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载JSON
    print(f"加载JSON文件: {json_path}")
    json_data = load_json_annotation(json_path)
    
    # 获取图片尺寸
    if args.image_size:
        w, h = map(int, args.image_size.split('x'))
        image_size = (w, h)
    else:
        image_size = extract_image_size_from_json(json_data)
        if image_size is None:
            print("错误: 无法从JSON获取图片尺寸，请使用 --image_size WxH 指定")
            return
    
    print(f"图片尺寸: {image_size[0]}x{image_size[1]}")
    
    # 获取所有label
    all_labels = get_all_labels(json_data)
    print(f"找到的物体: {all_labels}")
    
    # 确定要提取的物体
    if args.objects:
        target_labels = [l.strip() for l in args.objects.split(',')]
    else:
        target_labels = all_labels
    
    print(f"要提取的物体: {target_labels}")
    print()
    
    # 获取shapes
    shapes = json_data.get('shapes', [])
    
    # 从JSON文件名提取帧号（如 frame_00303.json -> frame_00303）
    frame_name = json_path.stem  # 去掉扩展名，得到 frame_00303
    
    # 为每个物体生成mask
    for label in target_labels:
        print(f"生成 {label} 的mask...")
        mask = create_mask_from_shapes(shapes, image_size, label_name=label)
        
        # 保存mask，包含帧号：frame_00303_book.png
        mask_path = output_dir / f"{frame_name}_{label}.png"
        mask_img = Image.fromarray(mask)
        mask_img.save(mask_path)
        # 输出绝对路径
        abs_path = mask_path.resolve()
        print(f"  ✓ 保存到: {abs_path}")
    
    abs_output_dir = output_dir.resolve()
    print(f"\n✓ 完成！所有mask保存在: {abs_output_dir}")


if __name__ == "__main__":
    main()

