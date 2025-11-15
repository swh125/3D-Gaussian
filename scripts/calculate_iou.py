#!/usr/bin/env python3
"""
计算2D mask的IoU
比较GT标注（JSON）和预测mask（PNG）
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


def create_mask_from_shapes(shapes, image_size, label_name=None, combine_all=False):
    """
    从shapes创建mask
    
    Args:
        shapes: 标注的shapes列表
        image_size: (width, height)
        label_name: 如果指定，只提取这个label的shapes
        combine_all: 如果True，合并所有物体的mask（用于计算整体IoU）
    
    Returns:
        mask: numpy array (H, W), 0=背景, 1=物体
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in shapes:
        label = shape.get('label', '')
        
        # 如果指定了label_name，只处理这个label
        if label_name and label != label_name:
            continue
        
        # 如果combine_all=False且指定了label_name，只处理这个label
        # 如果combine_all=True，处理所有label
        if not combine_all and label_name and label != label_name:
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
    
    # 转换为0-1的numpy数组
    mask_array = np.array(mask) / 255.0
    return mask_array


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


def load_mask_image(mask_path: Path):
    """加载mask图片，转换为0-1的numpy数组"""
    img = Image.open(mask_path)
    # 转换为灰度
    if img.mode != 'L':
        img = img.convert('L')
    mask = np.array(img)
    # 归一化到0-1
    if mask.max() > 1:
        mask = mask / 255.0
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5):
    """
    计算IoU
    
    Args:
        pred_mask: 预测mask (H, W), 值在0-1之间
        gt_mask: GT mask (H, W), 值在0-1之间
        threshold: 二值化阈值
    
    Returns:
        iou: IoU值
    """
    # 二值化
    pred_binary = (pred_mask >= threshold).astype(np.float32)
    gt_binary = (gt_mask >= threshold).astype(np.float32)
    
    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


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
    parser = argparse.ArgumentParser(description="计算2D mask的IoU")
    parser.add_argument("--json_file", type=str, required=True,
                       help="GT JSON标注文件路径（如 frame_00303.json）")
    parser.add_argument("--pred_mask", type=str, required=True,
                       help="预测mask图片路径（如 00007.png）")
    parser.add_argument("--image_size", type=str, default=None,
                       help="图片尺寸，格式：WxH（如 640x480），如果不提供会尝试从JSON读取")
    parser.add_argument("--object", type=str, default=None,
                       help="要计算IoU的物体名称（如 book），如果不提供会合并所有物体计算整体IoU")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="二值化阈值（默认0.5）")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    pred_mask_path = Path(args.pred_mask)
    
    if not json_path.exists():
        print(f"❌ 错误: JSON文件不存在: {json_path}")
        return
    
    if not pred_mask_path.exists():
        print(f"❌ 错误: 预测mask文件不存在: {pred_mask_path}")
        return
    
    # 加载JSON
    print(f"📄 加载JSON文件: {json_path}")
    json_data = load_json_annotation(json_path)
    
    # 获取图片尺寸
    if args.image_size:
        w, h = map(int, args.image_size.split('x'))
        image_size = (w, h)
    else:
        image_size = extract_image_size_from_json(json_data)
        if image_size is None:
            print("❌ 错误: 无法从JSON获取图片尺寸，请使用 --image_size WxH 指定")
            return
    
    print(f"📐 图片尺寸: {image_size[0]}x{image_size[1]}")
    
    # 加载预测mask
    print(f"🖼️  加载预测mask: {pred_mask_path}")
    pred_mask = load_mask_image(pred_mask_path)
    
    # 检查尺寸是否匹配
    if pred_mask.shape[1] != image_size[0] or pred_mask.shape[0] != image_size[1]:
        print(f"⚠️  警告: 尺寸不匹配！")
        print(f"   JSON尺寸: {image_size[0]}x{image_size[1]}")
        print(f"   Mask尺寸: {pred_mask.shape[1]}x{pred_mask.shape[0]}")
        print(f"   将调整mask尺寸以匹配JSON...")
        # 调整mask尺寸
        pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_mask_img = pred_mask_img.resize(image_size, Image.NEAREST)
        pred_mask = np.array(pred_mask_img) / 255.0
    
    # 获取所有label
    all_labels = get_all_labels(json_data)
    print(f"🏷️  找到的物体: {all_labels}")
    
    # 生成GT mask
    shapes = json_data.get('shapes', [])
    
    if args.object:
        # 计算单个物体的IoU
        if args.object not in all_labels:
            print(f"❌ 错误: 物体 '{args.object}' 不在JSON中")
            print(f"   可用的物体: {all_labels}")
            return
        
        print(f"🎯 计算物体 '{args.object}' 的IoU...")
        gt_mask = create_mask_from_shapes(shapes, image_size, label_name=args.object, combine_all=False)
    else:
        # 合并所有物体计算整体IoU
        print(f"🎯 计算所有物体合并后的IoU...")
        gt_mask = create_mask_from_shapes(shapes, image_size, label_name=None, combine_all=True)
    
    # 计算IoU
    iou = compute_iou(pred_mask, gt_mask, threshold=args.threshold)
    
    print()
    print("=" * 60)
    print(f"📊 IoU结果:")
    if args.object:
        print(f"   物体: {args.object}")
    else:
        print(f"   所有物体（合并）")
    print(f"   IoU: {iou:.4f} ({iou*100:.2f}%)")
    print("=" * 60)
    
    # 额外统计信息
    pred_binary = (pred_mask >= args.threshold).astype(np.float32)
    gt_binary = (gt_mask >= args.threshold).astype(np.float32)
    
    pred_area = pred_binary.sum()
    gt_area = gt_binary.sum()
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    print(f"\n📈 统计信息:")
    print(f"   GT mask面积: {gt_area:.0f} 像素")
    print(f"   预测mask面积: {pred_area:.0f} 像素")
    print(f"   交集: {intersection:.0f} 像素")
    print(f"   并集: {union:.0f} 像素")


if __name__ == "__main__":
    main()

