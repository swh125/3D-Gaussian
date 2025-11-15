#!/usr/bin/env python3
"""
检查图片是否按拍摄顺序排列
用于验证训练/测试集划分是否正确
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict


def check_image_order(data_path, images_dir="images"):
    """检查图片顺序"""
    images_path = Path(data_path) / images_dir
    
    if not images_path.exists():
        print(f"错误: 图片目录不存在: {images_path}")
        return False
    
    # 获取所有图片文件
    image_files = sorted([f for f in images_path.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(image_files) == 0:
        print(f"错误: 在 {images_path} 中未找到图片文件")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"\n图片目录: {images_path}")
    print("\n前10张图片:")
    for i, img in enumerate(image_files[:10]):
        print(f"  {i+1:3d}. {img.name}")
    
    if len(image_files) > 10:
        print(f"\n... (共 {len(image_files)} 张)")
        print("\n后10张图片:")
        for i, img in enumerate(image_files[-10:], start=len(image_files)-9):
            print(f"  {i:3d}. {img.name}")
    
    # 检查文件名模式
    print("\n文件名模式分析:")
    name_patterns = defaultdict(int)
    for img in image_files[:20]:  # 检查前20个
        name = img.stem
        # 尝试提取数字
        import re
        numbers = re.findall(r'\d+', name)
        if numbers:
            name_patterns[f"包含数字: {numbers[-1]}"] += 1
        else:
            name_patterns["无数字"] += 1
    
    for pattern, count in name_patterns.items():
        print(f"  {pattern}: {count} 个文件")
    
    # 检查是否按数字顺序
    print("\n顺序检查:")
    has_sequential_numbers = True
    prev_num = None
    for i, img in enumerate(image_files[:20]):
        import re
        numbers = re.findall(r'\d+', img.stem)
        if numbers:
            num = int(numbers[-1])
            if prev_num is not None:
                if num < prev_num:
                    print(f"  ⚠️  警告: 第 {i+1} 张图片的数字 ({num}) 小于前一张 ({prev_num})")
                    has_sequential_numbers = False
            prev_num = num
    
    if has_sequential_numbers:
        print("  ✅ 前20张图片按数字顺序排列")
    
    print(f"\n总结:")
    print(f"  - 总图片数: {len(image_files)}")
    print(f"  - 排序方式: 按文件名字符串排序")
    print(f"  - 第一张: {image_files[0].name}")
    print(f"  - 最后一张: {image_files[-1].name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="检查图片顺序")
    parser.add_argument("--data_path", type=str, required=True,
                       help="数据目录路径（包含images/文件夹）")
    parser.add_argument("--images_dir", type=str, default="images",
                       help="图片子目录名（默认: images）")
    
    args = parser.parse_args()
    
    check_image_order(args.data_path, args.images_dir)


if __name__ == "__main__":
    main()

