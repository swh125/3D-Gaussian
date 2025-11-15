#!/usr/bin/env python3
"""
轻微平滑mask，去除边缘锯齿，保持物体区域基本不变
"""

import sys
sys.path.insert(0, '.')

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm


def smooth_mask_slight(mask_path: Path, output_path: Path = None, kernel_size: int = 3):
    """
    轻微平滑mask
    
    Args:
        mask_path: 原始mask文件路径
        output_path: 输出路径（如果为None，覆盖原文件）
        kernel_size: 形态学操作kernel大小（默认3，轻微平滑）
    """
    # 加载mask
    mask_img = Image.open(mask_path)
    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')
    mask_np = np.array(mask_img)
    
    # 二值化
    mask_binary = (mask_np >= 128).astype(np.uint8) * 255
    
    # 轻微的闭运算：平滑边缘，去除小锯齿
    # kernel_size=3是轻微平滑，不会大幅改变区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_smooth = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    # 保存
    if output_path is None:
        output_path = mask_path
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(mask_smooth).save(output_path)
    return mask_smooth


def smooth_mask_directory(mask_dir: Path, output_dir: Path = None, kernel_size: int = 3):
    """
    批量平滑mask目录中的所有mask
    
    Args:
        mask_dir: 原始mask目录
        output_dir: 输出目录（如果为None，覆盖原文件）
        kernel_size: 形态学操作kernel大小
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        print(f"❌ Mask目录不存在: {mask_dir}")
        return
    
    if output_dir is None:
        output_dir = mask_dir
        print(f"⚠️  将覆盖原文件: {mask_dir}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 输出目录: {output_dir}")
    
    # 获取所有mask文件
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) == 0:
        print(f"⚠️  未找到mask文件: {mask_dir}")
        return
    
    print(f"📁 处理目录: {mask_dir}")
    print(f"📊 找到 {len(mask_files)} 个mask文件")
    print(f"🔧 Kernel大小: {kernel_size} (轻微平滑)")
    print()
    
    for mask_path in tqdm(mask_files, desc="平滑masks"):
        try:
            output_path = output_dir / mask_path.name
            smooth_mask_slight(mask_path, output_path, kernel_size=kernel_size)
        except Exception as e:
            print(f"⚠️  处理失败 {mask_path.name}: {e}")
    
    print()
    print(f"✓ 完成！平滑后的mask保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="轻微平滑mask，去除边缘锯齿")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="Mask目录路径（如 ./output/.../test/ours_30000/mask）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（如果为None，覆盖原文件）")
    parser.add_argument("--kernel_size", type=int, default=3,
                       help="形态学操作kernel大小（默认3，轻微平滑。越大越平滑，但可能改变区域）")
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    
    smooth_mask_directory(mask_dir, output_dir, kernel_size=args.kernel_size)


if __name__ == "__main__":
    main()

