#!/usr/bin/env python3
"""
优化的mask渲染脚本
通过后处理优化mask质量，解决opacity导致的"雾气"问题
不修改原始render.py和gaussian_renderer
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening
import cv2


def optimize_mask_postprocess(mask: torch.Tensor, fill_holes: bool = True, 
                             opening_kernel: int = 2, closing_kernel: int = 3) -> torch.Tensor:
    """
    后处理优化mask，解决opacity导致的半透明问题
    
    Args:
        mask: 原始渲染的mask (H, W) 或 (1, H, W)
        fill_holes: 是否填充空洞
        opening_kernel: 开运算kernel大小
        closing_kernel: 闭运算kernel大小
    
    Returns:
        优化后的mask
    """
    # 转换为numpy
    if len(mask.shape) == 3:
        mask = mask[0]
    mask_np = mask.cpu().numpy()
    
    # 使用更严格的阈值，减少半透明区域的影响
    # 先降低阈值保留更多有效区域，然后通过形态学操作清理
    mask_binary = (mask_np >= 0.3).astype(np.uint8) * 255
    
    if fill_holes:
        # 填充mask内部的小空洞
        mask_binary = binary_fill_holes(mask_binary > 127).astype(np.uint8) * 255
    
    # 形态学操作
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
    
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close)
    
    # 转换回tensor
    mask_optimized = torch.from_numpy(mask_binary.astype(np.float32) / 255.0).to(mask.device)
    
    return mask_optimized


def process_rendered_masks(mask_dir: Path, output_dir: Path = None, 
                          fill_holes: bool = True, opening_kernel: int = 2, 
                          closing_kernel: int = 3):
    """
    批量处理已渲染的mask文件
    
    Args:
        mask_dir: 原始mask目录
        output_dir: 输出目录（如果为None，覆盖原文件）
        fill_holes: 是否填充空洞
        opening_kernel: 开运算kernel大小
        closing_kernel: 闭运算kernel大小
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        print(f"❌ Mask目录不存在: {mask_dir}")
        return
    
    if output_dir is None:
        output_dir = mask_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有mask文件
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) == 0:
        print(f"⚠️  未找到mask文件: {mask_dir}")
        return
    
    print(f"📁 处理目录: {mask_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 找到 {len(mask_files)} 个mask文件")
    print()
    
    for i, mask_path in enumerate(mask_files):
        # 加载mask
        mask_img = Image.open(mask_path)
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')
        mask = torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0)
        
        # 优化
        mask_optimized = optimize_mask_postprocess(
            mask, fill_holes=fill_holes, 
            opening_kernel=opening_kernel, 
            closing_kernel=closing_kernel
        )
        
        # 保存
        output_path = output_dir / mask_path.name
        mask_save = (mask_optimized.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_save).save(output_path)
        
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(mask_files)}")
    
    print()
    print(f"✓ 完成！优化后的mask保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="优化已渲染的mask，解决opacity导致的雾气问题")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="原始mask目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（如果为None，覆盖原文件）")
    parser.add_argument("--fill_holes", action="store_true", default=True,
                       help="填充mask内部空洞（默认开启）")
    parser.add_argument("--no_fill_holes", action="store_false", dest="fill_holes",
                       help="不填充空洞")
    parser.add_argument("--opening_kernel", type=int, default=2,
                       help="开运算kernel大小（默认2）")
    parser.add_argument("--closing_kernel", type=int, default=3,
                       help="闭运算kernel大小（默认3）")
    
    args = parser.parse_args()
    
    process_rendered_masks(
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        fill_holes=args.fill_holes,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel
    )


if __name__ == "__main__":
    main()

