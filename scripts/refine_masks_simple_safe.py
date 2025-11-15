#!/usr/bin/env python3
"""
简单的mask边缘优化 - 保守方法，确保不报错，保持顺序，提高IoU
只做最基本的形态学操作，去除边缘光晕
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def refine_mask_simple(mask: np.ndarray, opening_kernel: int = 2, closing_kernel: int = 3) -> np.ndarray:
    """
    简单的边缘优化：保守的形态学操作
    只去除明显的光晕，尽量保持原始mask大小以提高IoU
    """
    mask_uint8 = mask.astype(np.uint8) if mask.max() <= 1.0 else (mask / 255).astype(np.uint8)
    
    # 1. 轻微开运算：只去除边缘的小噪点（kernel很小，不会大幅缩小mask）
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    
    # 2. 轻微闭运算：填补小洞（kernel小，不会大幅扩大mask）
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    return mask_uint8


def process_mask_file(mask_path: Path, output_path: Path, opening_kernel: int = 2, closing_kernel: int = 3):
    """处理单个mask文件"""
    try:
        # 加载mask
        if mask_path.suffix == '.pt':
            mask_tensor = torch.load(mask_path)
            if isinstance(mask_tensor, torch.Tensor):
                mask = mask_tensor.cpu().numpy()
            else:
                mask = np.array(mask_tensor)
            # 处理维度
            if len(mask.shape) > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()
            mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to read {mask_path}")
                return
            mask = (mask > 127).astype(np.uint8)
        
        # 简单优化
        refined = refine_mask_simple(mask, opening_kernel, closing_kernel)
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == '.pt':
            torch.save(torch.from_numpy(refined.astype(bool)), output_path)
        else:
            cv2.imwrite(str(output_path), (refined * 255).astype(np.uint8))
            
    except Exception as e:
        print(f"Error processing {mask_path.name}: {e}")
        # 如果出错，直接复制原文件
        import shutil
        shutil.copy2(mask_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Simple mask edge refinement - safe and conservative")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="Directory containing input masks")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for refined masks")
    parser.add_argument("--opening_kernel", type=int, default=2,
                       help="Opening kernel size (default: 2, small to preserve IoU)")
    parser.add_argument("--closing_kernel", type=int, default=3,
                       help="Closing kernel size (default: 3, small to preserve IoU)")
    parser.add_argument("--mask_ext", type=str, default=".pt",
                       help="Mask file extension (.pt or .png)")
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    
    # IMPORTANT: Sort by filename to maintain temporal order
    mask_files = sorted(list(mask_dir.glob(f"*{args.mask_ext}")), key=lambda x: x.name)
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Processing in temporal order (sorted by filename)")
    print(f"Opening kernel: {args.opening_kernel}, Closing kernel: {args.closing_kernel}")
    print()
    
    # 处理每个mask
    for mask_path in tqdm(mask_files, desc="Refining masks"):
        output_path = output_dir / mask_path.name
        process_mask_file(mask_path, output_path, args.opening_kernel, args.closing_kernel)
    
    print(f"\n✓ Refined masks saved to {output_dir}")
    print(f"  Total: {len(mask_files)} masks processed")


if __name__ == "__main__":
    main()

