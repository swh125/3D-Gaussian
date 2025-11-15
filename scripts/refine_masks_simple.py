#!/usr/bin/env python3
"""
简单的mask优化脚本 - 使用方法1（形态学细化）
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

def morphological_refinement(mask: np.ndarray, 
                           opening_kernel: int = 3,
                           closing_kernel: int = 5,
                           remove_small: int = 100) -> np.ndarray:
    """
    形态学细化：去除光晕和噪点
    """
    mask_uint8 = mask.astype(np.uint8)
    
    # 1. 开运算：去除边缘的小噪点和光晕
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    
    # 2. 闭运算：填补小洞
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. 移除小连通域
    if remove_small > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels > 1:
            # 找到最大连通域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # 保留最大连通域
            mask_uint8 = (labels == largest_label).astype(np.uint8)
    
    return mask_uint8


def process_mask(mask_path: Path, output_path: Path):
    """处理单个mask文件"""
    print(f"Processing: {mask_path.name}")
    
    # 加载mask
    if not mask_path.exists():
        print(f"  ERROR: {mask_path} not found, skipping...")
        return False
    
    try:
        mask_tensor = torch.load(mask_path)
        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.cpu().numpy()
        else:
            mask = mask_tensor
        
        # 处理多通道或批次维度
        if len(mask.shape) > 2:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask.squeeze()
        
        # 确保是二值mask
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)
        
        print(f"  Original mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # 应用形态学细化
        refined = morphological_refinement(mask, opening_kernel=3, closing_kernel=5, remove_small=100)
        
        print(f"  Refined mask shape: {refined.shape}, dtype: {refined.dtype}")
        print(f"  Original pixels: {mask.sum()}, Refined pixels: {refined.sum()}")
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(refined.astype(bool)), output_path)
        
        print(f"  ✓ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {mask_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置
    MASK_DIR = Path("./segmentation_res")
    OUTPUT_DIR = Path("./segmentation_res_refined")
    
    # 要处理的文件列表
    MASK_FILES = [
        "book_optimized.pt",
        "juice_optimized.pt",
        "pencil_case_optimized.pt",
        "umbrella_optimized.pt",
        "glasses_optimized.pt"
    ]
    
    print("=" * 60)
    print("Mask Refinement - Method 1 (Morphological)")
    print("=" * 60)
    print(f"Input directory: {MASK_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for mask_name in MASK_FILES:
        mask_path = MASK_DIR / mask_name
        output_path = OUTPUT_DIR / mask_name
        
        if process_mask(mask_path, output_path):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Completed: {success_count}/{len(MASK_FILES)} masks processed")
    print(f"Refined masks saved to: {OUTPUT_DIR}")
    print("=" * 60)

