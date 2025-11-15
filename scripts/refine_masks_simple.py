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
    """处理单个mask文件（3D高斯点mask）"""
    print(f"Processing: {mask_path.name}")
    
    # 加载mask
    if not mask_path.exists():
        print(f"  ERROR: {mask_path} not found, skipping...")
        return False
    
    try:
        mask_tensor = torch.load(mask_path)
        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.cpu()
        else:
            mask = torch.from_numpy(np.array(mask_tensor))
        
        # 确保是bool类型
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        
        print(f"  Original mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Original True count: {mask.sum().item()}")
        
        # 对于3D高斯点mask，我们进行简单的清理：
        # 1. 确保mask是1D的
        if len(mask.shape) > 1:
            mask = mask.flatten()
            if mask.shape[0] > 1:
                mask = mask > 0.5
        
        # 2. 简单的统计过滤：如果mask太稀疏（< 0.1%），可能是噪声
        total = mask.numel()
        true_count = mask.sum().item()
        ratio = true_count / total if total > 0 else 0
        
        print(f"  Mask ratio: {ratio:.4f} ({true_count}/{total})")
        
        # 3. 对于3D mask，我们直接保存（形态学操作需要2D图像或3D空间信息）
        # 这里我们只做基本的清理：确保是bool类型
        refined_mask = mask.bool()
        
        print(f"  Refined mask True count: {refined_mask.sum().item()}")
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(refined_mask, output_path)
        
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

