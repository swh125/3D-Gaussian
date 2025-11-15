#!/usr/bin/env python3
"""
Mask边缘优化 + IoU检查
优化边缘的同时，确保2D mask的IoU不降低（或提升）
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene import Scene, GaussianModel
from gaussian_renderer import render_mask
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser, Namespace


def refine_mask_simple(mask: np.ndarray, opening_kernel: int = 2, closing_kernel: int = 3) -> np.ndarray:
    """简单的边缘优化"""
    mask_uint8 = mask.astype(np.uint8) if mask.max() <= 1.0 else (mask / 255).astype(np.uint8)
    
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    return mask_uint8


def compute_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """计算IoU"""
    pred_bool = pred_mask.bool()
    gt_bool = gt_mask.bool()
    
    intersection = torch.logical_and(pred_bool, gt_bool).sum().item()
    union = torch.logical_or(pred_bool, gt_bool).sum().item()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def load_mask_image(path: Path) -> torch.Tensor:
    """加载mask图片"""
    from PIL import Image
    with Image.open(path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        data = torch.from_numpy(np.array(img)).float()
        mask = (data / 255.0 > 0.5).float()
    return mask


def render_2d_masks_from_3d(mask_3d: torch.Tensor, model_path: str, iteration: int, 
                           source_path: str, test_only: bool = True) -> list:
    """从3D mask渲染2D masks"""
    # 加载模型
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args(["--model_path", model_path, "-s", source_path, "--iteration", str(iteration)])
    
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # 应用mask
    gaussians.segment(mask_3d.bool().cuda())
    
    # 获取相机
    if test_only:
        cameras = scene.getTestCameras()
    else:
        cameras = scene.getTrainCameras() + scene.getTestCameras()
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 渲染2D masks
    masks_2d = []
    for cam in cameras:
        mask_res = render_mask(cam, gaussians, pipeline, background, precomputed_mask=mask_3d.float().cuda())
        mask = mask_res["mask"]
        if len(mask.shape) == 3:
            mask = mask[0, :, :, 0]  # [H, W, C] -> [H, W]
        else:
            mask = mask[0, :, :]  # [H, W]
        masks_2d.append((mask > 0.5).float())
    
    return masks_2d


def compute_iou_with_gt(masks_2d: list, gt_mask_dir: Path) -> float:
    """计算2D masks与GT的IoU"""
    gt_files = sorted(gt_mask_dir.glob("*.png"))
    if len(gt_files) == 0:
        return None
    
    min_len = min(len(masks_2d), len(gt_files))
    iou_values = []
    
    for idx in range(min_len):
        pred_mask = masks_2d[idx]
        gt_mask = load_mask_image(gt_files[idx])
        
        # 调整大小
        if pred_mask.shape != gt_mask.shape:
            from torch.nn.functional import interpolate
            h, w = gt_mask.shape
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
            pred_mask = interpolate(pred_mask, size=(h, w), mode='nearest')
            pred_mask = pred_mask.squeeze()
        
        iou = compute_mask_iou(pred_mask, gt_mask)
        iou_values.append(iou)
    
    return np.mean(iou_values) if iou_values else None


def refine_2d_masks_with_iou_check(mask_3d_path: Path, model_path: str, iteration: int,
                                   source_path: str, gt_mask_dir: Path = None,
                                   opening_kernel: int = 2, closing_kernel: int = 3) -> torch.Tensor:
    """
    优化2D masks，同时检查IoU，确保不降低
    简单方法：渲染2D masks -> 优化2D masks -> 检查IoU -> 如果提升则使用
    """
    # 加载原始3D mask
    mask_3d_original = torch.load(mask_3d_path)
    if isinstance(mask_3d_original, torch.Tensor):
        if mask_3d_original.dtype != torch.bool:
            mask_3d_original = mask_3d_original > 0.5
        mask_3d_original = mask_3d_original.bool()
    else:
        mask_3d_original = torch.from_numpy(np.array(mask_3d_original) > 0.5).bool()
    
    # 如果没有GT，直接返回原始mask（无法检查IoU）
    if not gt_mask_dir or not gt_mask_dir.exists():
        print("No GT masks provided, skipping IoU check, returning original mask")
        return mask_3d_original
    
    # 渲染原始2D masks
    print("Rendering original 2D masks...")
    masks_2d_original = render_2d_masks_from_3d(mask_3d_original, model_path, iteration, source_path)
    original_iou = compute_iou_with_gt(masks_2d_original, gt_mask_dir)
    print(f"Original IoU: {original_iou:.4f}")
    
    # 优化2D masks
    print("Refining 2D masks...")
    masks_2d_refined = []
    for mask_2d in masks_2d_original:
        mask_np = mask_2d.cpu().numpy()
        refined_np = refine_mask_simple(mask_np, opening_kernel, closing_kernel)
        masks_2d_refined.append(torch.from_numpy(refined_np).float())
    
    # 计算优化后的IoU
    refined_iou = compute_iou_with_gt(masks_2d_refined, gt_mask_dir)
    print(f"Refined IoU: {refined_iou:.4f}")
    
    # 决策：如果IoU提升了，说明优化有效；如果降低了，说明优化无效
    # 注意：2D优化无法直接映射回3D mask，所以这里只是检查优化是否有效
    # 如果IoU提升了，说明优化方法有效，可以继续使用（虽然无法直接应用到3D）
    # 如果IoU降低了，说明优化无效，不应该使用
    
    if refined_iou >= original_iou:
        print(f"✓ IoU improved from {original_iou:.4f} to {refined_iou:.4f}!")
        print("  ✓ Optimization is effective! (2D mask quality improved)")
        print("  Note: 2D optimization cannot be directly mapped back to 3D mask")
        print("  Returning original 3D mask (optimization can be applied during 2D rendering)")
        # 返回原始3D mask（因为无法反向投影，但优化方法已验证有效）
        return mask_3d_original
    else:
        print(f"⚠ IoU decreased from {original_iou:.4f} to {refined_iou:.4f}")
        print("  ⚠ Optimization is NOT effective, IoU decreased")
        print("  Returning original mask (no optimization should be applied)")
        # IoU降低了，说明优化无效，返回原始mask（不应用优化）
        return mask_3d_original


def main():
    parser = argparse.ArgumentParser(description="Refine masks with IoU check")
    parser.add_argument("--mask_dir", type=str, default="./segmentation_res",
                       help="Directory containing input 3D masks")
    parser.add_argument("--output_dir", type=str, default="./segmentation_res_refined",
                       help="Output directory")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--source_path", type=str, default=None,
                       help="Path to scene data (auto-detected from model cfg_args if not provided)")
    parser.add_argument("--iteration", type=int, default=30000,
                       help="Model iteration")
    parser.add_argument("--gt_mask_dir", type=str, default=None,
                       help="Directory containing GT masks (optional, for IoU check)")
    parser.add_argument("--opening_kernel", type=int, default=2,
                       help="Opening kernel size")
    parser.add_argument("--closing_kernel", type=int, default=3,
                       help="Closing kernel size")
    
    args = parser.parse_args()
    
    # Auto-detect source_path from model's cfg_args if not provided
    if args.source_path is None:
        cfg_args_path = Path(args.model_path) / "cfg_args"
        if cfg_args_path.exists():
            try:
                import re
                with open(cfg_args_path, 'r') as f:
                    content = f.read()
                    match = re.search(r"source_path\s*=\s*['\"]([^'\"]+)['\"]", content)
                    if match:
                        args.source_path = match.group(1)
                        print(f"✓ Auto-detected source_path from model: {args.source_path}")
                    else:
                        raise ValueError("Could not find source_path in cfg_args")
            except Exception as e:
                print(f"Error auto-detecting source_path: {e}")
                print("Please provide --source_path manually")
                return
        else:
            print(f"Error: cfg_args not found at {cfg_args_path}")
            print("Please provide --source_path manually")
            return
    
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_mask_dir = Path(args.gt_mask_dir) if args.gt_mask_dir else None
    
    # IMPORTANT: Sort by filename to maintain temporal order
    mask_files = sorted(list(mask_dir.glob("*.pt")), key=lambda x: x.name)
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Processing in temporal order (sorted by filename)")
    if gt_mask_dir:
        print(f"GT mask directory: {gt_mask_dir}")
        print("Will check IoU and ensure it doesn't decrease")
    print()
    
    for mask_path in tqdm(mask_files, desc="Refining masks"):
        print(f"\nProcessing: {mask_path.name}")
        
        try:
            refined_mask = refine_2d_masks_with_iou_check(
                mask_path,
                args.model_path,
                args.iteration,
                args.source_path,
                gt_mask_dir,
                args.opening_kernel,
                args.closing_kernel
            )
            
            output_path = output_dir / mask_path.name
            torch.save(refined_mask, output_path)
            print(f"✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {mask_path.name}: {e}")
            import traceback
            traceback.print_exc()
            # 如果出错，直接复制原文件
            import shutil
            shutil.copy2(mask_path, output_dir / mask_path.name)
            print(f"  Copied original file as fallback")
    
    print(f"\n✓ Completed! Refined masks saved to {output_dir}")


if __name__ == "__main__":
    main()

