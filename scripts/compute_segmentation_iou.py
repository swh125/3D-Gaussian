#!/usr/bin/env python
"""
Compute IoU (Intersection over Union) for segmentation masks.

This script compares predicted masks (from 3DGS rendering) with ground-truth masks.

Typical usage:
    # Option 1: Compare rendered masks with GT masks from directories
    python scripts/compute_segmentation_iou.py \
        --pred_mask_dir ./output/model/test/ours_30000/mask \
        --gt_mask_dir ./data/video_scene/gt_masks/test

    # Option 2: Compare 3D mask (from GUI save) with GT masks
    python scripts/compute_segmentation_iou.py \
        --pred_mask_3d ./segmentation_res/book_baseline.pt \
        --gt_mask_dir ./data/video_scene/gt_masks/test \
        --model_path ./output/model \
        --iteration 30000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_mask


def mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between two binary masks."""
    pred_bool = pred_mask.bool()
    gt_bool = gt_mask.bool()
    
    intersection = torch.logical_and(pred_bool, gt_bool).sum().item()
    union = torch.logical_or(pred_bool, gt_bool).sum().item()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def load_mask_image(path: Path) -> torch.Tensor:
    """Load mask image as binary tensor (0 or 1)."""
    with Image.open(path) as img:
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        data = torch.from_numpy(np.array(img)).float()
        # Normalize to [0, 1] and threshold at 0.5
        mask = (data / 255.0 > 0.5).float()
    return mask


def load_3d_mask_and_render_2d(mask_3d_path: str, model_path: str, iteration: int, 
                                test_cameras, gaussians, pipeline, background) -> List[torch.Tensor]:
    """Load 3D mask and render 2D masks for all test cameras."""
    # Load 3D mask
    mask_3d = torch.load(mask_3d_path)
    if isinstance(mask_3d, torch.Tensor):
        if mask_3d.dtype != torch.bool:
            mask_3d = mask_3d > 0.5
    else:
        raise ValueError(f"Unsupported mask format in {mask_3d_path}")
    
    # Apply mask to gaussians
    gaussians.segment(mask_3d)
    
    # Render masks for all test cameras
    masks_2d = []
    for cam in tqdm(test_cameras, desc="Rendering 2D masks from 3D mask"):
        mask_res = render_mask(cam, gaussians, pipeline, background, precomputed_mask=mask_3d)
        mask = mask_res["mask"]
        mask = (mask > 0.5).float()
        masks_2d.append(mask[0, :, :])  # Remove batch dimension
    
    return masks_2d


def compute_iou_from_directories(pred_dir: Path, gt_dir: Path, verbose: bool = False) -> Tuple[float, float]:
    """Compute IoU from directories of mask images."""
    pred_files = sorted(pred_dir.glob("*.png"))
    if len(pred_files) == 0:
        raise RuntimeError(f"No mask images found in {pred_dir}")
    
    iou_values = []
    
    for pred_path in tqdm(pred_files, desc="Computing IoU"):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            if verbose:
                print(f"Warning: GT mask not found for {pred_path.name}, skipping.")
            continue
        
        pred_mask = load_mask_image(pred_path)
        gt_mask = load_mask_image(gt_path)
        
        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            # Resize to match
            from torch.nn.functional import interpolate
            h, w = gt_mask.shape
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            pred_mask = interpolate(pred_mask, size=(h, w), mode='nearest')
            pred_mask = pred_mask.squeeze()
        
        iou = mask_iou(pred_mask, gt_mask)
        iou_values.append(iou)
        
        if verbose:
            print(f"{pred_path.name}: IoU={iou:.4f}")
    
    if len(iou_values) == 0:
        raise RuntimeError("No valid mask pairs found!")
    
    mean_iou = np.mean(iou_values)
    std_iou = np.std(iou_values)
    
    return mean_iou, std_iou


def compute_iou_from_3d_mask(mask_3d_path: str, gt_mask_dir: Path, model_path: str, 
                              iteration: int, source_path: str) -> Tuple[float, float]:
    """Compute IoU from 3D mask (saved from GUI) and GT masks."""
    # Load model
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args(["--model_path", model_path, "-s", source_path, "--iteration", str(iteration)])
    
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval')
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        raise RuntimeError("No test cameras found!")
    
    # Render 2D masks from 3D mask
    masks_2d = load_3d_mask_and_render_2d(
        mask_3d_path, model_path, iteration, test_cameras, gaussians, pipeline, background
    )
    
    # Load GT masks and compute IoU
    gt_files = sorted(gt_mask_dir.glob("*.png"))
    if len(gt_files) != len(masks_2d):
        print(f"Warning: Number of GT masks ({len(gt_files)}) != number of rendered masks ({len(masks_2d)})")
        min_len = min(len(gt_files), len(masks_2d))
        gt_files = gt_files[:min_len]
        masks_2d = masks_2d[:min_len]
    
    iou_values = []
    for idx, gt_path in enumerate(tqdm(gt_files, desc="Computing IoU")):
        pred_mask = masks_2d[idx]
        gt_mask = load_mask_image(gt_path)
        
        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            from torch.nn.functional import interpolate
            h, w = gt_mask.shape
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
            pred_mask = interpolate(pred_mask, size=(h, w), mode='nearest')
            pred_mask = pred_mask.squeeze()
        
        iou = mask_iou(pred_mask, gt_mask)
        iou_values.append(iou)
    
    mean_iou = np.mean(iou_values)
    std_iou = np.std(iou_values)
    
    return mean_iou, std_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IoU for segmentation masks.")
    
    # Option 1: Directories
    parser.add_argument("--pred_mask_dir", type=str, default=None,
                        help="Directory containing predicted mask images.")
    parser.add_argument("--gt_mask_dir", type=str, required=True,
                        help="Directory containing ground-truth mask images.")
    
    # Option 2: 3D mask
    parser.add_argument("--pred_mask_3d", type=str, default=None,
                        help="Path to 3D mask .pt file (from GUI save).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model (required if using --pred_mask_3d).")
    parser.add_argument("--source_path", type=str, default=None,
                        help="Path to scene data (required if using --pred_mask_3d).")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Model iteration (required if using --pred_mask_3d).")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-image IoU values.")
    
    args = parser.parse_args()
    
    if args.pred_mask_3d is not None:
        if args.model_path is None or args.source_path is None:
            parser.error("--model_path and --source_path are required when using --pred_mask_3d")
    elif args.pred_mask_dir is None:
        parser.error("Either --pred_mask_dir or --pred_mask_3d must be provided")
    
    return args


def main():
    args = parse_args()
    gt_mask_dir = Path(args.gt_mask_dir)
    
    if not gt_mask_dir.is_dir():
        raise FileNotFoundError(f"GT mask directory not found: {gt_mask_dir}")
    
    if args.pred_mask_3d is not None:
        # Compute from 3D mask
        mean_iou, std_iou = compute_iou_from_3d_mask(
            args.pred_mask_3d, gt_mask_dir, args.model_path, 
            args.iteration, args.source_path
        )
    else:
        # Compute from directories
        pred_mask_dir = Path(args.pred_mask_dir)
        if not pred_mask_dir.is_dir():
            raise FileNotFoundError(f"Predicted mask directory not found: {pred_mask_dir}")
        
        mean_iou, std_iou = compute_iou_from_directories(
            pred_mask_dir, gt_mask_dir, args.verbose
        )
    
    print("\n=== IoU Summary ===")
    print(f"GT mask dir  : {gt_mask_dir}")
    if args.pred_mask_3d:
        print(f"Pred mask 3D : {args.pred_mask_3d}")
    else:
        print(f"Pred mask dir : {args.pred_mask_dir}")
    print(f"Mean IoU     : {mean_iou:.4f} ± {std_iou:.4f}")


if __name__ == "__main__":
    main()





