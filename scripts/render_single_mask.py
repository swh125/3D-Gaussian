#!/usr/bin/env python3
"""
Render a single mask image from test set.
Usage:
    python scripts/render_single_mask.py \
        -m <model_path> \
        --precomputed_mask problem.pt \
        --test_idx 14 \
        --iteration 30000
"""

import sys
import os
# 添加项目根目录到路径（必须在导入其他模块之前）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from scene import Scene, GaussianModel, FeatureGaussianModel
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_mask
import torchvision
import cv2
import numpy as np

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def render_single_mask(dataset, pipeline, test_idx, iteration, precomputed_mask_path, apply_morphology=True, opening_kernel=2, closing_kernel=3):
    """
    Render a single mask image from test set.
    
    Args:
        dataset: ModelParams object
        pipeline: PipelineParams object
        test_idx: Index in test set (0-based, so 14 means the 15th image)
        iteration: Model iteration to load
        precomputed_mask_path: Path to the precomputed 3D mask (.pt file)
        apply_morphology: Whether to apply morphological operations
        opening_kernel: Kernel size for opening operation
        closing_kernel: Kernel size for closing operation
    """
    # Load precomputed mask
    # Try to find the mask file if it's a relative path
    if not os.path.isabs(precomputed_mask_path) and not os.path.exists(precomputed_mask_path):
        # Try common locations
        possible_paths = [
            precomputed_mask_path,  # Original path
            os.path.join("./segmentation_res", precomputed_mask_path),  # GUI default location
            os.path.join(dataset.model_path, precomputed_mask_path),  # Model directory
            os.path.join(os.path.dirname(dataset.model_path), "segmentation_res", precomputed_mask_path),  # Parent/segmentation_res
        ]
        for path in possible_paths:
            if os.path.exists(path):
                precomputed_mask_path = path
                break
        else:
            # If still not found, try with just the filename in segmentation_res
            if not os.path.dirname(precomputed_mask_path):
                seg_res_path = os.path.join("./segmentation_res", precomputed_mask_path)
                if os.path.exists(seg_res_path):
                    precomputed_mask_path = seg_res_path
    
    if not os.path.exists(precomputed_mask_path):
        print(f"Error: Mask file not found: {precomputed_mask_path}")
        print(f"Tried paths:")
        print(f"  - {precomputed_mask_path}")
        print(f"  - ./segmentation_res/{os.path.basename(precomputed_mask_path)}")
        print(f"  - {os.path.join(dataset.model_path, os.path.basename(precomputed_mask_path))}")
        return None
    
    print(f"Loading precomputed mask from: {precomputed_mask_path}")
    precomputed_mask = torch.load(precomputed_mask_path)
    
    # Convert to float if needed (for render_mask)
    if isinstance(precomputed_mask, torch.Tensor) and precomputed_mask.dtype == torch.bool:
        mask_for_render = precomputed_mask.float()
    else:
        mask_for_render = precomputed_mask.float() if isinstance(precomputed_mask, torch.Tensor) else precomputed_mask
    
    # Move to GPU
    if isinstance(mask_for_render, torch.Tensor):
        mask_for_render = mask_for_render.to(device="cuda")
    
    # Load model
    print(f"Loading model from: {dataset.model_path}")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, None, load_iteration=iteration, shuffle=False, mode='eval', target='scene')
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    
    if test_idx >= len(test_cameras):
        print(f"Error: test_idx {test_idx} is out of range. Test set has {len(test_cameras)} images.")
        return
    
    # Get the specific view
    view = test_cameras[test_idx]
    print(f"Rendering mask for test image {test_idx} (the {test_idx + 1}th image in test set)")
    
    # Setup background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Render mask
    mask_res = render_mask(view, gaussians, pipeline, background, precomputed_mask=mask_for_render)
    mask = mask_res["mask"]
    mask[mask < 0.5] = 0
    mask[mask != 0] = 1
    mask = mask[0, :, :].detach().cpu().numpy()
    
    # Apply morphological operations if requested
    if apply_morphology:
        try:
            # Convert to uint8 for OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Fill holes
            mask_filled = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            
            # Opening: remove small noise and halos
            if opening_kernel > 0:
                mask_opened = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel)))
            else:
                mask_opened = mask_filled
            
            # Closing: smooth edges and remove jaggedness
            if closing_kernel > 0:
                mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)))
            else:
                mask_closed = mask_opened
            
            # Convert back to binary
            mask = (mask_closed > 127).astype(np.float32)
        except Exception as e:
            print(f"Warning: Morphological operations failed: {e}")
            print("Using original mask without morphology")
    
    # Create output directory
    output_dir = os.path.join(model_path, "test", f"ours_{iteration}", "mask_single")
    makedirs(output_dir, exist_ok=True)
    
    # Save mask
    output_path = os.path.join(output_dir, f"test_{test_idx:05d}.png")
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    torchvision.utils.save_image(mask_tensor, output_path)
    
    print(f"✓ Mask saved to: {output_path}")
    print(f"  Test index: {test_idx} (the {test_idx + 1}th image in test set)")
    return output_path

if __name__ == "__main__":
    parser = ArgumentParser(description="Render a single mask image")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_idx", type=int, required=True, help="Index in test set (0-based, so 14 means the 15th image)")
    parser.add_argument("--precomputed_mask", type=str, required=True, help="Path to precomputed 3D mask (.pt file)")
    parser.add_argument("--no_morphology", action="store_true", help="Disable morphological operations")
    parser.add_argument("--opening_kernel", type=int, default=2, help="Kernel size for opening operation")
    parser.add_argument("--closing_kernel", type=int, default=3, help="Kernel size for closing operation")
    
    # Use get_combined_args with target_cfg_file to avoid target attribute error
    args = get_combined_args(parser, target_cfg_file="cfg_args")
    
    print("=" * 60)
    print("Rendering Single Mask")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Precomputed mask: {args.precomputed_mask}")
    print(f"Test index: {args.test_idx} (the {args.test_idx + 1}th image)")
    print(f"Iteration: {args.iteration}")
    print(f"Morphology: {'Disabled' if args.no_morphology else 'Enabled'}")
    if not args.no_morphology:
        print(f"  Opening kernel: {args.opening_kernel}")
        print(f"  Closing kernel: {args.closing_kernel}")
    print("=" * 60)
    print()
    
    safe_state(False)
    
    apply_morphology = not args.no_morphology
    output_path = render_single_mask(
        model.extract(args),
        pipeline.extract(args),
        args.test_idx,
        args.iteration,
        args.precomputed_mask,
        apply_morphology=apply_morphology,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel
    )
    
    if output_path:
        print()
        print("=" * 60)
        print("✓ Rendering complete!")
        print(f"Output: {output_path}")
        print("=" * 60)

