#!/usr/bin/env python
"""
Generate Ground Truth (GT) masks for test set images using SAM.

This script uses SAM to generate high-quality masks that can serve as ground truth
for IoU evaluation. You can also manually refine these masks if needed.

Usage:
    python scripts/generate_gt_masks.py \
        --image_dir /home/bygpu/data/video_scene/images \
        --output_dir /home/bygpu/data/video_scene/gt_masks/test \
        --sam_checkpoint_path /path/to/sam_vit_h.pth \
        --test_indices 295-334  # Last 40 frames (or adjust based on your split)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except ImportError:
    _HAS_SAM = False
    print("Warning: segment_anything not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


def parse_indices(indices_str: str) -> List[int]:
    """Parse index range string like '295-334' or '0,1,2,3'."""
    if '-' in indices_str:
        start, end = map(int, indices_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in indices_str.split(',')]


def generate_masks_for_images(image_dir: Path, output_dir: Path, sam_checkpoint: str,
                              test_indices: List[int], model_type: str = "vit_h",
                              min_mask_pixels: int = 100) -> None:
    """Generate masks for specified image indices using SAM."""
    if not _HAS_SAM:
        raise ImportError("segment_anything is required. Install it first.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAM model
    print(f"Loading SAM model ({model_type}) from {sam_checkpoint}...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_pixels,
    )
    
    # Get all images
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images. Processing {len(test_indices)} test images...")
    
    for idx in tqdm(test_indices, desc="Generating GT masks"):
        if idx >= len(image_files):
            print(f"Warning: Index {idx} out of range (max: {len(image_files)-1}), skipping.")
            continue
        
        image_path = image_files[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        if len(masks) == 0:
            print(f"Warning: No masks generated for {image_path.name}")
            # Create empty mask
            mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # Combine all masks (union)
            mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for m in masks:
                mask_combined[m['segmentation']] = 255
        
        # Save mask
        mask_name = image_path.stem + "_gt_mask.png"
        mask_path = output_dir / mask_name
        Image.fromarray(mask_combined).save(mask_path)
    
    print(f"\nGenerated {len(test_indices)} GT masks in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate GT masks using SAM.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for GT masks.")
    parser.add_argument("--sam_checkpoint_path", type=str, required=True,
                        help="Path to SAM checkpoint (e.g., sam_vit_h.pth).")
    parser.add_argument("--test_indices", type=str, default="295-334",
                        help="Test image indices (e.g., '295-334' or '0,1,2,3').")
    parser.add_argument("--model_type", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type.")
    parser.add_argument("--min_mask_pixels", type=int, default=100,
                        help="Minimum pixels for a mask to be kept.")
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    test_indices = parse_indices(args.test_indices)
    
    generate_masks_for_images(
        image_dir, output_dir, args.sam_checkpoint_path,
        test_indices, args.model_type, args.min_mask_pixels
    )


if __name__ == "__main__":
    main()





