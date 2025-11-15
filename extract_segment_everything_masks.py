#!/usr/bin/env python3
import os
from typing import List, Sequence

import cv2
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_checkpoints(primary: str, extras: str) -> List[str]:
    checkpoints = []
    for path in [primary, *(extras.split(',') if extras else [])]:
        path = path.strip()
        if path:
            checkpoints.append(path)
    seen = []
    for ckpt in checkpoints:
        if ckpt not in seen:
            seen.append(ckpt)
    return seen


def ensure_kernel_size(k: int) -> int:
    if k <= 1:
        return 0
    if k % 2 == 0:
        k += 1
    return max(1, k)


def morphological_close(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = ensure_kernel_size(kernel_size)
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    mask_f = mask.unsqueeze(0).unsqueeze(0).float()
    dilated = torch.nn.functional.conv2d(mask_f, kernel, padding=pad) > 0
    eroded = torch.nn.functional.conv2d(dilated.float(), kernel, padding=pad) == (kernel_size * kernel_size)
    return eroded[0, 0].bool()


def remove_small_components(mask: torch.Tensor, min_pixels: int) -> torch.Tensor:
    if min_pixels <= 0:
        return mask
    mask_np = mask.cpu().numpy().astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_np)
    if num_labels <= 1:
        return mask
    keep = np.zeros_like(mask_np, dtype=np.uint8)
    for label in range(1, num_labels):
        size = int((labels == label).sum())
        if size >= min_pixels:
            keep[labels == label] = 1
    return torch.from_numpy(keep).to(mask.device, dtype=torch.bool)


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    if union == 0:
        return 0.0
    return inter / union


def cleanup_masks(masks: Sequence[dict], image_hw: Sequence[int], *, device: torch.device,
                  min_pred_iou: float, min_stability: float, min_pixels: int,
                  min_ratio: float, closing_kernel: int, min_component_pixels: int,
                  dedup_iou_thresh: float, max_masks: int) -> List[torch.Tensor]:
    H, W = image_hw
    total_pixels = H * W
    filtered: List[torch.Tensor] = []

    masks_sorted = sorted(masks, key=lambda m: m.get('predicted_iou', 0.0), reverse=True)
    for mask_dict in masks_sorted:
        pred_iou = mask_dict.get('predicted_iou', 0.0)
        stability = mask_dict.get('stability_score', 1.0)
        if pred_iou < min_pred_iou or stability < min_stability:
            continue

        raw_mask = mask_dict['segmentation']
        if isinstance(raw_mask, torch.Tensor):
            mask = raw_mask.to(device=device, dtype=torch.bool)
        else:
            mask = torch.from_numpy(raw_mask).to(device=device, dtype=torch.bool)

        if closing_kernel > 1:
            mask = morphological_close(mask, closing_kernel)

        if min_component_pixels > 0:
            mask = remove_small_components(mask, min_component_pixels)

        area = int(mask.sum().item())
        if area == 0:
            continue
        if min_pixels > 0 and area < min_pixels:
            continue
        if min_ratio > 0.0 and (area / total_pixels) < min_ratio:
            continue

        duplicate = False
        for kept in filtered:
            if mask_iou(mask, kept) >= dedup_iou_thresh:
                duplicate = True
                break
        if duplicate:
            continue

        filtered.append(mask)
        if max_masks > 0 and len(filtered) >= max_masks:
            break

    return filtered


if __name__ == "__main__":
    parser = ArgumentParser(description="SAM segment-everything mask extraction with filtering")
    parser.add_argument("--image_root", type=str, default="/datasets/nerf_data/360_v2/garden/")
    parser.add_argument("--sam_checkpoint_path", type=str, default="./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--extra_checkpoints", type=str, default="", help="Comma separated extra SAM checkpoints")
    parser.add_argument("--sam_arch", type=str, default="vit_h")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--downsample_type", type=str, choices=["image", "mask"], default="image")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)
    parser.add_argument("--min_mask_region_area", type=int, default=100)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=1)
    parser.add_argument("--min_mask_pixels", type=int, default=200, help="Discard masks smaller than this many pixels")
    parser.add_argument("--min_mask_ratio", type=float, default=0.0, help="Discard masks with area ratio below this threshold")
    parser.add_argument("--min_predicted_iou", type=float, default=0.90)
    parser.add_argument("--min_stability_score", type=float, default=0.90)
    parser.add_argument("--min_component_pixels", type=int, default=150, help="Remove connected components smaller than this from each mask")
    parser.add_argument("--closing_kernel", type=int, default=3, help="Odd kernel size for morphological closing (<=1 disables)")
    parser.add_argument("--dedup_iou_thresh", type=float, default=0.95, help="IoU threshold to treat masks as duplicates")
    parser.add_argument("--max_masks_per_image", type=int, default=0, help="Keep at most this many masks per image (0 = no limit)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    checkpoints = parse_checkpoints(args.sam_checkpoint_path, args.extra_checkpoints)
    if not checkpoints:
        raise ValueError("No SAM checkpoints provided")

    device = torch.device(args.device)
    print("Initializing SAM models...")
    generators: List[SamAutomaticMaskGenerator] = []
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
        print(f"  Loading checkpoint: {ckpt}")
        sam_model = sam_model_registry[args.sam_arch](checkpoint=ckpt).to(device)
        generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_layers=args.crop_n_layers,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            min_mask_region_area=args.min_mask_region_area,
        )
        generators.append(generator)

    downsample_manually = False
    if args.downsample == 1 or args.downsample_type == "mask":
        image_dir = os.path.join(args.image_root, "images")
    else:
        image_dir = os.path.join(args.image_root, f"images_{args.downsample}")
        if not os.path.exists(image_dir):
            image_dir = os.path.join(args.image_root, "images")
            downsample_manually = True
            print("No pre-downsampled images found; resizing on the fly.")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = os.path.join(args.image_root, "sam_masks")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting SAM masks with filtering...")
    
    # IMPORTANT: Sort images by filename to maintain temporal order from original video
    # This ensures masks are extracted in the same order as the original video
    # The order must match the order used in dataset_readers.py for consistency
    image_names = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
    print(f"[mask extraction] Processing {len(image_names)} images in temporal order (sorted by filename)")
    
    for filename in tqdm(image_names):
        stem = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is None:
            print(f"Warning: failed to read image {filename}, skip.")
            continue

        if args.downsample_type == "image" and args.downsample > 1 and downsample_manually:
            img = cv2.resize(
                img,
                dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),
                interpolation=cv2.INTER_LINEAR,
            )
        original_hw = (img.shape[0], img.shape[1])

        raw_masks = []
        for generator in generators:
            raw_masks.extend(generator.generate(img))

        processed_masks = cleanup_masks(
            raw_masks,
            original_hw,
            device=device,
            min_pred_iou=args.min_predicted_iou,
            min_stability=args.min_stability_score,
            min_pixels=args.min_mask_pixels,
            min_ratio=args.min_mask_ratio,
            closing_kernel=args.closing_kernel,
            min_component_pixels=args.min_component_pixels,
            dedup_iou_thresh=args.dedup_iou_thresh,
            max_masks=args.max_masks_per_image,
        )

        if not processed_masks:
            torch.save(torch.zeros((0, *original_hw), dtype=torch.bool), os.path.join(output_dir, f"{stem}.pt"))
                continue

        masks_tensor = torch.stack([m.cpu() for m in processed_masks], dim=0)

        if args.downsample_type == "mask" and args.downsample > 1:
            masks_tensor = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(1).float(),
                size=(original_hw[0] // args.downsample, original_hw[1] // args.downsample),
                mode="bilinear",
                align_corners=False,
            ) > 0.5
            masks_tensor = masks_tensor.squeeze(1)

        torch.save(masks_tensor.bool(), os.path.join(output_dir, f"{stem}.pt"))