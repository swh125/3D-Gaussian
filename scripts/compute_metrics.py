#!/usr/bin/env python
"""
Compute reconstruction metrics (PSNR / SSIM / LPIPS) between rendered images and ground-truth views.

Typical usage:
    python scripts/compute_metrics.py \
        --model_path ./output/77e56970-f \
        --set train \
        --iteration 30000

This looks for:
    <model_path>/<set>/ours_<iteration>/renders/*.png
    <model_path>/<set>/ours_<iteration>/gt/*.png

You can also override --render_dir / --gt_dir directly.
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
from PIL import Image
from tqdm import tqdm

from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn

try:
    from lpipsPyTorch.modules.lpips import LPIPS
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR / SSIM / LPIPS for rendered results.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model output directory.")
    parser.add_argument("--set", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to evaluate when using --model_path.")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration tag inside renders/gt folders.")
    parser.add_argument("--render_dir", type=str, default=None, help="Override path to rendered images.")
    parser.add_argument("--gt_dir", type=str, default=None, help="Override path to ground-truth images.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Path to mask directory (for object-level evaluation). If provided, metrics will be computed only in mask regions.")
    parser.add_argument("--device", type=str, default="cuda", help="torch device to use (cuda or cpu).")
    parser.add_argument("--ext", type=str, default="png", help="Image file extension.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs to evaluate.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image metrics.")
    args = parser.parse_args()

    if args.render_dir is None or args.gt_dir is None:
        if args.model_path is None:
            parser.error("Either --model_path or both --render_dir/--gt_dir must be provided.")
        base = Path(args.model_path)
        render_dir = base / args.set / f"ours_{args.iteration}" / "renders"
        gt_dir = base / args.set / f"ours_{args.iteration}" / "gt"
        args.render_dir = str(render_dir)
        args.gt_dir = str(gt_dir)
        
        # Auto-detect mask directory if not provided
        if args.mask_dir is None:
            mask_dir = base / args.set / f"ours_{args.iteration}" / "mask"
            if mask_dir.exists():
                args.mask_dir = str(mask_dir)

    return args


def _load_image(path: Path) -> torch.Tensor:
    """Load image as float tensor in [0, 1] with shape (1, C, H, W)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        data = torch.from_numpy(
            torch.ByteTensor(bytearray(img.tobytes())).view(img.size[1], img.size[0], 3).numpy()
        ).float()
        tensor = data.permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor


def _load_mask(path: Path) -> torch.Tensor:
    """Load mask as float tensor in [0, 1] with shape (1, 1, H, W)."""
    with Image.open(path) as img:
        img = img.convert("L")
        data = torch.from_numpy(
            torch.ByteTensor(bytearray(img.tobytes())).view(img.size[1], img.size[0], 1).numpy()
        ).float()
        tensor = data.permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor


def load_image_pair(render_path: Path, gt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    return _load_image(render_path), _load_image(gt_path)


def psnr_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute PSNR only in mask regions."""
    # mask: (1, 1, H, W), expand to (1, 3, H, W)
    mask_expanded = mask.expand(-1, 3, -1, -1)
    # Only compute MSE in mask regions
    mse = (((img1 - img2) * mask_expanded) ** 2).sum() / (mask_expanded.sum() + 1e-8)
    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
    return psnr


def ssim_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute SSIM only in mask regions."""
    # Apply mask to both images
    mask_expanded = mask.expand(-1, 3, -1, -1)
    img1_masked = img1 * mask_expanded
    img2_masked = img2 * mask_expanded
    
    # Compute SSIM on masked images
    # Note: This is a simplified version. For more accurate masked SSIM, 
    # we would need to modify the SSIM function itself.
    return ssim_fn(img1_masked, img2_masked).item()


def summarize(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    tensor_vals = torch.tensor(values)
    return tensor_vals.mean().item(), tensor_vals.std(unbiased=False).item()


def compute_metrics(args: argparse.Namespace) -> None:
    render_dir = Path(args.render_dir)
    gt_dir = Path(args.gt_dir)

    if not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    render_files: List[Path] = sorted(render_dir.glob(f"*.{args.ext}"))
    if args.limit is not None:
        render_files = render_files[:args.limit]
    if len(render_files) == 0:
        raise RuntimeError(
            f"No render images (*.{args.ext}) found in {render_dir}. "
            "Run render.py to export renders/gt before computing metrics."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lpips_model = None
    if _HAS_LPIPS:
        try:
            lpips_model = LPIPS(net="alex").to(device)
        except TypeError:
            lpips_model = LPIPS().to(device)
        lpips_model.eval()
    else:
        print("Warning: LPIPS module not available, skipping LPIPS metric.")

    # Check if mask directory is provided
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    use_mask = mask_dir is not None and mask_dir.exists()
    
    if use_mask:
        print(f"✓ Using mask-based evaluation (object-level): {mask_dir}")
    else:
        print("⚠️  Computing metrics on full images (no mask provided)")

    psnr_values, ssim_values, lpips_values = [], [], []

    for render_path in tqdm(render_files, desc="Computing metrics"):
        gt_path = gt_dir / render_path.name
        if not gt_path.is_file():
            print(f"Warning: Ground-truth image not found for {render_path.name}, skipping.")
            continue

        render_tensor, gt_tensor = load_image_pair(render_path, gt_path)
        render_tensor = render_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        if use_mask:
            # Load mask
            mask_path = mask_dir / render_path.name
            if not mask_path.is_file():
                print(f"Warning: Mask not found for {render_path.name}, skipping.")
                continue
            mask_tensor = _load_mask(mask_path).to(device)
            # Threshold mask
            mask_binary = (mask_tensor > 0.5).float()
            
            # Check if mask has valid pixels
            if mask_binary.sum() < 100:  # Less than 100 pixels
                print(f"Warning: Mask for {render_path.name} has too few pixels, skipping.")
                continue
            
            # Compute metrics only in mask regions
            psnr_val = psnr_masked(render_tensor, gt_tensor, mask_binary).item()
            ssim_val = ssim_masked(render_tensor, gt_tensor, mask_binary)
            
            if lpips_model is not None:
                # For LPIPS, apply mask to both images
                mask_expanded = mask_binary.expand(-1, 3, -1, -1)
                render_masked = render_tensor * mask_expanded
                gt_masked = gt_tensor * mask_expanded
                lpips_val = lpips_model(render_masked * 2.0 - 1.0, gt_masked * 2.0 - 1.0).mean().item()
            else:
                lpips_val = None
        else:
            # Compute metrics on full images
            psnr_val = psnr_fn(render_tensor, gt_tensor).mean().item()
            ssim_val = ssim_fn(render_tensor, gt_tensor).item()
            
            if lpips_model is not None:
                lpips_val = lpips_model(render_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).mean().item()
            else:
                lpips_val = None

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        if lpips_val is not None:
            lpips_values.append(lpips_val)

        if args.verbose:
            lpips_str = f"{lpips_values[-1]:.6f}" if lpips_model is not None and lpips_val is not None else "N/A"
            print(f"{render_path.name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_str}")

    psnr_mean, psnr_std = summarize(psnr_values)
    ssim_mean, ssim_std = summarize(ssim_values)
    lpips_mean, lpips_std = summarize(lpips_values) if lpips_values else (float("nan"), float("nan"))

    print("\n=== Metrics Summary ===")
    if use_mask:
        print(f"Evaluation: Object-level (mask-based)")
        print(f"Mask dir   : {mask_dir}")
    else:
        print(f"Evaluation: Full image")
    print(f"Render dir : {render_dir}")
    print(f"GT dir     : {gt_dir}")
    print(f"Samples    : {len(psnr_values)}")
    print(f"PSNR       : {psnr_mean:.4f} ± {psnr_std:.4f}")
    print(f"SSIM       : {ssim_mean:.4f} ± {ssim_std:.4f}")
    if _HAS_LPIPS:
        if lpips_values:
            print(f"LPIPS      : {lpips_mean:.4f} ± {lpips_std:.4f}")
        else:
            print("LPIPS      : (no valid samples)")
    else:
        print("LPIPS      : (skipped – lpips module not available)")


if __name__ == "__main__":
    args = parse_args()
    compute_metrics(args)

