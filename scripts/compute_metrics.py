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
from pathlib import Path
from typing import List, Tuple

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


def load_image_pair(render_path: Path, gt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    return _load_image(render_path), _load_image(gt_path)


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
        raise RuntimeError(f"No render images (*.{args.ext}) found in {render_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lpips_model = None
    if _HAS_LPIPS:
        lpips_model = LPIPS(net="alex").to(device)
        lpips_model.eval()
    else:
        print("Warning: LPIPS module not available, skipping LPIPS metric.")

    psnr_values, ssim_values, lpips_values = [], [], []

    for render_path in tqdm(render_files, desc="Computing metrics"):
        gt_path = gt_dir / render_path.name
        if not gt_path.is_file():
            print(f"Warning: Ground-truth image not found for {render_path.name}, skipping.")
            continue

        render_tensor, gt_tensor = load_image_pair(render_path, gt_path)
        render_tensor = render_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        psnr_val = psnr_fn(render_tensor, gt_tensor).mean().item()
        ssim_val = ssim_fn(render_tensor, gt_tensor).item()
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

        if lpips_model is not None:
            lpips_val = lpips_model(render_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).mean().item()
            lpips_values.append(lpips_val)

        if args.verbose:
            lpips_str = f"{lpips_values[-1]:.6f}" if lpips_model is not None else "N/A"
            print(f"{render_path.name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_str}")

    psnr_mean, psnr_std = summarize(psnr_values)
    ssim_mean, ssim_std = summarize(ssim_values)
    lpips_mean, lpips_std = summarize(lpips_values) if lpips_values else (float("nan"), float("nan"))

    print("\n=== Metrics Summary ===")
    print(f"Render dir : {render_dir}")
    print(f"GT dir     : {gt_dir}")
    print(f"Samples    : {len(psnr_values)}")
    print(f"PSNR       : {psnr_mean:.4f} ± {psnr_std:.4f}")
    print(f"SSIM       : {ssim_mean:.4f} ± {ssim_std:.4f}")
    if _HAS_LPIPS:
        print(f"LPIPS      : {lpips_mean:.4f} ± {lpips_std:.4f}")
    else:
        print("LPIPS      : (skipped – lpips module not available)")


if __name__ == "__main__":
    args = parse_args()
    compute_metrics(args)

