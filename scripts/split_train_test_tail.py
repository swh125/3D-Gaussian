#!/usr/bin/env python3
"""
Move the last N rendered frames from train -> test so that metrics can be
computed on a less similar subset.
"""

import argparse
import shutil
from pathlib import Path


MIN_TRAIN_FRAMES = 4


def move_tail(src_dir: Path, dst_dir: Path, count: int) -> int:
    files = sorted(src_dir.glob("*.png"))
    total = len(files)
    if count <= 0 or total == 0:
        return 0
    if total <= MIN_TRAIN_FRAMES:
        print(f"[split] Skip {src_dir} -> only {total} frame(s); need > {MIN_TRAIN_FRAMES} to create a test split.")
        return 0

    effective = min(count, max(0, total - MIN_TRAIN_FRAMES))
    if effective == 0:
        print(f"[split] Skip {src_dir} -> leaving at least {MIN_TRAIN_FRAMES} frames for training.")
        return 0

    tail = files[-effective:]
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for file in tail:
        shutil.move(str(file), dst_dir / file.name)
        moved += 1
    return moved


def main():
    parser = argparse.ArgumentParser(description="Move tail of renders/gt to test split.")
    parser.add_argument("--model_path", type=Path, required=True, help="Model directory (contains train/ours_xxx).")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration tag (default: 30000).")
    parser.add_argument("--test_last", type=int, default=40, help="Number of tail frames to move to test split.")
    args = parser.parse_args()

    iteration_tag = f"ours_{args.iteration}"
    train_root = args.model_path / "train" / iteration_tag
    test_root = args.model_path / "test" / iteration_tag

    if not train_root.exists():
        raise FileNotFoundError(f"Train renders directory not found: {train_root}")

    total_moved = 0
    for folder in ["renders", "gt"]:
        src = train_root / folder
        if not src.exists():
            continue
        dst = test_root / folder
        moved = move_tail(src, dst, args.test_last)
        total_moved = max(total_moved, moved)
        print(f"[split] Moved {moved} files from {src} -> {dst}")

    if total_moved == 0:
        print("[split] Nothing moved; check test_last or renders availability.")
    else:
        print(f"[split] Completed. Test split now contains {total_moved} frames.")


if __name__ == "__main__":
    main()

