#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil


def move_subset(src_dir: Path, dst_dir: Path, interval: int) -> int:
    files = sorted(src_dir.glob("*.png"))
    if not files:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for idx, file in enumerate(files):
        if interval <= 0:
            break
        if idx % interval == 0:
            shutil.move(str(file), dst_dir / file.name)
            moved += 1
    return moved


def main():
    parser = argparse.ArgumentParser(description="Split rendered train views into train/test subsets by interval.")
    parser.add_argument("--model_path", required=True, type=Path, help="Model output directory (contains train/ours_xxx).")
    parser.add_argument("--iteration", default=30000, type=int, help="Iteration tag used in renders folders.")
    parser.add_argument("--interval", default=8, type=int, help="Take every 'interval'-th frame as test.")
    args = parser.parse_args()

    iteration_tag = f"ours_{args.iteration}"
    train_base = args.model_path / "train" / iteration_tag
    test_base = args.model_path / "test" / iteration_tag

    if not train_base.exists():
        raise FileNotFoundError(f"Train renders directory not found: {train_base}")

    for subdir in ["renders", "gt"]:
        src = train_base / subdir
        if not src.exists():
            continue
        dst = test_base / subdir
        moved = move_subset(src, dst, args.interval)
        print(f"[split] Moved {moved} files from {src} -> {dst}")


if __name__ == "__main__":
    main()

