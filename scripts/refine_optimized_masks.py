#!/usr/bin/env python3
"""
处理optimized zip中的renders和masks，去除光晕
使用更强的形态学操作处理mask，然后用mask清理renders边缘的光晕
"""

import sys
sys.path.insert(0, '.')

import zipfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def refine_mask_strong(mask_path: Path, opening_kernel: int = 3, closing_kernel: int = 5, 
                       iterations: int = 2):
    """
    使用更强的形态学操作去除光晕
    
    Args:
        mask_path: mask文件路径
        opening_kernel: 开运算kernel大小（去除小光晕）
        closing_kernel: 闭运算kernel大小（平滑边缘）
        iterations: 形态学操作迭代次数
    """
    # 加载mask
    mask_img = Image.open(mask_path)
    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')
    mask_np = np.array(mask_img)
    
    # 二值化（使用更严格的阈值去除半透明区域）
    mask_binary = (mask_np >= 128).astype(np.uint8) * 255
    
    # 多次开运算去除光晕（去除小的噪声和光晕）
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        for _ in range(iterations):
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
    
    # 多次闭运算平滑边缘（连接断开的区域，平滑边缘）
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        for _ in range(iterations):
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close)
    
    # 填充内部空洞
    from scipy.ndimage import binary_fill_holes
    mask_binary = binary_fill_holes(mask_binary > 127).astype(np.uint8) * 255
    
    # 再次闭运算确保边缘平滑
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close)
    
    return mask_binary


def refine_render_with_mask(render_path: Path, mask_binary: np.ndarray):
    """
    使用处理后的mask清理render边缘的光晕
    
    Args:
        render_path: render文件路径
        mask_binary: 处理后的二值mask (H, W), 0或255
    """
    # 加载render
    render_img = Image.open(render_path)
    if render_img.mode != 'RGB':
        render_img = render_img.convert('RGB')
    render_np = np.array(render_img)
    
    # 检查尺寸是否匹配
    if render_np.shape[:2] != mask_binary.shape:
        # 调整mask尺寸以匹配render
        mask_img = Image.fromarray(mask_binary)
        mask_img = mask_img.resize((render_np.shape[1], render_np.shape[0]), Image.NEAREST)
        mask_binary = np.array(mask_img)
    
    # 将mask转换为0-1的布尔mask
    mask_bool = (mask_binary > 127).astype(bool)
    
    # 在mask区域外（光晕区域）设为黑色
    render_cleaned = render_np.copy()
    render_cleaned[~mask_bool] = [0, 0, 0]
    
    return render_cleaned


def process_zip_masks(zip_path: Path, output_zip_path: Path = None,
                     opening_kernel: int = 3, closing_kernel: int = 5, iterations: int = 2):
    """
    处理zip文件中的mask
    
    Args:
        zip_path: 原始zip文件路径
        output_zip_path: 输出zip文件路径（如果为None，覆盖原文件）
        opening_kernel: 开运算kernel大小
        closing_kernel: 闭运算kernel大小
        iterations: 形态学操作迭代次数
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"❌ Zip文件不存在: {zip_path}")
        return
    
    if output_zip_path is None:
        output_zip_path = zip_path
    else:
        output_zip_path = Path(output_zip_path)
    
    # 创建临时目录
    temp_dir = Path(zip_path.parent) / f"{zip_path.stem}_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 解压zip
        print(f"📦 解压 {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"✓ 解压完成")
        
        # 查找所有mask目录
        mask_dirs = list(temp_dir.rglob("mask"))
        if not mask_dirs:
            print(f"⚠️  未找到mask目录")
            return
        
        print(f"\n🔍 找到 {len(mask_dirs)} 个mask目录")
        
        # 处理每个mask目录，同时处理对应的renders
        total_masks = 0
        total_renders = 0
        for mask_dir in mask_dirs:
            mask_files = sorted(mask_dir.glob("*.png"))
            if len(mask_files) == 0:
                continue
            
            # 找到对应的renders目录
            renders_dir = mask_dir.parent / "renders"
            if not renders_dir.exists():
                print(f"\n📁 处理目录: {mask_dir}")
                print(f"   ⚠️  未找到对应的renders目录: {renders_dir}")
                # 只处理masks
                for mask_path in tqdm(mask_files, desc="  处理masks"):
                    try:
                        mask_refined = refine_mask_strong(
                            mask_path, 
                            opening_kernel=opening_kernel,
                            closing_kernel=closing_kernel,
                            iterations=iterations
                        )
                        Image.fromarray(mask_refined).save(mask_path)
                        total_masks += 1
                    except Exception as e:
                        print(f"   ⚠️  处理失败 {mask_path.name}: {e}")
                continue
            
            print(f"\n📁 处理目录: {mask_dir}")
            print(f"   找到 {len(mask_files)} 个mask文件")
            print(f"   对应的renders目录: {renders_dir}")
            
            # 同时处理masks和renders
            for mask_path in tqdm(mask_files, desc="  处理masks和renders"):
                try:
                    # 处理mask
                    mask_refined = refine_mask_strong(
                        mask_path, 
                        opening_kernel=opening_kernel,
                        closing_kernel=closing_kernel,
                        iterations=iterations
                    )
                    Image.fromarray(mask_refined).save(mask_path)
                    total_masks += 1
                    
                    # 处理对应的render
                    render_path = renders_dir / mask_path.name
                    if render_path.exists():
                        render_cleaned = refine_render_with_mask(render_path, mask_refined)
                        Image.fromarray(render_cleaned).save(render_path)
                        total_renders += 1
                except Exception as e:
                    print(f"   ⚠️  处理失败 {mask_path.name}: {e}")
        
        # 重新打包
        print(f"\n📦 重新打包...")
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for file_path in tqdm(temp_dir.rglob("*"), desc="  打包文件"):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zip_out.write(file_path, arcname)
        
        print(f"\n✓ 完成！")
        print(f"   处理了 {total_masks} 个mask文件")
        print(f"   处理了 {total_renders} 个render文件")
        print(f"   输出文件: {output_zip_path}")
        
    finally:
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"✓ 清理临时文件")


def main():
    parser = argparse.ArgumentParser(description="处理optimized zip中的renders和masks，去除光晕")
    parser.add_argument("--zip_path", type=str, required=True,
                       help="原始zip文件路径（如 ~/Desktop/items_optimized_gui_render.zip）")
    parser.add_argument("--output_zip", type=str, default=None,
                       help="输出zip文件路径（如果为None，覆盖原文件）")
    parser.add_argument("--opening_kernel", type=int, default=3,
                       help="开运算kernel大小，用于去除光晕（默认3，越大去除越多）")
    parser.add_argument("--closing_kernel", type=int, default=5,
                       help="闭运算kernel大小，用于平滑边缘（默认5，越大越平滑）")
    parser.add_argument("--iterations", type=int, default=2,
                       help="形态学操作迭代次数（默认2，越多效果越强）")
    
    args = parser.parse_args()
    
    zip_path = Path(args.zip_path).expanduser()
    output_zip = Path(args.output_zip).expanduser() if args.output_zip else None
    
    process_zip_masks(
        zip_path=zip_path,
        output_zip_path=output_zip,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel,
        iterations=args.iterations
    )


if __name__ == "__main__":
    main()

