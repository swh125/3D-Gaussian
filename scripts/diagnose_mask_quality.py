#!/usr/bin/env python3
"""
诊断mask质量问题
检查mask的连通性、碎片化程度等
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from scipy import ndimage
from skimage import measure


def load_mask(mask_path: Path):
    """加载mask图片"""
    img = Image.open(mask_path)
    if img.mode != 'L':
        img = img.convert('L')
    mask = np.array(img)
    if mask.max() > 1:
        mask = mask / 255.0
    return (mask >= 0.5).astype(np.uint8)


def analyze_mask_quality(mask: np.ndarray, mask_name: str):
    """分析mask质量"""
    print(f"\n{'='*60}")
    print(f"分析: {mask_name}")
    print(f"{'='*60}")
    
    # 基本统计
    total_pixels = mask.size
    mask_pixels = mask.sum()
    coverage = mask_pixels / total_pixels * 100
    
    print(f"总像素数: {total_pixels}")
    print(f"Mask像素数: {mask_pixels} ({coverage:.2f}%)")
    
    # 连通域分析
    labeled_mask, num_features = ndimage.label(mask)
    print(f"\n连通域数量: {num_features}")
    
    if num_features > 0:
        # 每个连通域的大小
        component_sizes = []
        for i in range(1, num_features + 1):
            size = (labeled_mask == i).sum()
            component_sizes.append(size)
        
        component_sizes = sorted(component_sizes, reverse=True)
        
        print(f"最大连通域: {component_sizes[0]} 像素 ({component_sizes[0]/mask_pixels*100:.2f}%)")
        
        if len(component_sizes) > 1:
            print(f"第二大连通域: {component_sizes[1]} 像素 ({component_sizes[1]/mask_pixels*100:.2f}%)")
        
        # 碎片化程度：小连通域（<1%总mask面积）的数量
        threshold = mask_pixels * 0.01
        small_components = sum(1 for s in component_sizes if s < threshold)
        print(f"小碎片数量（<1%总mask）: {small_components}")
        
        # 碎片化指标：小碎片占总mask的比例
        small_fragments_area = sum(s for s in component_sizes if s < threshold)
        fragmentation_ratio = small_fragments_area / mask_pixels if mask_pixels > 0 else 0
        print(f"碎片化比例: {fragmentation_ratio:.4f} ({fragmentation_ratio*100:.2f}%)")
        
        # 如果碎片化严重，给出警告
        if fragmentation_ratio > 0.1 or small_components > 10:
            print(f"\n⚠️  警告: Mask碎片化严重！")
            print(f"   建议检查3D mask或重新渲染该帧")
    
    # 边缘平滑度（计算边缘像素比例）
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    edge_pixels = mask.sum() - eroded.sum()
    edge_ratio = edge_pixels / mask_pixels if mask_pixels > 0 else 0
    print(f"\n边缘像素比例: {edge_ratio:.4f}")
    
    # 空洞检测（mask内部的背景区域）
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(mask)
    holes = filled.sum() - mask.sum()
    hole_ratio = holes / mask_pixels if mask_pixels > 0 else 0
    print(f"空洞像素数: {holes} ({hole_ratio*100:.2f}%)")
    
    if hole_ratio > 0.05:
        print(f"⚠️  警告: Mask内部有较多空洞")


def compare_masks(mask_paths, mask_names):
    """比较多个mask的质量"""
    print("\n" + "="*60)
    print("Mask质量对比")
    print("="*60)
    
    results = []
    for mask_path, mask_name in zip(mask_paths, mask_names):
        mask = load_mask(mask_path)
        
        # 计算碎片化指标
        labeled_mask, num_features = ndimage.label(mask)
        mask_pixels = mask.sum()
        
        if num_features > 0:
            component_sizes = []
            for i in range(1, num_features + 1):
                size = (labeled_mask == i).sum()
                component_sizes.append(size)
            
            component_sizes = sorted(component_sizes, reverse=True)
            threshold = mask_pixels * 0.01
            small_components = sum(1 for s in component_sizes if s < threshold)
            small_fragments_area = sum(s for s in component_sizes if s < threshold)
            fragmentation_ratio = small_fragments_area / mask_pixels if mask_pixels > 0 else 0
            
            results.append({
                'name': mask_name,
                'num_components': num_features,
                'fragmentation': fragmentation_ratio,
                'small_components': small_components,
                'coverage': mask_pixels / mask.size * 100
            })
        else:
            results.append({
                'name': mask_name,
                'num_components': 0,
                'fragmentation': 0,
                'small_components': 0,
                'coverage': 0
            })
    
    # 打印对比表格
    print(f"\n{'Mask名称':<20} {'连通域数':<12} {'碎片化比例':<15} {'小碎片数':<12} {'覆盖率':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<20} {r['num_components']:<12} {r['fragmentation']:<15.4f} {r['small_components']:<12} {r['coverage']:<10.2f}%")
    
    # 找出异常mask
    avg_fragmentation = np.mean([r['fragmentation'] for r in results])
    for r in results:
        if r['fragmentation'] > avg_fragmentation * 2 or r['num_components'] > 50:
            print(f"\n⚠️  异常Mask: {r['name']}")
            print(f"   连通域数: {r['num_components']}, 碎片化: {r['fragmentation']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="诊断mask质量")
    parser.add_argument("--mask", type=str, help="单个mask文件路径")
    parser.add_argument("--masks", type=str, nargs='+', help="多个mask文件路径（用于对比）")
    parser.add_argument("--names", type=str, nargs='+', help="mask名称（与--masks对应）")
    
    args = parser.parse_args()
    
    if args.mask:
        # 分析单个mask
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"❌ Mask文件不存在: {mask_path}")
            return
        
        mask = load_mask(mask_path)
        analyze_mask_quality(mask, mask_path.name)
    
    elif args.masks:
        # 对比多个mask
        mask_paths = [Path(p) for p in args.masks]
        mask_names = args.names if args.names else [p.name for p in mask_paths]
        
        if len(mask_names) != len(mask_paths):
            print("❌ --names的数量必须与--masks相同")
            return
        
        for mask_path in mask_paths:
            if not mask_path.exists():
                print(f"❌ Mask文件不存在: {mask_path}")
                return
        
        compare_masks(mask_paths, mask_names)
        
        # 也单独分析每个mask
        for mask_path, mask_name in zip(mask_paths, mask_names):
            mask = load_mask(mask_path)
            analyze_mask_quality(mask, mask_name)
    else:
        print("请提供 --mask 或 --masks 参数")
        parser.print_help()


if __name__ == "__main__":
    main()

