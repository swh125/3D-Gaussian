#!/usr/bin/env python3
"""
批量计算多个mask的IoU
"""

import subprocess
import sys
from pathlib import Path


def main():
    # 配置路径
    script_dir = Path(__file__).parent
    calculate_iou_script = script_dir / "calculate_iou.py"
    
    # 对应关系：optimized_mask -> gt_json
    # 去掉 00015/311，加上 00007/303
    pairs = [
        ("00007.png", "frame_00303.json"),
        ("00023.png", "frame_00319.json"),
        ("00031.png", "frame_00327.json"),
        ("00039.png", "frame_00335.json"),
    ]
    
    # 基础路径（optimized结果在桌面）
    desktop = Path.home() / "Desktop"
    # optimized结果路径（假设解压后或直接在zip中）
    optimized_mask_dir = desktop / "items_optimized_gui_render" / "test" / "ours_30000" / "mask"
    # 如果zip还没解压，先尝试解压
    zip_path = desktop / "items_optimized_gui_render.zip"
    if zip_path.exists() and not optimized_mask_dir.exists():
        import zipfile
        print(f"📦 解压 {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(desktop)
        print(f"✓ 解压完成")
    
    gt_json_dir = desktop
    
    print("=" * 80)
    print("批量计算IoU")
    print("=" * 80)
    print()
    
    results = []
    
    for optimized_mask_name, gt_json_name in pairs:
        optimized_mask_path = optimized_mask_dir / optimized_mask_name
        gt_json_path = gt_json_dir / gt_json_name
        
        print(f"📊 计算: {optimized_mask_name} <-> {gt_json_name}")
        print(f"   Optimized: {optimized_mask_path}")
        print(f"   GT: {gt_json_path}")
        
        if not optimized_mask_path.exists():
            print(f"   ❌ Optimized mask不存在: {optimized_mask_path}")
            results.append((optimized_mask_name, gt_json_name, None, "Mask文件不存在"))
            print()
            continue
        
        if not gt_json_path.exists():
            print(f"   ❌ GT JSON不存在: {gt_json_path}")
            results.append((optimized_mask_name, gt_json_name, None, "JSON文件不存在"))
            print()
            continue
        
        # 运行计算IoU的脚本
        try:
            cmd = [
                sys.executable,
                str(calculate_iou_script),
                "--json_file", str(gt_json_path),
                "--pred_mask", str(optimized_mask_path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir.parent)
            
            if result.returncode == 0:
                # 从输出中提取IoU值
                output = result.stdout
                iou_value = None
                for line in output.split('\n'):
                    if 'IoU:' in line:
                        try:
                            # 提取IoU值，例如 "IoU: 0.8523 (85.23%)"
                            parts = line.split('IoU:')
                            if len(parts) > 1:
                                iou_str = parts[1].strip().split()[0]
                                iou_value = float(iou_str)
                        except:
                            pass
                
                print(output)
                results.append((optimized_mask_name, gt_json_name, iou_value, "成功"))
            else:
                print(f"   ❌ 计算失败:")
                print(result.stderr)
                results.append((optimized_mask_name, gt_json_name, None, f"错误: {result.stderr[:100]}"))
        
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            results.append((optimized_mask_name, gt_json_name, None, f"异常: {str(e)}"))
        
        print()
        print("-" * 80)
        print()
    
    # 汇总结果
    print("=" * 80)
    print("📊 汇总结果 (Optimized)")
    print("=" * 80)
    print(f"{'Optimized Mask':<20} {'GT JSON':<25} {'IoU':<15} {'状态':<20}")
    print("-" * 80)
    
    for optimized_mask_name, gt_json_name, iou_value, status in results:
        if iou_value is not None:
            iou_str = f"{iou_value:.4f} ({iou_value*100:.2f}%)"
        else:
            iou_str = "N/A"
        print(f"{optimized_mask_name:<20} {gt_json_name:<25} {iou_str:<15} {status:<20}")
    
    # 计算平均IoU（只计算成功的）
    successful_ious = [r[2] for r in results if r[2] is not None]
    if successful_ious:
        avg_iou = sum(successful_ious) / len(successful_ious)
        print("-" * 80)
        print(f"平均IoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
        print(f"成功计算: {len(successful_ious)}/{len(results)}")
    else:
        print("-" * 80)
        print("❌ 没有成功计算的结果")


if __name__ == "__main__":
    main()


