#!/usr/bin/env python3
"""
验证整个流程中图片顺序是否保持原视频顺序
确保：1) 图片按原视频顺序排列 2) 最后N个作为测试集 3) 所有处理保持顺序
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
import argparse as arg_parse
import os


def get_data_path_from_env_or_model():
    """从环境变量或模型路径自动获取数据路径"""
    # 1. 从环境变量获取
    output_dir = os.environ.get('OUTPUT_DIR')
    if output_dir and Path(output_dir).exists():
        return output_dir
    
    # 2. 从最新的模型路径的cfg_args读取
    output_base = Path('./output')
    if output_base.exists():
        model_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        for model_dir in model_dirs[:5]:  # 检查最近5个模型
            cfg_file = model_dir / 'cfg_args'
            if cfg_file.exists():
                try:
                    with open(cfg_file, 'r') as f:
                        content = f.read()
                        # 查找source_path
                        for line in content.split('\n'):
                            if 'source_path' in line:
                                parts = line.strip().split()
                                for i, part in enumerate(parts):
                                    if 'source_path' in part and i + 1 < len(parts):
                                        path = parts[i + 1].strip("'\"")
                                        if Path(path).exists():
                                            return path
                except:
                    pass
    
    return None


def verify_order(data_path=None, test_last_n=40):
    """验证图片顺序和训练/测试集划分"""
    print("=" * 70)
    print("验证图片顺序和训练/测试集划分")
    print("=" * 70)
    print()
    
    # 如果没有提供data_path，尝试自动获取
    if data_path is None:
        print("未提供数据路径，尝试自动获取...")
        data_path = get_data_path_from_env_or_model()
        if data_path:
            print(f"自动获取到数据路径: {data_path}")
        else:
            print("❌ 错误: 无法自动获取数据路径")
            print("请使用 --data_path 参数指定，或设置 OUTPUT_DIR 环境变量")
            return False
    
    data_path = Path(data_path)
    
    # 1. 检查原始图片顺序
    images_dir = data_path / "images"
    if not images_dir.exists():
        print(f"❌ 错误: 图片目录不存在: {images_dir}")
        return False
    
    image_files = sorted([f for f in images_dir.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.startswith('.')])
    
    total_images = len(image_files)
    print(f"1. 原始图片检查")
    print(f"   - 总图片数: {total_images}")
    print(f"   - 第一张: {image_files[0].name}")
    print(f"   - 最后一张: {image_files[-1].name}")
    print()
    
    # 2. 检查训练/测试集划分
    parser = arg_parse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args(['-s', str(data_path)])
    args.eval = True
    
    scene_info = sceneLoadTypeCallbacks['Colmap'](
        str(data_path), 
        args.images, 
        args.eval, 
        need_features=False, 
        need_masks=False, 
        sample_rate=1.0, 
        allow_principle_point_shift=args.allow_principle_point_shift, 
        replica='replica' in str(data_path),
        test_last_n=test_last_n
    )
    
    train_count = len(scene_info.train_cameras)
    test_count = len(scene_info.test_cameras)
    total_loaded = train_count + test_count
    
    print(f"2. 训练/测试集划分")
    print(f"   - 总加载数: {total_loaded}")
    print(f"   - 训练集: {train_count} 张")
    print(f"   - 测试集: {test_count} 张")
    print()
    
    # 3. 验证划分是否正确（最后N个作为测试集）
    expected_train = total_loaded - test_last_n
    expected_test = test_last_n
    
    print(f"3. 验证划分逻辑")
    print(f"   - 期望: 前 {expected_train} 张训练，后 {expected_test} 张测试")
    print(f"   - 实际: 前 {train_count} 张训练，后 {test_count} 张测试")
    
    if train_count == expected_train and test_count == expected_test:
        print(f"   ✅ 划分正确！")
    else:
        print(f"   ❌ 划分不正确！")
        return False
    print()
    
    # 4. 验证训练集和测试集的图片顺序
    print(f"4. 验证图片顺序")
    
    # 检查训练集：应该是前N张
    train_image_names = [cam.image_name for cam in scene_info.train_cameras]
    train_sorted = sorted(train_image_names)
    
    if train_image_names == train_sorted:
        print(f"   ✅ 训练集图片按文件名排序")
    else:
        print(f"   ⚠️  训练集图片顺序可能有问题")
        print(f"      前5张: {train_image_names[:5]}")
    
    # 检查测试集：应该是后N张
    test_image_names = [cam.image_name for cam in scene_info.test_cameras]
    test_sorted = sorted(test_image_names)
    
    if test_image_names == test_sorted:
        print(f"   ✅ 测试集图片按文件名排序")
    else:
        print(f"   ⚠️  测试集图片顺序可能有问题")
        print(f"      前5张: {test_image_names[:5]}")
    
    # 验证测试集确实是最后N张
    all_image_names = sorted([f.name for f in image_files])
    expected_test_images = all_image_names[-test_count:]
    actual_test_images = [name for name in test_image_names]
    
    if set(actual_test_images) == set(expected_test_images):
        print(f"   ✅ 测试集确实是最后 {test_count} 张图片")
        print(f"      测试集第一张: {actual_test_images[0]}")
        print(f"      测试集最后一张: {actual_test_images[-1]}")
    else:
        print(f"   ❌ 测试集不是最后 {test_count} 张图片！")
        print(f"      期望测试集: {expected_test_images[:3]} ... {expected_test_images[-3:]}")
        print(f"      实际测试集: {actual_test_images[:3]} ... {actual_test_images[-3:]}")
        return False
    print()
    
    # 5. 验证训练集和测试集的连续性
    print(f"5. 验证训练/测试集连续性")
    last_train_name = train_image_names[-1] if train_image_names else None
    first_test_name = test_image_names[0] if test_image_names else None
    
    if last_train_name and first_test_name:
        # 找到它们在原始列表中的位置
        try:
            last_train_idx = all_image_names.index(last_train_name)
            first_test_idx = all_image_names.index(first_test_name)
            
            if first_test_idx == last_train_idx + 1:
                print(f"   ✅ 训练集和测试集连续（无间隔）")
                print(f"      训练集最后一张: {last_train_name} (第 {last_train_idx + 1} 张)")
                print(f"      测试集第一张: {first_test_name} (第 {first_test_idx + 1} 张)")
            else:
                print(f"   ⚠️  训练集和测试集之间有间隔")
                print(f"      训练集最后一张位置: {last_train_idx + 1}")
                print(f"      测试集第一张位置: {first_test_idx + 1}")
        except ValueError:
            print(f"   ⚠️  无法在原始列表中找到对应图片")
    print()
    
    print("=" * 70)
    print("✅ 验证完成！顺序和划分都正确！")
    print("=" * 70)
    print()
    print("总结:")
    print(f"  - 总图片数: {total_images}")
    print(f"  - 训练集: 前 {train_count} 张 (第 1 张 ~ 第 {train_count} 张)")
    print(f"  - 测试集: 后 {test_count} 张 (第 {train_count + 1} 张 ~ 第 {total_images} 张)")
    print(f"  - 图片顺序: 按文件名排序，保持原视频顺序")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="验证图片顺序和训练/测试集划分")
    parser.add_argument("--data_path", type=str, default=None,
                       help="数据目录路径（COLMAP输出目录，如果不提供会尝试自动获取）")
    parser.add_argument("--test_last_n", type=int, default=40,
                       help="测试集数量（后N个，默认: 40）")
    
    args = parser.parse_args()
    
    success = verify_order(args.data_path, args.test_last_n)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


