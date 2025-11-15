#!/usr/bin/env python3
"""
检查训练/测试集划分情况
显示划分后的统计信息
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
import argparse as arg_parse


def check_split(data_path, test_last_n=0, eval_mode=True):
    """检查训练/测试集划分"""
    print(f"数据路径: {data_path}")
    print(f"测试集数量: {test_last_n} (后N个)")
    print(f"评估模式: {eval_mode}")
    print()
    
    # 创建参数解析器
    parser = arg_parse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args(['-s', data_path])
    
    if eval_mode:
        args.eval = True
    else:
        args.eval = False
    
    try:
        # 加载场景信息
        scene_info = sceneLoadTypeCallbacks['Colmap'](
            data_path, 
            args.images, 
            args.eval, 
            need_features=False, 
            need_masks=False, 
            sample_rate=1.0, 
            allow_principle_point_shift=args.allow_principle_point_shift, 
            replica='replica' in data_path,
            test_last_n=test_last_n
        )
        
        total = len(scene_info.train_cameras) + len(scene_info.test_cameras)
        train_count = len(scene_info.train_cameras)
        test_count = len(scene_info.test_cameras)
        
        print("=" * 50)
        print("训练/测试集划分结果")
        print("=" * 50)
        print(f"总图片数: {total}")
        print(f"训练集: {train_count} 张 (前 {train_count} 张)")
        print(f"测试集: {test_count} 张 (后 {test_count} 张)")
        print()
        
        if test_count > 0:
            print("训练集范围: 第 1 张 ~ 第 {} 张".format(train_count))
            print("测试集范围: 第 {} 张 ~ 第 {} 张".format(train_count + 1, total))
            print()
            
            # 显示一些示例文件名
            print("训练集示例 (前5张):")
            for i, cam in enumerate(scene_info.train_cameras[:5]):
                print(f"  {i+1}. {cam.image_name}")
            
            if len(scene_info.train_cameras) > 5:
                print(f"  ... (共 {train_count} 张)")
            
            print("\n测试集示例 (前5张):")
            for i, cam in enumerate(scene_info.test_cameras[:5]):
                print(f"  {i+1}. {cam.image_name}")
            
            if len(scene_info.test_cameras) > 5:
                print(f"  ... (共 {test_count} 张)")
        
        print()
        print("=" * 50)
        
        # 验证划分是否正确
        if test_last_n > 0:
            expected_train = total - test_last_n
            if train_count == expected_train and test_count == test_last_n:
                print("✅ 划分正确!")
            else:
                print(f"⚠️  划分可能不正确:")
                print(f"   期望: {expected_train} 训练 + {test_last_n} 测试")
                print(f"   实际: {train_count} 训练 + {test_count} 测试")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="检查训练/测试集划分")
    parser.add_argument("--data_path", type=str, required=True,
                       help="数据目录路径（COLMAP输出目录）")
    parser.add_argument("--test_last_n", type=int, default=0,
                       help="测试集数量（后N个，默认: 0=不划分）")
    parser.add_argument("--no_eval", action="store_true",
                       help="不使用评估模式（所有图片都用于训练）")
    
    args = parser.parse_args()
    
    check_split(args.data_path, args.test_last_n, not args.no_eval)


if __name__ == "__main__":
    main()




