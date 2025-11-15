#!/usr/bin/env python3
"""
检查 Scene 实际加载了多少相机
"""
import sys
import os
sys.path.insert(0, '.')

from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
import argparse

def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    
    if len(sys.argv) < 2:
        print("用法: python scripts/check_loaded_cameras.py <source_path>")
        sys.exit(1)
    
    source_path = sys.argv[1]
    
    args = parser.parse_args(['-s', source_path])
    args = model.extract(args)
    
    print(f"检查路径: {args.source_path}")
    print(f"是否存在 sparse 目录: {os.path.exists(os.path.join(args.source_path, 'sparse'))}")
    print()
    
    try:
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, 
            args.images, 
            args.eval, 
            need_features=False, 
            need_masks=False, 
            sample_rate=1.0, 
            allow_principle_point_shift=args.allow_principle_point_shift, 
            replica='replica' in args.model_path
        )
        
        print(f"训练集相机数: {len(scene_info.train_cameras)}")
        print(f"测试集相机数: {len(scene_info.test_cameras)}")
        print(f"总计: {len(scene_info.train_cameras) + len(scene_info.test_cameras)}")
        
        if len(scene_info.train_cameras) + len(scene_info.test_cameras) < 10:
            print("\n⚠️  警告: 加载的相机数量过少！")
            print("可能的原因:")
            print("1. source_path 路径不正确")
            print("2. sparse/0 目录不存在或数据不完整")
            print("3. sample_rate 设置过小（当前为 1.0）")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()












