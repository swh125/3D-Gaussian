#!/usr/bin/env python3
"""
加载优化后的mask并应用到模型，用于在GUI中查看
"""

import torch
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel, FeatureGaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

def load_mask_and_apply(model_path: str, mask_path: str, iteration: int = 30000):
    """
    加载mask并应用到模型
    
    Args:
        model_path: 模型路径
        mask_path: mask文件路径（.pt文件）
        iteration: 模型迭代次数
    """
    print(f"Loading mask from: {mask_path}")
    mask = torch.load(mask_path)
    if isinstance(mask, torch.Tensor):
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask = mask.bool()
    else:
        mask = torch.from_numpy(np.array(mask) > 0.5).bool()
    
    print(f"Mask shape: {mask.shape}, True count: {mask.sum().item()}")
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    
    # 读取配置文件
    cfg_file = os.path.join(model_path, "cfg_args")
    args_dict = {}
    if os.path.exists(cfg_file):
        print(f"Loading config from: {cfg_file}")
        with open(cfg_file, 'r') as f:
            cfg_content = f.read()
            cfg_args = eval(cfg_content)
            args_dict = vars(cfg_args)
    
    args_dict['model_path'] = model_path
    if 'convert_SHs_python' not in args_dict:
        args_dict['convert_SHs_python'] = False
    if 'compute_cov3D_python' not in args_dict:
        args_dict['compute_cov3D_python'] = False
    if 'debug' not in args_dict:
        args_dict['debug'] = False
    
    args = Namespace(**args_dict)
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    
    # 加载高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # 应用mask
    print("Applying mask to gaussians...")
    gaussians.segment(mask.cuda())
    
    print("✓ Mask applied successfully!")
    print(f"  Remaining points: {gaussians.get_xyz.shape[0]}")
    
    return gaussians, scene


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load optimized mask and apply to model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--mask_path", type=str, required=True,
                       help="Path to mask file (.pt)")
    parser.add_argument("--iteration", type=int, default=30000,
                       help="Model iteration to load")
    
    args = parser.parse_args()
    
    load_mask_and_apply(args.model_path, args.mask_path, args.iteration)





