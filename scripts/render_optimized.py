#!/usr/bin/env python3
"""
优化的渲染脚本 - 解决mask渲染时的opacity雾气问题
在渲染mask时，临时将mask区域的opacity设为1.0，避免半透明导致的雾气
同时确保渲染训练集和测试集的2D mask和彩色图像
"""

import sys
import os
# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from scene import Scene, GaussianModel, FeatureGaussianModel
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature, render_mask
import torchvision
import importlib

# 导入原始的render模块，但我们要修改render_mask的调用
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def render_mask_with_fixed_opacity(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                                   scaling_modifier=1.0, precomputed_mask=None):
    """
    优化的render_mask：在渲染前将mask区域的opacity设为1.0，解决雾气问题
    """
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    
    # 创建screenspace_points
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置rasterization配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    
    # 关键优化：临时修改opacity，将mask区域的opacity设为1.0
    original_opacity = pc.get_opacity
    opacity = original_opacity.clone()
    
    if precomputed_mask is not None:
        # precomputed_mask是(N,)或(N,1)形状
        mask_bool = (precomputed_mask.squeeze() > 0.5)
        if mask_bool.device != opacity.device:
            mask_bool = mask_bool.to(opacity.device)
        # opacity的形状是(N, 1)，需要正确索引
        if len(opacity.shape) == 2 and opacity.shape[1] == 1:
            opacity[mask_bool, 0] = 1.0
        else:
            opacity[mask_bool] = 1.0
    else:
        # 使用pc.get_mask的情况
        pc_mask = pc.get_mask
        if pc_mask is not None:
            mask_bool = (pc_mask.squeeze() > 0.5)
            if mask_bool.device != opacity.device:
                mask_bool = mask_bool.to(opacity.device)
            if len(opacity.shape) == 2 and opacity.shape[1] == 1:
                opacity[mask_bool, 0] = 1.0
            else:
                opacity[mask_bool] = 1.0

    mask = pc.get_mask if precomputed_mask is None else precomputed_mask
    if len(mask.shape) == 1 or mask.shape[-1] == 1:
        mask = mask.squeeze().unsqueeze(-1).repeat([1,3]).cuda()

    shs = None
    colors_precomp = mask

    # 准备scales和rotations
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 渲染
    rendered_mask, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,  # 使用修改后的opacity
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    return {"mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def render_set_optimized(model_path, name, iteration, views, gaussians, pipeline, background, 
                        target, precomputed_mask=None, apply_morphology=True, opening_kernel=2, closing_kernel=3):
    """
    优化的render_set：使用修复opacity的render_mask
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    if target == 'feature':
        render_func = render_contrastive_feature
    elif target == 'contrastive_feature':
        render_func = render_contrastive_feature
    elif target == 'xyz':
        render_func = render
    else:
        render_func = render

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        res = render_func(view, gaussians, pipeline, background)

        if target == 'seg':
            assert precomputed_mask is not None, 'Rendering 2D segmentation mask requires a precomputed mask.'
            # 转换mask类型
            mask_for_render = precomputed_mask
            if isinstance(mask_for_render, torch.Tensor) and mask_for_render.dtype == torch.bool:
                mask_for_render = mask_for_render.float()
            # 使用优化的render_mask（修复opacity）
            mask_res = render_mask_with_fixed_opacity(view, gaussians, pipeline, background, precomputed_mask=mask_for_render)

        rendering = res["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if target == 'seg':
            mask = mask_res["mask"]
            # 使用更严格的阈值
            mask[mask < 0.3] = 0
            mask[mask >= 0.3] = 1
            mask = mask[0, :, :]
            
            # 应用形态学操作
            if apply_morphology:
                try:
                    import cv2
                    import numpy as np
                    from scipy.ndimage import binary_fill_holes
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    
                    # 填充空洞
                    mask_binary = (mask_np > 127).astype(bool)
                    filled = binary_fill_holes(mask_binary).astype(np.uint8) * 255
                    mask_np = np.maximum(mask_np, filled)
                    
                    # 开运算：去除光晕
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
                    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel_open)
                    
                    # 闭运算：平滑边缘
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
                    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close)
                    
                    mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(mask.device)
                except ImportError:
                    pass
            
            torchvision.utils.save_image(mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        
        if target == 'seg' or target == 'scene':
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        elif 'feature' in target:
            torch.save(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".pt"))
        elif target == 'xyz':
            torch.save(rendering, os.path.join(render_path, 'xyz_{0:05d}'.format(idx) + ".pt"))


def render_sets_optimized(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                          skip_train: bool, skip_test: bool, segment: bool = False, 
                          target='scene', idx=0, precomputed_mask=None, apply_morphology=True):
    """
    优化的render_sets：确保渲染训练集和测试集
    """
    if segment:
        assert target == 'seg' or target == 'coarse_seg_everything' or (precomputed_mask is not None), "Segmentation only works with target seg!"
    
    gaussians, feature_gaussians = None, None
    with torch.no_grad():
        if precomputed_mask is not None:
            if '.pt' in precomputed_mask:
                precomputed_mask = torch.load(precomputed_mask)
            elif '.npy' in precomputed_mask:
                import numpy as np
                precomputed_mask = torch.from_numpy(np.load(precomputed_mask)).cuda()
                precomputed_mask[precomputed_mask > 0] = 1
                precomputed_mask[precomputed_mask != 1] = 0
                precomputed_mask = precomputed_mask.bool()
            if isinstance(precomputed_mask, torch.Tensor):
                if segment and (target == 'scene' or target == 'coarse_seg_everything' or target == 'seg'):
                    if precomputed_mask.dtype != torch.bool:
                        precomputed_mask = precomputed_mask > 0.5
                    precomputed_mask = precomputed_mask.to(device="cuda", dtype=torch.bool)
                else:
                    if precomputed_mask.dtype != torch.float32:
                        precomputed_mask = precomputed_mask.float()
                    precomputed_mask = precomputed_mask.to(device="cuda")

        gaussians = GaussianModel(dataset.sh_degree)
        if target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
            feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

        scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, 
                     shuffle=False, mode='eval', 
                     target=target if target != 'xyz' and precomputed_mask is None else 'scene')

        if segment and target != 'seg':
            gaussians.segment(precomputed_mask)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        if 'feature' in target:
            gaussians = feature_gaussians
            bg_color = [1 for i in range(dataset.feature_dim)] if dataset.white_background else [0 for i in range(dataset.feature_dim)]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 确保渲染训练集和测试集
        if not skip_train:
            print(f"渲染训练集 ({len(scene.getTrainCameras())} 个视图)...")
            render_set_optimized(dataset.model_path, "train", scene.loaded_iter, 
                               scene.getTrainCameras(), gaussians, pipeline, background, 
                               target, precomputed_mask=precomputed_mask, apply_morphology=apply_morphology)

        if not skip_test:
            print(f"渲染测试集 ({len(scene.getTestCameras())} 个视图)...")
            render_set_optimized(dataset.model_path, "test", scene.loaded_iter, 
                               scene.getTestCameras(), gaussians, pipeline, background, 
                               target, precomputed_mask=precomputed_mask, apply_morphology=apply_morphology)


if __name__ == "__main__":
    parser = ArgumentParser(description="优化的渲染脚本 - 修复mask渲染时的opacity雾气问题")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--source_path', '-s', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--target', type=str, default='seg', choices=['seg', 'scene', 'feature'])
    parser.add_argument('--precomputed_mask', type=str, default=None)
    parser.add_argument('--no_morphology', action='store_true', help='禁用形态学操作')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test_last_n', type=int, default=None)
    
    args = parser.parse_args()
    
    # 创建seg_cfg_args（如果需要）
    if args.target == 'seg':
        seg_cfg_args_path = os.path.join(args.model_path, 'seg_cfg_args')
        if not os.path.exists(seg_cfg_args_path):
            cfg_args_path = os.path.join(args.model_path, 'cfg_args')
            if os.path.exists(cfg_args_path):
                import shutil
                shutil.copy(cfg_args_path, seg_cfg_args_path)
                print(f"已创建 seg_cfg_args: {seg_cfg_args_path}")
    
    # 设置参数
    dataset = ModelParams(model_path=args.model_path, source_path=args.source_path, 
                         images='images', eval=args.eval, test_last_n=args.test_last_n)
    pipeline = PipelineParams()
    
    apply_morphology = not args.no_morphology
    
    print("=" * 60)
    print("优化的渲染脚本 - 修复opacity雾气问题")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.source_path}")
    print(f"迭代次数: {args.iteration}")
    print(f"目标: {args.target}")
    print(f"3D Mask: {args.precomputed_mask}")
    print(f"形态学操作: {'启用' if apply_morphology else '禁用'}")
    print(f"渲染训练集: {'否' if args.skip_train else '是'}")
    print(f"渲染测试集: {'否' if args.skip_test else '是'}")
    print("=" * 60)
    print()
    
    render_sets_optimized(dataset, args.iteration, pipeline, 
                        args.skip_train, args.skip_test, 
                        args.segment, args.target, 
                        precomputed_mask=args.precomputed_mask,
                        apply_morphology=apply_morphology)
    
    print()
    print("=" * 60)
    print("✓ 渲染完成！")
    print("=" * 60)

