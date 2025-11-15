"""
分割mask边缘优化脚本
用于减少边框光晕和不清晰问题
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy import ndimage
from skimage import morphology, filters, segmentation


def edge_refinement(mask: np.ndarray, method='gaussian_blur', **kwargs) -> np.ndarray:
    """
    边缘细化方法
    
    Args:
        mask: 二值mask (H, W)
        method: 细化方法
        **kwargs: 方法特定参数
    
    Returns:
        细化后的mask
    """
    if method == 'gaussian_blur':
        # 方法1: 高斯模糊 + 阈值
        sigma = kwargs.get('sigma', 1.0)
        threshold = kwargs.get('threshold', 0.5)
        
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
        refined = (blurred > threshold).astype(np.uint8)
        return refined
    
    elif method == 'morphological_gradient':
        # 方法2: 形态学梯度（保留边缘，去除光晕）
        kernel_size = kwargs.get('kernel_size', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 先腐蚀再膨胀，得到更清晰的边缘
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        return dilated
    
    elif method == 'distance_transform':
        # 方法3: 距离变换 + 分水岭（去除边缘模糊）
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        threshold = kwargs.get('threshold', 0.3 * dist_transform.max())
        refined = (dist_transform > threshold).astype(np.uint8)
        return refined
    
    elif method == 'active_contour':
        # 方法4: 活动轮廓（snake算法）- 需要scikit-image
        try:
            from skimage.segmentation import active_contour
            from skimage.filters import gaussian
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return mask
            
            # 使用最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 初始化snake
            s = np.linspace(0, 2*np.pi, len(largest_contour))
            init = np.array([largest_contour[:, 0, 1], largest_contour[:, 0, 0]]).T
            
            # 平滑图像
            smoothed = gaussian(mask.astype(float), sigma=1.0)
            
            # 运行active contour
            snake = active_contour(smoothed, init, alpha=0.015, beta=10, gamma=0.001)
            
            # 创建新mask
            refined = np.zeros_like(mask)
            snake_int = snake.astype(int)
            cv2.fillPoly(refined, [snake_int], 1)
            return refined
        except ImportError:
            print("Warning: scikit-image not available, using fallback method")
            return morphological_refinement(mask, **kwargs)
    
    elif method == 'bilateral_filter':
        # 方法5: 双边滤波（保持边缘的同时去噪）
        d = kwargs.get('d', 5)
        sigma_color = kwargs.get('sigma_color', 50)
        sigma_space = kwargs.get('sigma_space', 50)
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(mask_uint8, d, sigma_color, sigma_space)
        refined = (filtered > 127).astype(np.uint8)
        return refined
    
    else:
        return mask


def morphological_refinement(mask: np.ndarray, 
                           opening_kernel: int = 2,
                           closing_kernel: int = 3,
                           remove_small: int = 0) -> np.ndarray:
    """
    形态学细化：去除光晕和噪点
    
    Args:
        mask: 输入mask
        opening_kernel: 开运算核大小（去除小噪点）
        closing_kernel: 闭运算核大小（填补小洞）
        remove_small: 移除小于此像素数的连通域
    
    Returns:
        细化后的mask
    """
    mask_uint8 = mask.astype(np.uint8)
    
    # 1. 开运算：去除边缘的小噪点和光晕
    if opening_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    
    # 2. 闭运算：填补小洞
    if closing_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. 移除小连通域
    if remove_small > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels > 1:
            # 找到最大连通域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # 保留最大连通域
            mask_uint8 = (labels == largest_label).astype(np.uint8)
    
    return mask_uint8


def edge_aware_refinement(mask: np.ndarray, image: np.ndarray = None,
                         edge_threshold: float = 0.1) -> np.ndarray:
    """
    边缘感知细化：利用原图的边缘信息
    
    Args:
        mask: 分割mask
        image: 原始图像（可选，如果有的话可以更好地保留边缘）
        edge_threshold: 边缘阈值
    
    Returns:
        细化后的mask
    """
    if image is not None:
        # 使用Canny边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # 在边缘附近保留mask，远离边缘的地方使用形态学操作
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edge_mask = edge_dilated > 0
        
        # 在边缘区域保持原mask，非边缘区域进行形态学操作
        refined = mask.copy()
        non_edge = ~edge_mask
        refined[non_edge] = morphological_refinement(mask[non_edge], opening_kernel=5, closing_kernel=3)[non_edge]
    else:
        # 没有原图时，使用标准方法
        refined = morphological_refinement(mask)
    
    return refined


def multi_scale_refinement(mask: np.ndarray, scales: list = [1.0, 0.5, 2.0]) -> np.ndarray:
    """
    多尺度细化：在不同尺度下细化，然后融合
    
    Args:
        mask: 输入mask
        scales: 尺度列表
    
    Returns:
        融合后的mask
    """
    refined_masks = []
    
    for scale in scales:
        if scale == 1.0:
            scaled_mask = mask
        else:
            h, w = mask.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            scaled_mask = cv2.resize(scaled_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        refined = morphological_refinement(scaled_mask.astype(np.uint8))
        refined_masks.append(refined)
    
    # 投票融合
    refined = np.mean(refined_masks, axis=0) > 0.5
    return refined.astype(np.uint8)


def process_mask_file(mask_path: Path, output_path: Path, 
                     method: str = 'morphological',
                     image_path: Path = None,
                     **kwargs) -> None:
    """
    处理单个mask文件
    
    Args:
        mask_path: 输入mask路径
        output_path: 输出mask路径
        method: 细化方法
        image_path: 原始图像路径（可选，用于边缘感知细化）
        **kwargs: 方法特定参数
    """
    # 加载mask
    if mask_path.suffix == '.pt':
        mask_tensor = torch.load(mask_path)
        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.cpu().numpy()
        else:
            mask = mask_tensor
        # 处理多通道或批次维度
        if len(mask.shape) > 2:
            mask = mask[0] if mask.shape[0] == 1 else mask
        mask = mask.astype(np.uint8)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
    
    # 加载原图（如果提供）
    image = None
    if image_path and image_path.exists():
        image = cv2.imread(str(image_path))
    
    # 应用细化方法
    if method == 'morphological':
        refined = morphological_refinement(mask, **kwargs)
    elif method == 'edge_aware':
        refined = edge_aware_refinement(mask, image, **kwargs)
    elif method == 'multi_scale':
        refined = multi_scale_refinement(mask, **kwargs)
    elif method in ['gaussian_blur', 'morphological_gradient', 'distance_transform', 
                    'bilateral_filter']:
        refined = edge_refinement(mask, method=method, **kwargs)
    else:
        refined = mask
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == '.pt':
        torch.save(torch.from_numpy(refined.astype(bool)), output_path)
    else:
        cv2.imwrite(str(output_path), (refined * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Refine segmentation masks to reduce edge halos")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="Directory containing input masks")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for refined masks")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Directory containing original images (optional, for edge-aware refinement)")
    parser.add_argument("--method", type=str, default="morphological",
                       choices=["morphological", "edge_aware", "multi_scale", 
                               "gaussian_blur", "morphological_gradient", 
                               "distance_transform", "bilateral_filter"],
                       help="Refinement method")
    parser.add_argument("--opening_kernel", type=int, default=2,
                       help="Opening kernel size for morphological operations (default: 2, conservative)")
    parser.add_argument("--closing_kernel", type=int, default=3,
                       help="Closing kernel size for morphological operations (default: 3, conservative)")
    parser.add_argument("--remove_small", type=int, default=0,
                       help="Remove connected components smaller than this (default: 0, disabled to preserve IoU)")
    parser.add_argument("--mask_ext", type=str, default=".pt",
                       help="Mask file extension (.pt or .png)")
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # IMPORTANT: Sort mask files by filename to maintain temporal order from original video
    # This ensures the order matches the original video sequence
    mask_files = sorted(list(mask_dir.glob(f"*{args.mask_ext}")), key=lambda x: x.name)
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Using refinement method: {args.method}")
    print(f"[order guarantee] Processing masks in temporal order (sorted by filename)")
    
    # 处理每个mask
    for mask_path in tqdm(mask_files, desc="Refining masks"):
        # 构建输出路径
        output_path = output_dir / mask_path.name
        
        # 查找对应的原图（如果提供）
        image_path = None
        if image_dir:
            # 尝试不同的图像扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_image = image_dir / (mask_path.stem + ext)
                if potential_image.exists():
                    image_path = potential_image
                    break
        
        # 处理mask
        process_mask_file(
            mask_path,
            output_path,
            method=args.method,
            image_path=image_path,
            opening_kernel=args.opening_kernel,
            closing_kernel=args.closing_kernel,
            remove_small=args.remove_small
        )
    
    print(f"\nRefined masks saved to {output_dir}")


if __name__ == "__main__":
    main()


