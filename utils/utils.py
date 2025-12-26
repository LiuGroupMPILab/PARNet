
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
def gaussian_window(window_size=11, sigma=1.5, channel=1):
    """
    生成二维高斯窗口，用于SSIM计算
    window_size: 窗口大小（一般11）
    sigma: 高斯核标准差
    channel: 输入通道数
    """
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()  # 归一化

    # 外积生成二维高斯核
    window_1d = g.unsqueeze(1)              # (window_size, 1)
    window_2d = window_1d @ window_1d.T     # (window_size, window_size)

    # 升维成 (channel, 1, window_size, window_size)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channel, 1, 1, 1)
    return window

def center_crop(tensor, crop_size=50):
    """
    tensor: [N, C, H, W]
    """
    _, _, H, W = tensor.shape
    assert crop_size <= H and crop_size <= W

    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2

    return tensor[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
def compute_ssim_batch(img1, img2, data_range=1.0):
    """
    img1, img2: torch.Tensor [N, 1, H, W]
    返回: 平均 SSIM
    """
    img1 = center_crop(img1, 100)
    img2 = center_crop(img2, 100)
    assert img1.shape == img2.shape

    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    ssim_list = []
    for i in range(img1.shape[0]):
        ssim_val = ssim(
            img1[i, 0],      # [H, W]
            img2[i, 0],
            data_range=data_range
        )
        ssim_list.append(ssim_val)

    return float(np.mean(ssim_list))
def ssim_index(img1, img2, window_size=11, sigma=1.5, data_range=1.0, K1=0.01, K2=0.03):
    """
    计算SSIM，输入必须是 [N, C, H, W] 的Tensor
    img1, img2: [N, C, H, W], float，范围 [0, data_range]
    返回: 平均SSIM标量(float)
    """
    img1 = center_crop(img1, 50)
    img2 = center_crop(img2, 50)
    # images = img2.squeeze(1).detach().cpu().numpy()  # [64, 26, 13]
    # # 设置子图的行列数（例如 8x8 网格）
    # rows, cols = 1, 4
    # fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    #
    # # 遍历并显示 64 张图片
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(images[i], cmap='gray')  # 选择灰度 colormap
    #     ax.axis('off')  # 隐藏坐标轴
    #
    # plt.tight_layout()
    # plt.show()

    assert img1.shape == img2.shape, f"Image shapes are different: {img1.shape} vs {img2.shape}"
    channel = img1.shape[1]

    # 高斯窗口移到相同设备
    window = gaussian_window(window_size, sigma, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, groups=channel, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, groups=channel, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel, padding=window_size // 2) - mu1_mu2

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()  # 输出标量(float)



def weights_init_kaiming(lyr):
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant_(lyr.bias.data, 0.0)


def rotate_center_circle(arr, angle):
    """
    在正方形二维数组中，以中心为圆心、边长一半为半径的圆形区域逆时针旋转一定角度
    :param arr: 输入二维数组 (N x N)
    :param angle: 旋转角度（度），逆时针为正
    :return: 旋转后的二维数组
    """
    assert arr.ndim == 2, "输入必须是二维数组"
    h, w = arr.shape
    assert h == w, "输入必须是正方形数组"

    # 中心 & 半径
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = w / 2.0

    # 角度转弧度
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 生成掩膜
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    # 拷贝原数组
    out = arr.copy()

    # 圆内点坐标
    ys, xs = np.nonzero(mask)
    dx = xs - cx
    dy = ys - cy

    # 逆向旋转到原图位置
    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    # 最近邻采样
    src_x = np.rint(src_x).astype(int)
    src_y = np.rint(src_y).astype(int)

    # 限制在边界内
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    out[ys[valid], xs[valid]] = arr[src_y[valid], src_x[valid]]

    return out