import torch
import torch.nn.functional as F
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict

def center_crop(tensor, crop_size=50):
    """
    tensor: [N, C, H, W]
    """
    _, _, H, W = tensor.shape
    assert crop_size <= H and crop_size <= W

    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2

    return tensor[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
def calculate_image_metricsv2(
    valGround: torch.Tensor,
    valOut: torch.Tensor,
    data_range: float = 1.0
) -> Dict[str, dict]:
    """
    按“每张图像”计算 PSNR / RMSE / NRMSE / SSIM，
    然后统计 mean 和 std（论文级标准）

    Args:
        valGround: [N, C, H, W] torch.Tensor
        valOut:    [N, C, H, W] torch.Tensor
        data_range: 图像动态范围（归一化图像一般为 1.0）

    Returns:
        metrics: dict
    """
    # valGround = center_crop(valGround, 100)
    # valOut = center_crop(valOut, 100)
    assert valGround.shape == valOut.shape, "GT 与预测尺寸必须一致"

    # ---- 转 numpy，skimage 只吃 numpy ----
    gt = valGround.detach().cpu().numpy()
    out = valOut.detach().cpu().numpy()

    N, C, H, W = gt.shape

    psnr_list = []
    rmse_list = []
    nrmse_list = []
    ssim_list = []

    for i in range(N):
        gt_img = gt[i]
        out_img = out[i]

        # ---------- PSNR ----------
        psnr_val = peak_signal_noise_ratio(
            gt_img,
            out_img,
            data_range=data_range
        )
        psnr_list.append(psnr_val)

        # ---------- RMSE ----------
        mse = np.mean((gt_img - out_img) ** 2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        # ---------- NRMSE（L2 归一化） ----------
        nrmse = np.linalg.norm(gt_img - out_img) / (
            np.linalg.norm(gt_img) + 1e-12
        )
        nrmse_list.append(nrmse)

        # ---------- SSIM ----------
        # 对多通道，skimage 支持 channel_axis
        ssim_val = structural_similarity(
            gt_img[0],  # [H, W]
            out_img[0],
            data_range=data_range,
        )
        ssim_list.append(ssim_val)

    # ---- 统计均值和方差 ----
    metrics = {
        "PSNR": {
            "mean": float(np.mean(psnr_list)),
            "std":  float(np.std(psnr_list)),
            "list": psnr_list
        },
        "RMSE": {
            "mean": float(np.mean(rmse_list)),
            "std":  float(np.std(rmse_list)),
            "list": rmse_list
        },
        "NRMSE": {
            "mean": float(np.mean(nrmse_list)),
            "std":  float(np.std(nrmse_list)),
            "list": nrmse_list
        },
        "SSIM": {
            "mean": float(np.mean(ssim_list)),
            "std":  float(np.std(ssim_list)),
            "list": ssim_list
        }
    }
    for k, v in metrics.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    return metrics


def calculate_image_metrics(valGround, valOut, max_val=1.0, window_size=11, K1=0.01, K2=0.03):
    """
    计算每张图像的PSNR、SSIM和RMSE

    参数:
    - valGround: torch.Tensor, shape [N, C, H, W]
    - valOut: torch.Tensor, shape [N, C, H, W]
    - max_val: 图像最大值
    - window_size: SSIM高斯窗口大小
    - K1, K2: SSIM常数

    返回:
    - metrics: dict, 包含每个指标的 mean, std, 和列表
    """
    assert valGround.shape == valOut.shape, "尺寸必须一致"
    N, C, H, W = valGround.shape

    # 计算每张图片的MSE和RMSE


    # 每张图的NRMSE
    rmse_list = torch.norm(valGround.view(N, -1) - valOut.view(N, -1), dim=1) / torch.norm(valGround.view(N, -1), dim=1)

    rmse_mean = rmse_list.mean().item()
    rmse_std = rmse_list.std().item()

    # 计算每张图片的PSNR
    mse = torch.mean((valGround - valOut) ** 2, dim=[1, 2, 3])
    psnr_list = 10 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))
    psnr_mean = psnr_list.mean().item()
    psnr_std = psnr_list.std().item()

    # 简单实现单通道SSIM（平均每张图像）
    # 使用均值和方差计算公式
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    ssim_list = []
    for i in range(N):
        img1 = valGround[i]
        img2 = valOut[i]
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var(unbiased=False)
        sigma2 = img2.var(unbiased=False)
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        ssim_list.append(ssim_val.item())

    ssim_tensor = torch.tensor(ssim_list)
    ssim_mean = ssim_tensor.mean().item()
    ssim_std = ssim_tensor.std().item()

    print(psnr_mean, rmse_mean, ssim_mean)
    print(psnr_std, rmse_std, ssim_std)
    return

# 示例调用
# metrics = calculate_image_metrics(valGround, valOut, max_val=1.0)
# print(f"PSNR: {metrics['PSNR']['mean']:.2f} ± {metrics['PSNR']['std']:.2f}")
# print(f"RMSE: {metrics['RMSE']['mean']:.4f} ± {metrics['RMSE']['std']:.4f}")
# print(f"SSIM: {metrics['SSIM']['mean']:.4f} ± {metrics['SSIM']['std']:.4f}")
