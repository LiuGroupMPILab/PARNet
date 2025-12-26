import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import cv2
from data import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时规避 OpenMP 冲突

# 假设你已经提取了一个 batch 的数据
mraFolderPath = "datasets/"
trainDatasetlable = MRAdatasetH5NoScale(mraFolderPath + "train_PreFBPFigure.h5", prefetch=True, dim=2,
                                        device=torch.device('cuda'))

# 获取整个数据集的所有图像并转换成一个 Tensor
all_images = []
for i in range(len(trainDatasetlable)):  # 遍历整个数据集
    image = trainDatasetlable[i]  # 获取单张图像
    all_images.append(image.unsqueeze(0))  # 将其增加一个维度，成为 [1, C, H, W]，然后加入列表

# 将所有图像堆叠成一个 Tensor，形状为 [N, C, H, W]
image_tensor = torch.cat(all_images, dim=0)

image_tensor = image_tensor[0:10]
transform = transforms.Compose([
        transforms.Resize([128,128], antialias=True),  # 调整图片大小
    ])
image_tensor = transform(image_tensor)


def apply_bilateral_filter(image, d=6, sigma_color=25, sigma_space=25):
    """使用 OpenCV 进行双边滤波以保持边缘平滑"""

    # 如果是二维图像（单通道），加上一个通道维度，使其形状变为 [1, H, W]
    if image.dim() == 2:  # 如果是二维图像
        image = image.unsqueeze(0)  # 将 [H, W] 转换为 [1, H, W]

    # 将 PyTorch tensor 转换为 NumPy 数组并改变维度顺序 [C, H, W] -> [H, W, C]
    image_np = image.cpu().numpy().transpose(1, 2, 0)  # 转换为 [H, W, C]

    # 使用双边滤波
    if image_np.ndim == 3:  # 彩色图像
        blurred_image = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)
    else:  # 单通道图像
        blurred_image = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)

    # 如果图像是单通道的，不进行 permute
    if blurred_image.ndim == 2:  # 单通道图像
        blurred_image = torch.from_numpy(blurred_image).float().to(image.device)  # 直接转换为 [H, W]
    else:  # 三通道图像
        blurred_image = torch.from_numpy(blurred_image).permute(2, 0, 1).float().to(image.device)  # 转换回 [C, H, W]

    # 如果是单通道图像，移除通道维度
    if image.dim() == 3 and image.shape[0] == 1:
        blurred_image = blurred_image.squeeze(0)  # 移除通道维度，恢复为 [H, W]

    return blurred_image


def enhance_edges(image, d=9, sigma_color=75, sigma_space=75, edge_weight=1.5):
    """增强边缘并结合原图像"""

    # 如果是二维图像（单通道），加上一个通道维度，使其形状变为 [1, H, W]
    if image.dim() == 2:  # 如果是二维图像
        image = image.unsqueeze(0)  # 将 [H, W] 转换为 [1, H, W]

    # 将 PyTorch tensor 转换为 NumPy 数组并改变维度顺序 [C, H, W] -> [H, W, C]
    image = image.cpu().numpy().transpose(1, 2, 0)  # 转换为 [H, W, C]

    # 使用双边滤波进行初步模糊
    blurred_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # 检查图像是否已经是灰度图（单通道）
    if len(blurred_image.shape) == 3 and blurred_image.shape[2] == 3:
        # 转换为灰度图进行边缘检测
        blurred_image_gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    else:
        # 如果已经是灰度图，直接使用
        blurred_image_gray = blurred_image

    # 确保灰度图是 uint8 类型
    blurred_image_gray = (blurred_image_gray * 255).astype(np.uint8)

    # 进行 Canny 边缘检测
    edges_bgr = cv2.Canny(blurred_image_gray, threshold1=100, threshold2=200)

    # 将边缘图像转换为三通道的 BGR 图像（以便与原图像进行加权合成）
    # edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 调整图像尺寸，确保大小一致
    if blurred_image_gray.shape[:2] != edges_bgr.shape[:2]:
        edges_bgr = cv2.resize(edges_bgr, (blurred_image_gray.shape[1], blurred_image_gray.shape[0]))

    # 增强边缘：将边缘图像与模糊图像进行加权
    edges_bgr = edges_bgr.astype(np.uint8)
    enhanced_image = cv2.addWeighted(blurred_image_gray, 1, edges_bgr, edge_weight, 0)

    # 转换回 PyTorch tensor 并返回

    enhanced_image = torch.from_numpy(enhanced_image).float()  # 转换回 [C, H, W]
    enhanced_image = enhanced_image.unsqueeze(0)
    return enhanced_image
# 对 batch 中的每张图片应用高斯模糊
image_batch_smoothed = [enhance_edges(img) for img in image_tensor]

# 只显示前 8 张图片
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# 遍历并显示前 8 张图片
for i, ax in enumerate(axes.flat):
    if i < len(image_batch_smoothed):  # 如果批次中的图像数量大于或等于 8
        img = image_batch_smoothed[i]#.cpu().numpy() # 获取图像并转到 CPU 上
        if img.ndim == 3:  # 如果是三通道图像（C, H, W）
            img = img[0, :, :]  # 选择第一个通道作为灰度图像
        ax.imshow(img, cmap='gray')  # 使用灰度 colormap
        ax.axis('off')  # 隐藏坐标轴
    else:
        ax.axis('off')  # 隐藏空的子图

plt.tight_layout()
plt.show()
