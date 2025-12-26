import numpy as np
from scipy.io import loadmat
import torch
from torchvision.utils import save_image

import os


# 把mat的二维数据保存成png
def mat_to_image(path, save_path):
    # 读取 mat 文件
    data = loadmat(path)
    print(data.keys())
    proj_mag_crop_bc = data['proj_mag_crop_bc']  # 这里改成你的实际 key   sinogram_abs_3f_D   IR_crop proj_mag_crop_bc

    image = proj_mag_crop_bc.T.squeeze()  # 转置 + 压缩多余维度

    # 转为 float32
    image = image.astype(np.float32)

    # 归一化到 0-1，防止保存图片时溢出
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # 扩展 batch 和 channel 维度: [1, 1, H, W]
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    # 保存为 PNG 图片
    save_image(image, save_path)
    print(f"图片已保存到: {save_path}")




# 调用示例
# mat_to_image("datasets/ground/0918/M.mat", "datasets/ground/0918/M2.png")
