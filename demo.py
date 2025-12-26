import h5py
import numpy as np

def check_dataset_shapes(dataset1_path, dataset2_path, num_samples=5000):
    """
    检查两个数据集的图像形状，确保它们的形状一致。
    """
    with h5py.File(dataset1_path, 'r') as f1, h5py.File(dataset2_path, 'r') as f2:
        # 获取第一个数据集的前 num_samples 个图像的形状
        shape1 = f1[f'image_0'][:].shape
        shape2 = f2[f'image_0'][:].shape

        print(f"Shape of dataset1 image: {shape1}")
        print(f"Shape of dataset2 image: {shape2}")

# 示例使用：
dataset1_path = "datasets/train_FBPzimuvari.h5"
dataset2_path = "datasets/train_FBPFigure.h5"
check_dataset_shapes(dataset1_path, dataset2_path)
