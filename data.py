from torch.utils.data import Dataset
import random
import h5py
import torch
import numpy as np

from torchvision import transforms

transformations = transforms.Compose([transforms.ToTensor()])  # It also scales to 0-1 by dividing by 255.



class MRAdatasetH5NoScale(Dataset):
    def __init__(self, inputpath, targetpath, transform=None, prefetch=True, dim=2, device=None):

        super(MRAdatasetH5NoScale, self).__init__()

        # 打开输入数据和目标数据文件
        self.h5f_input = h5py.File(inputpath, 'r')
        self.h5f_target = h5py.File(targetpath, 'r')

        self.keys = list(self.h5f_input.keys())  # 获取所有的样本名称

        self.prefetch = prefetch
        self.transform = transform  # 数据变换
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.prefetch:
            # 预加载所有数据到内存

            self.input_data = torch.zeros((len(self.keys), 1, *(np.array(self.h5f_input[self.keys[0]])).shape[-dim:]))
            self.target_data = torch.zeros((len(self.keys), 1, *(np.array(self.h5f_target[self.keys[0]])).shape[-dim:]))

            for i in range(len(self.keys)):
                self.input_data[i] = torch.tensor(np.array(self.h5f_input[self.keys[i]]))
                self.target_data[i] = torch.tensor(np.array(self.h5f_target[self.keys[i]]))

            # 数据预处理：归一化数据
            self.input_data = self.input_data.to(self.device).float() / self.input_data.float().max()
            self.target_data = self.target_data.to(self.device).float() / self.target_data.float().max()
            self.h5f_input.close()
            self.h5f_target.close()

        else:
            random.shuffle(self.keys)  # 打乱数据
            self.h5f_input.close()
            self.h5f_target.close()

    def __len__(self):
        """返回数据集的大小"""
        return len(self.keys)

    def __getitem__(self, index):
        """
        获取输入数据和目标数据
        :param index: 索引
        :return: 返回一个样本
        """
        theIndex = index % len(self.keys)

        if self.prefetch:
            input_data = self.input_data[theIndex]
            target_data = self.target_data[theIndex]
        else:
            # 获取数据
            key = self.keys[theIndex]
            input_data = np.array(self.h5f_input[key])
            target_data = np.array(self.h5f_target[key])

            # 如果有transform，应用数据增强或归一化
            if self.transform:
                input_data = self.transform(input_data)
                target_data = self.transform(target_data)

            # 转换为tensor并移动到设备
            input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            target_data = torch.tensor(target_data, dtype=torch.float32).to(self.device)

        return input_data, target_data

    def openFile(self, inputpath, targetpath):
        """打开文件"""
        self.h5f_input = h5py.File(inputpath, 'r')
        self.h5f_target = h5py.File(targetpath, 'r')

    def closeFile(self):
        """关闭文件"""
        self.h5f_input.close()
        self.h5f_target.close()
