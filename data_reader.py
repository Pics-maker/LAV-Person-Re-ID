import os
import re

import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, root_dir, RGB_dir, gait_dir, mapper: list, img_size=(128, 128)):
        """
        :param root_dir: 数据集根目录
        :param RGB_dir: 彩色图像目录
        :param gait_dir: 步态数据目录
        :param mapper: 数据集对应的ID-nm-angle映射
        :param rgb_size: 调整RGB大小为多少
        """
        self.root_dir = root_dir
        self.RGB_dir = RGB_dir
        self.gait_dir = gait_dir
        self.mapper = mapper
        self.img_size = img_size

        self.RGB_root_path = self.root_dir + self.RGB_dir
        self.gait_root_path = self.root_dir + self.gait_dir

        self.RGB_path_list = os.listdir(self.RGB_root_path)
        self.gait_path_list = os.listdir(self.gait_root_path)

        # 角度对应权重的map
        self.weight = {
            "000": [1, 0.5],
            "018": [1, 0.6],
            "036": [1, 0.7],
            "054": [1, 0.8],
            "072": [1, 0.9],
            "090": [1, 1.0],
            "108": [1, 0.9],
            "126": [1, 0.8],
            "144": [1, 0.7],
            "162": [1, 0.6],
            "180": [1, 0.5]
        }

    def __getitem__(self, item):
        """
        :param item: index
        :return: [[RGB1, RGB2, RGB3, RGB4], gait, appearance_weight, gait_weight], ID
        """
        # 确定是第几个人的第几个状态
        info = self.mapper[item]  # 形如'001_nm-01_000'的字符串
        ID, nm_ID, angle = info.split("_")

        # 读取RGB图片
        RGB_path = os.path.join(self.RGB_root_path, ID, nm_ID, angle)  # RGB文件夹路径
        RGB_list = os.listdir(RGB_path)
        RGB_list.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))  # 对文件列表排序
        num = len(RGB_list)
        second = int(num * 0.3)
        third = int(num * 0.7)
        # 连接图片的路径
        first_path = os.path.join(RGB_path, RGB_list[0])
        second_path = os.path.join(RGB_path, RGB_list[second])
        third_path = os.path.join(RGB_path, RGB_list[third])
        last_path = os.path.join(RGB_path, RGB_list[-1])

        RGB_tensor = []
        for p in [first_path, second_path, third_path, last_path]:
            img = cv2.resize(cv2.imread(p), self.img_size)
            img_tensor = transforms.ToTensor()(img)
            RGB_tensor.append(img_tensor)

        # 读取步态图片
        gait_path = os.path.join(self.gait_root_path, ID, nm_ID, angle + ".jpg")  # 步态图像路径
        gait = cv2.imread(gait_path)
        gait = cv2.resize(gait, self.img_size)
        gait_tensor = transforms.ToTensor()(gait)

        id = int(ID) - 1  # index索引要从0开始

        return [RGB_tensor, gait_tensor, self.weight[angle][0], self.weight[angle][1]], id

    def __len__(self):
        return len(self.mapper)
