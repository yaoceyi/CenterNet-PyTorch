# 计算数据集均值方差,可以使用这部分代码计算数据集的均值方差,然后在测试时使用
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def get_mean_std(img_root):
    means = 0
    stds = 0
    img_list = os.listdir(img_root)
    num = len(img_list)
    for img in tqdm(img_list):
        img = os.path.join(img_root, img)
        # 将所有格式的图片都转为RGB格式,然后在w,h维度上计算均值和方差
        img = np.array(Image.open(img).convert('RGB'))
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        means += mean
        stds += std
    mean = means / (num * 255)
    std = stds / (num * 255)
    print(mean)
    print(std)


get_mean_std(r'D:\py_pro\CenterNet-PyTorch\data\kalete\JPGImages')