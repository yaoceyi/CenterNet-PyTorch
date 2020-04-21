import numpy as np


class Config:
    def __init__(self):
        # 训练和测试共用的配置
        self.res_name = 'resnet50'
        self.num_cls = 16  # 类别数
        # 注意CenterNet是没有背景类的
        self.class_name = ["BoneMan", "Hatter", "FatMan", "LittleMan", "Cowboy", "Werewolf", "Bena", "Papjo", "Kokoyi",
                           "GreenDwarf", "Giselle", "Bellett", "StormRider", "Gold", "Door", "Close", ]
        self.mean = np.array([0.8292903, 0.74784886, 0.80975633], dtype=np.float32).reshape(1, 1, 3)  # 数据集的归一化均值
        self.std = np.array([0.1553852, 0.20757463, 0.16293081], dtype=np.float32).reshape(1, 1, 3)  # 数据集的归一化方差
        self.max_object = 20  # 一张图里最多有多少个目标，可以设大，不能设小
        self.input_size = 512   # 网络输入尺寸
        self.train_txt = r'D:\py_pro\CenterNet-PyTorch\data\kalete\\train.txt'   # 训练集路径
        self.val_txt = r'D:\py_pro\CenterNet-PyTorch\data\kalete\\val.txt'   # 验证集路径

        # 训练部分专用的配置
        self.lr = 0.001
        self.batch_size = 4
        self.num_epoch = 15

        # 测试部分专用的配置
        self.load_model = ''  # 测试用的权重文件的路径
        self.test_dir = r'D:\py_pro\CenterNet-PyTorch\data\coco\\val2017'  # 存放测试图片的文件夹
        self.iou_threshold = 0.5    # 计算mAP时pred_box与gt_box的iou值,TP的条件之一

cfg = Config()







