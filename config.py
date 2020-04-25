import numpy as np


class Config:
    def __init__(self):
        # 训练和测试共用的配置
        self.res_name = 'resnet50'
        self.num_cls = 16  # 类别数
        # 注意CenterNet是没有背景类的,从头到尾都没有.它是通过cls_score来判断是背景还是某一类别.一般背景类的值会低于0.3
        self.class_name = ["BoneMan", "Hatter", "FatMan", "LittleMan", "Cowboy", "Werewolf", "Bena", "Papjo", "Kokoyi",
                           "GreenDwarf", "Giselle", "Bellett", "StormRider", "Gold", "Door", "Close", ]
        # 进行数据预处理时的均值和方差,每个任务场景都不相同.需要分别计算
        self.mean = [0.8292903, 0.74784886, 0.80975633]
        self.std = [0.1553852, 0.20757463, 0.16293081]
        self.max_object = 20  # 一张图里最多有多少个目标，可以设大，不能设小
        self.input_size = 512   # 网络输入尺寸
        self.train_txt = r'D:\py_pro\CenterNet-PyTorch\data\kalete\\train.txt'   # 训练集路径
        self.val_txt = r'D:\py_pro\CenterNet-PyTorch\data\kalete\\val.txt'   # 验证集路径
        self.load_model = r'weights/map_0.9676.pt'  # 继续训练或测试的权重

        # 训练部分专用的配置
        self.lr = 1e-4
        self.batch_size = 4
        self.num_epoch = 40

        # 测试部分专用的配置
        self.test_dir = r'D:\py_pro\CenterNet-PyTorch\test'  # 存放测试图片的文件夹
        self.iou_threshold = 0.5    # 计算mAP时pred_box与gt_box的iou值,TP的条件之一

cfg = Config()







