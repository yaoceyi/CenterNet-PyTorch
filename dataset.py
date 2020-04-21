import numpy as np
from torch.utils.data import Dataset
from utils.utils import gaussian_radius, draw_umich_gaussian, pad_to_square
import math
from config import cfg
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


class ListDataset(Dataset):
    def __init__(self, data_txt):
        super().__init__()
        # 均值和方差是需要对训练集计算得来的.数据集不同,值也不同
        self.mean = np.array([0.8292903, 0.74784886, 0.80975633], dtype=np.float32)
        self.std = np.array([0.1553852, 0.20757463, 0.16293081], dtype=np.float32)
        # 一张图片中最多有多少个标注目标,这里可以根据具体任务场景灵活设置,原始值是50.我这里修改为20
        self.max_objs = 20
        with open(data_txt) as f:
            self.path_id_box = f.readlines()

    def __len__(self):
        return len(self.path_id_box)

    def __getitem__(self, index):
        """
        :param index: 随机索引
        :return: inp:3*512*512的图片                                       torch.Size([3, 512, 512])
                 hm:cls_id通道上目标中心点位置有高斯分布图的num_cls*128*128    (16, 128, 128)
                 offset_mask:代表了哪些位置有目标                            (max_objs,)
                 ind:代表了某个目标中心点在128*128特征图上的索引               (max_objs,)
                 wh:目标在128*128尺寸下的的实际宽高                           (max_objs, 2)
                 offset:目标中心点在128*128尺寸上的偏移值,相对于整点坐标来说    (max_objs, 2)
        """
        path_id_box = self.path_id_box[index].split()
        img_path = path_id_box[0]
        cls_id = path_id_box[1::5]
        x_min = path_id_box[2::5]
        y_min = path_id_box[3::5]
        x_max = path_id_box[4::5]
        y_max = path_id_box[5::5]
        # ToTensor这一步已经包含了归一化(1/255.0)  额外转换成RGB是为了防止 PNG等格式图片有四通道
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # 将图片填充至方形,并计算出pad在各个方向上的填充长度
        img, pad, max_side = pad_to_square(img, 0)
        # 将图片resize到网络输入大小
        img = F.interpolate(img.unsqueeze(0), size=(cfg.input_size, cfg.input_size), mode="nearest").squeeze(0)
        inp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        output_h = cfg.input_size // 4
        output_w = cfg.input_size // 4

        hm = np.zeros((cfg.num_cls, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset = np.zeros((self.max_objs, 2), dtype=np.float32)

        ind = np.zeros((self.max_objs), dtype=np.int64)  # 目标中心点在128×128特征图中的索引
        true_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 记录有多少个真值
        num_objs = len(cls_id)
        # 先计算出原始图片尺寸resize到网络输入尺寸需要放缩的倍数,然后再除以4
        box_scale = cfg.input_size / max_side / 4
        for k in range(num_objs):
            # 以下四个坐标是在网络输入尺寸缩小4倍下的特征图上的坐标
            x_min_ = (int(x_min[k]) + pad[0]) * box_scale
            y_min_ = (int(y_min[k]) + pad[1]) * box_scale
            x_max_ = (int(x_max[k]) + pad[2]) * box_scale
            y_max_ = (int(y_max[k]) + pad[3]) * box_scale

            cls_id_ = int(cls_id[k])
            # 缩小4倍后bbox的h和w
            h, w = y_max_ - y_min_, x_max_ - x_min_
            # gaussian_radius(高斯半径),这个方法的具体原理我不清楚,不过按照它给出的结果来看h,w的最小值约约为radius的3~4倍
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            # 计算出了center点
            ct = np.array([(x_min_ + x_max_) / 2, (y_min_ + y_max_) / 2], dtype=np.float32)

            # 向下取整
            ct_int = ct.astype(np.int32)
            # 因为下方画高斯图必须用int型的中心坐标画，所以引入了reg，而ct之所以不是int，就是下采样导致的
            # hm在cls_id通道上的某一目标的热力图,注意！完整的hm是128*128的.其中大部分为0,这里只是其中一部分--目标区域的数值情况
            # [[0.00315111 0.02732372 0.05613476 0.02732372 0.00315111]
            #  [0.02732372 0.23692776 0.48675226 0.23692776 0.02732372]
            #  [0.05613476 0.48675226 1. 0.48675226 0.05613476]
            #  [0.02732372 0.23692776 0.48675226 0.23692776 0.02732372]
            #  [0.00315111 0.02732372 0.05613476 0.02732372 0.00315111]]
            draw_umich_gaussian(hm[cls_id_], ct_int, radius)
            # 在128*128尺寸下的目标宽高
            wh[k] = w, h
            # 将目标在热力图中的位置赋予给ind[k]
            ind[k] = ct_int[1] * output_w + ct_int[0]
            # 偏移值的获取
            offset[k] = ct - ct_int
            # hm,wh,offset的掩膜,有目标的位置为,否则为0
            true_mask[k] = 1

        return inp, hm, true_mask, ind, wh, offset


class EvalDataset(Dataset):
    def __init__(self, data_txt):
        super().__init__()
        # 均值和方差是需要对训练集计算得来的.数据集不同,值也不同
        self.mean = np.array([0.8292903, 0.74784886, 0.80975633], dtype=np.float32)
        self.std = np.array([0.1553852, 0.20757463, 0.16293081], dtype=np.float32)
        # 一张图片中最多有多少个标注目标,这里可以根据具体任务场景灵活设置,原始值是50.我这里修改为20
        with open(data_txt) as f:
            self.path_id_box = f.readlines()

    def __len__(self):
        return len(self.path_id_box)

    def __getitem__(self, index):
        """
        :param index: 随机索引
        :return: inp:3*512*512的图片       torch.Size([3, 512, 512])

        """
        path_id_box = self.path_id_box[index].split()
        img_path = path_id_box.pop(0)
        # ToTensor这一步已经包含了归一化(1/255.0)  额外转换成RGB是为了防止 PNG等格式图片有四通道
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # 将图片填充至方形,并计算出pad在各个方向上的填充长度
        img, pad, max_side = pad_to_square(img, 0)
        # 将图片resize到网络输入大小
        img = F.interpolate(img.unsqueeze(0), size=(cfg.input_size, cfg.input_size), mode="nearest").squeeze(0)
        inp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        id_box = ' '.join(path_id_box)
        id_box = np.fromstring(id_box, dtype=np.float32, sep=' ').reshape(-1, 5)
        # 创建尺寸一致的数据掩膜
        neat_id = np.zeros(shape=20, dtype=int)
        neat_box = np.zeros(shape=(20, 4), dtype=np.float32)
        # 将真实鼠标移植到掩膜上
        neat_id[:id_box.shape[0]] = id_box[:, 0]
        # 将再不规则尺寸下的坐标转换为网络输入尺寸下的坐标形式,方便后面计算mAP
        neat_box[:id_box.shape[0],:] = (id_box[:, 1:]+pad)/max_side*cfg.input_size
        return inp, neat_box, neat_id
