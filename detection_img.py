import cv2
import numpy as np
import torch
from model import CenterNet
from config import cfg
import colorsys
from utils import post_process

class CtdetDetector():
    def __init__(self):
        print('\n正在创建模型...')

    def process(self, images):
        hm, wh, reg = self.model(images)
        # 通过最后三层得到[bbox、置信度、类别]
        dets = post_process(hm, wh, reg, cfg.max_object)
        return dets

    def run(self, image_path):

        # 初步预处理
        image = cv2.imread(image_path)
        temp_image = ((image / 255. - cfg.mean) / cfg.std).astype(np.float32)
        temp_image = temp_image.transpose(2, 0, 1)[np.newaxis, :]
        images = torch.from_numpy(temp_image)

        dets = self.process(images)


# 防止文件夹里还有别的文件，以至于报错
image_ext = ['jpg', 'jpeg', 'png', 'webp']

if __name__ == '__main__':
    model = CenterNet(cfg.res_name, cfg.classes, from_zero=True)
    model.load_state_dict(torch.load('.'))
    model.eval()
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(cfg.class_name), 1., 1.) for x in range(len(cfg.class_name))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    with torch.no_grad():
        pass