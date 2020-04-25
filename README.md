参考:https://github.com/lz-pku-1997/easy-pytorch-CenterNet

官方代码地址:https://github.com/xingyizhou/CenterNet

# 使用方法
train.py进行训练

detection_img.py利用文件夹中的图片进行测试模型效果

如需更改配置进入config.py

主干网络使用的是resnet系列,没有DCN也不需要任何编译.且在Window环境下.

使用VOC数据集格式进行训练的,对数据准备部分进行了修改.只要在data/kalete文件夹下中的Annotations准备xml文件,JPGImages准备JPG文件.
然后运行xml2txt.py即可.训练集:验证集=9:1
