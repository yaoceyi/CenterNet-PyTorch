参考:https://github.com/lz-pku-1997/easy-pytorch-CenterNet

官方代码地址:https://github.com/xingyizhou/CenterNet

# 使用方法
train.py进行训练

如需更改配置进入config.py

主干网络使用的是resnet系列,没有DCN也不需要任何编译.且在Window环境下.

使用VOC数据集格式进行训练的,对数据准备部分进行了修改.

添加了mAP计算,但是好像中间出了某些问题导致无法正常计算.但是loss是正常下降的

mAP问题以及其他一些功能有时间就会解决.