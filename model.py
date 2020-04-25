import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
resnet_spec = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # ResNet18及ResNet34中的Block是由 两个3*3的conv组成
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        # ResNet50及ResNet101、ResNet152中的block是由 1*1的conv + 3*3的conv + 1*1的conv组成
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_c, out_c)
        self.bn1 = norm_layer(out_c)
        self.conv2 = conv3x3(out_c, out_c, stride)
        self.bn2 = norm_layer(out_c)
        self.conv3 = conv1x1(out_c, out_c * self.expansion)
        self.bn3 = norm_layer(out_c * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CenterNet(nn.Module):
    def __init__(self, res_name, num_cls, load_resnet, head_conv=64):
        super(CenterNet, self).__init__()
        self.heads = {'hm': num_cls, 'wh': 2, 'reg': 2}

        resnets = {
            'resnet18': [BasicBlock, [2, 2, 2, 2]],
            'resnet34': [BasicBlock, [3, 4, 6, 3]],
            'resnet50': [Bottleneck, [3, 4, 6, 3]],
            'resnet101': [Bottleneck, [3, 4, 23, 3]],
            'resnet152': [Bottleneck, [3, 8, 36, 3]],
        }
        # 获取res_name类型下的 block类型和各层重复的次数
        block = resnets[res_name][0]
        layers = resnets[res_name][1]
        self.res_name = res_name

        # 这个in_c代表的是每个Block中第一个conv的输入维度
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 不管是多少层的ResNet其中每个Block的输入维度都是相同的 并且有规律翻倍 64 128 256 512
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 三次上采样
        self.deconv_layers = self._make_deconv_layer()

        # heatmap层
        self.hm = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, num_cls, kernel_size=1))
        # wh层
        self.wh = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1))
        # 偏移值回归层
        self.reg = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 2, kernel_size=1))
        if not load_resnet:
            self.init_weights(res_name)

    # ResNet中Block的构建
    def _make_layer(self, block, out_c, num_block, stride=1):
        downsample = None
        # 这里面其实除了第一个Block的stride为1,其他都为2. or 后面的条件是为了确保Block内开始和结束的通道数不一致的情况下才会触发
        # 因为不管ResNet多少层,其中的Block的开始和结束通道都为 in_c 和 out_c*expansion 只不过18和34的expansion为1,其他的为4而已
        # 所以ResNet18、34 第一个Block都是没有downsample的,对于这两个网络来说stride != 1 是完全可以应付的
        # 而其他的ResNet网络则需要第二个条件判断,因为它们在第一个Block有维度的变化(*4)
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_c, out_c * block.expansion, stride),
                nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        # 将每个Block的末尾conv输出通道数传给下一个Block的起始conv的输入维度
        self.in_c = out_c * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_c, out_c, ))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            planes = 256
            layers.append(nn.ConvTranspose2d(self.in_c, planes, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_c = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        hm = self.hm(x).sigmoid_()
        wh = self.wh(x)
        reg = self.reg(x)

        return hm, wh, reg

    # 初始化转置卷积和BN
    def init_weights(self, res_name):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化一下三个输出层
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)

        # 加载resnet预训练权重,model_dir为指定下载路径 .为当前路径,如果当前路径没有该权重则自动下载
        url = model_urls[res_name]
        resnet_state_dict = model_zoo.load_url(url, model_dir='.')
        print('\n特征提取网络{}已加载'.format(res_name))
        self.load_state_dict(resnet_state_dict, strict=False)  # strict=False会自动忽略resnet以外的权重加载
