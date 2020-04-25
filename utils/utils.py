import torch
import numpy as np
import torch.nn.functional as F


def _gather_feat(feat, ind):
    # feat [2, 128*128, 2]
    # ind  [2, max_obj]
    dim = feat.size(2)
    # [2, max_obj] -> [2, max_obj, 1] -> [2, max_obj, 2]
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # 这一步就是获取预测的wh或reg在目标位置上(理论)的预测值,实际ind上面大概率会有几个0值,代表空出来的目标位置
    feat = feat.gather(1, ind)
    return feat


def _transpose_and_gather_feat(feat, ind):
    # feat: tensor([2, 2, 128, 128])
    # ind:  tensor([2, max_obj]),
    # 这里有一点奇怪,不使用contiguous方法居然也可以view成功
    feat = feat.permute(0, 2, 3, 1)  # [2, 2, 128, 128] -> [2, 128, 128, 2]
    feat = feat.reshape(feat.size(0), -1, feat.size(3))  # [2, 128, 128, 2] -> [2, 16384, 2]
    feat = _gather_feat(feat, ind)
    return feat


def pad_to_square(img, pad_value):
    # 该函数目的只是将img填充为边长为max(h,w)的正方形,resize的操作后面才执行
    c, h, w = img.shape
    max_side = max(h, w)
    dim_diff = np.abs(h - w)
    # 为了下面pad做准备
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 通过比较宽高大小,来生成不同的pad数据
    pad = (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)
    # 填充paddimg,
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad, max_side


def gaussian_radius(det_size, min_overlap=0.7):
    # 高斯半径: 通过给定的w,h计算出相应的半径
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius):
    """
    :param heatmap: 某一真实目标的热力图,实际上这里传入进来的heatmap是会在数据本身修改的
    :param center:  该真实目标的中心坐标 (int)
    :param radius:  热半径,该值是根据 真实物体的 w,h通过高斯半径(gaussian_radius)计算得出的
    :return:
    """
    diameter = 2 * radius + 1  # 例如半径是2时,直径是5. 这里+1的目的就是为了准备一个中心点
    # 以直径为边长,生成一个2维的 直径*直径 高斯分布图,具体就是在图的中心值为1,离中心越远,值越小
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 获取真实物体中心点坐标 该坐标已经经过padding处理以及除以4之后的坐标
    x, y = int(center[0]), int(center[1])
    # 热力图的尺寸,一般为输入图像尺寸的四分之一
    height, width = heatmap.shape
    # 这里进行min操作的目的在于防止真实物体的中心点过于靠近热力图边缘时,防止热半径超出热力图范围,
    # +1的操作是因为切片操作中取左不取右,所以要取到理想中的值,:右边的值需要+1
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # 获取热力图中真实物体周围热半径内(radius)的热力图,如果目标热半径超出热力图的话,则舍弃超出部分
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # 同理提取出对应的高斯分布,目标不在热力图边缘的正常情况下,masked_gaussian == gaussian
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    # 一般情况下 masked_gaussian.shape == masked_heatmap.shape
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # 如果某一个类的两个高斯分布重叠，重叠的点取最大值就行,这里的out返回值就是修改后的masked_heatmap
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    else:
        print('gauss', masked_gaussian.shape)
        print('heatmap', masked_heatmap.shape)
        exit()
    # return heatmap


def _nms(heat):
    hmax = torch.nn.functional.max_pool2d(heat, (3, 3), stride=1, padding=1)
    keep = (hmax == heat)
    return heat * keep


def _topk(scores, K):
    """
    该方法主要是挑选出score最大的K个的相关信息
    1.先在每个channel上选出前K个score及index
    2.再在所有channel上选出前K个score及index
    3.期间获取其他和ind有关的信息,如cls_id x,y坐标等
    4.通过步骤3中得到的index + 步骤2中得到的index得到所有channel上前K个score在h*w大小上的index,(h,w)上的x,y
    :param scores: (batch, num_cls, h, w)
    :param K:       int
    :return: topk_score->(batch, K)所有channel上的scores排名前K个的scores
             topk_cls  ->(batch, K)所有channel上的scores排名前K个的scores所属的cls_id
             topk_inds ->(batch, K)所有channel上的scores排名前K个的scores在(h*w)上的索引
             topk_ys   ->(batch, K)所有channel上的scores排名前K个的scores在(h,w)上的y坐标
             topk_xs   ->(batch, K)所有channel上的scores排名前K个的scores在(h,w)上的x坐标
    """
    batch, num_cls, h, w = scores.size()
    # view: (batch_size, num_cls, h, w) -> (batch_size, num_cls, h*w)
    # topk_scores与topk_inds            -> (batch_size, num_cls, K) 取每个类channel上概率最大的K个score及index
    # topk_inds 代表了在每张图片每个channel上 h*w 个score中最大的K个score的索引,即 topk_inds 取值区间在[0,h*w)
    topk_scores, topk_inds = torch.topk(scores.view(batch, num_cls, -1), K)
    # 这个topk_inds也隐藏了 类别层的信息，需要取余去掉类别层信息，得到x、y坐标
    topk_xs = (topk_inds % w).int().float()
    topk_ys = (topk_inds / w).int().float()

    # 在所有类channel上再整合一次，得到一张图片中前50个概率最大的score及index
    # topk_ind代表了在在每张图片上num_cls*K个score中最大的K个score的索引,即 topk_ind 取值区间在[0,num_cls*K)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # 前K个socre所属的类别索引,因为topk_ind在[0,num_cls*K)区间,所以它可以代表所属的类别
    topk_cls = (topk_ind / K).int()

    # 取得最终所需的前50个信息,注意这里是一层一层往上取索引的. 先通过 topk_ind 获取 所有channels上前K个索引
    # 再通过 topk_inds 获取有channels索引的前K个索引.topk_ys、topk_xs同理 不过源码这样写有些多余
    # topk_inds = _gather_feat(topk_inds.view(batch, -1,1), topk_ind).view(batch, K)
    # topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    topk_inds = topk_inds.view(batch, -1).gather(1, topk_ind)
    topk_xs = topk_xs.view(batch, -1).gather(1, topk_ind)
    topk_ys = topk_ys.view(batch, -1).gather(1, topk_ind)

    return topk_score, topk_cls, topk_inds, topk_xs, topk_ys


def post_process(hm, wh, offset, K):
    """
    对网络预测出的hm,wh,offset进行过滤处理
    :param hm:  (batch_size, num_cls, h,w)
    :param wh:  (batch_size, 2, h,w)
    :param offset: (batch_size, 2, h,w)
    :param K:   int
    :return:
    """
    # hm只留下比周围8个点都大的那些点
    hm = _nms(hm)

    # 得到每张图scores最高的前50个点的 scores、坐标信息、类别、坐标信息中的y坐标、坐标信息中的x坐标
    topk_score, topk_cls, topk_inds, topk_xs, topk_ys = _topk(hm, K=K)

    # 取得每张图前50名scores对应位置的偏移值
    offset = _transpose_and_gather_feat(offset, topk_inds)  # torch.Size([batch_size, K, 2])
    # 对基础x,y坐标加上偏移值
    x = topk_xs + offset[:, :, 0]
    y = topk_ys + offset[:, :, 1]

    wh = _transpose_and_gather_feat(wh, topk_inds)
    # 对预测的坐标转换为在网路输入尺寸下的坐标形式
    detections = torch.stack([(x - wh[..., 0] / 2) * 4,
                              (y - wh[..., 1] / 2) * 4,
                              (x + wh[..., 0] / 2) * 4,
                              (y + wh[..., 1] / 2) * 4,
                              topk_score,
                              topk_cls.float()], dim=2)
    return detections


def box_iou(box_a, box_b, eps=1e-5):
    # 计算 N个box与M个box的iou需要使用到torch与numpy的广播特性
    tl = np.maximum(box_a[..., :2], box_b[..., :2])
    br = np.minimum(box_a[..., 2:], box_b[..., 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(box_a[..., 2:] - box_a[..., :2], axis=2)
    area_b = np.prod(box_b[..., 2:] - box_b[..., :2], axis=2)
    iou = area_i / (area_a + area_b - area_i + eps)
    return iou
