import torch
from utils.utils import _transpose_and_gather_feat

# 默认输入尺寸512*512
# max_obj为手动设置的(一张图片中出现的最多目标数)


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, ind, gt):
        """
        :param pred: tensor([2, 2, 128, 128])
        :param mask: tensor([2, max_obj]),
        :param ind:  tensor([2, max_obj]),
        :param gt:   tensor([2, max_obj, 2]),
        :return:
        """
        # 先利用ind获取wh或reg在目标位置上的预测结果,在利用目标实际位置的掩膜,确认有多少个目标.进而计算 loss
        pred = _transpose_and_gather_feat(pred, ind)
        # 需要扩维到和pred、gt相同的维度,才方便后续过滤操作
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # 对wh或reg进行loss计算时,取平均loss.即一对wh或reg有对应两个loss需要除以2.所以这里默认mask.sum()随着扩维翻倍了
        loss = torch.nn.functional.l1_loss(pred * mask, gt * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        对focal loss 进行了一些修改,结果与 CornerNet的完全相同但是速度会快一些,不过稍微会多占用一点显存
        :param pred: batch_size, 2, 128, 128
        :param gt:   batch_size, 2, 128, 128
        :return:
        """
        # 所有中心点
        pos_inds = gt.eq(1).float()
        # 所有非中心点
        neg_inds = gt.lt(1).float()

        # 在非中心点区域的权重系数 (1-gt)**β
        neg_weights = torch.pow(1 - gt, 4)

        # heatmap的损失 可参考 https://blog.csdn.net/baobei0112/article/details/94392343
        # 原帖见 https://zhuanlan.zhihu.com/p/66048276 但是排版没有上面那个舒服
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # loss要除以gt里的目标数，如果gt目标数为0，那就除以1 因为 log函数在 0~1区间内都为负数,所以在loss前加了负号
        # 有一点疑问,按理说求loss时应该计算均值,但是else下面只除以num_pos,虽然作者论文中也是这个意思.但是不太明白为什么这样做...
        if num_pos == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos

        return loss


class CenterLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = RegL1Loss()

    def forward(self, output, hm, true_mask, ind, wh, offset):
        """                 pred_hm                                 pred_wh                 pred_offset
        :param output: (tensor([2, num_cls, 128, 128]), tensor([2, 2, 128, 128]), tensor([2, 2, 128, 128]))
        :param hm:          tensor([2, num_cls, 128, 128]
        :param true_mask:   tensor([2, max_obj]
        :param ind:         tensor([2, max_obj]
        :param wh:          tensor([2, max_obj, 2]
        :param offset:      tensor([2, max_obj, 2]
        :return:
        """
        pred_hm, pred_wh, pred_off = output
        # 限制热力图的数值范围
        pred_hm = torch.clamp(pred_hm, min=1e-4, max=1 - 1e-4)
        # 构建hm_loss层、wh_loss层、off_loss层
        hm_loss = self.focal_loss(pred_hm, hm)

        wh_loss = self.l1_loss(pred_wh, true_mask, ind, wh)
        off_loss = self.l1_loss(pred_off, true_mask, ind, offset)

        total_loss = hm_loss + 0.1 * wh_loss + off_loss

        return total_loss, hm_loss, wh_loss, off_loss

