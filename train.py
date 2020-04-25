import torch
from model import CenterNet
from dataset import TrainDataset, EvalDataset
from utils.loss import CenterLoss
from config import cfg
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.eval import Eval
from terminaltables import AsciiTable
import visdom
import numpy as np


def save_model(path, epoch, model, optimizer):
    data = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(data, path)


if __name__ == '__main__':
    # 将会让程序在开始时花费一点额外时间,为整个网络的每个卷积层搜索最适合它的卷积实现算法,进而实现网络的加速.
    # 适用场景是网络结构和输入维度尺寸是固定的,反之将会导致程序不停地做优化,反而会耗费更多的时间.
    vis = visdom.Visdom(env='CenterNet')
    torch.backends.cudnn.benchmark = True
    print('\n正在初始化CenterNet网络及权重...')
    model = CenterNet(cfg.res_name, cfg.num_cls, load_resnet=cfg.load_model).cuda()
    if cfg.load_model:
        model.load_state_dict(torch.load(cfg.load_model))

    train_loader = DataLoader(TrainDataset(cfg.train_txt), batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(EvalDataset(cfg.val_txt), batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    loss_func = CenterLoss()
    mAP = 0
    lr = cfg.lr
    for epoch in range(1, cfg.num_epoch):
        # 根据训练的epoch进度调节lr
        if epoch in [20,30]:
            lr *= 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        model.train()
        for inp, hm, true_mask, ind, wh, offset in tqdm(train_loader):
            inp, hm, true_mask, ind, wh, offset = inp.cuda(), hm.cuda(), true_mask.cuda(), ind.cuda(), wh.cuda(), offset.cuda()
            outputs = model(inp)
            total_loss, hm_loss, wh_loss, off_loss = loss_func(outputs, hm, true_mask, ind, wh, offset)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # save_model(os.path.join(os.path.dirname(__file__), 'weights', '_{}.pth'.format(epoch)), epoch, model,
        #            optimizer)
        eval_result = Eval(model=model, test_loader=val_loader)
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        for p, r, ap, f1, cls_id in zip(*eval_result):
            ap_table += [[cls_id, cfg.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        print('\n' + AsciiTable(ap_table).table)
        eval_map = round(eval_result[2].mean(), 4)
        print("Epoch %d/%d ---- mAP:%.4f Loss:%.4f" % (epoch, cfg.num_epoch, eval_map, total_loss.item()))
        vis.line(X=np.array([epoch]), Y=np.array([hm_loss.item()]), win='hm', update=None if epoch == 1 else 'append',
                 opts={'title': 'hm'})
        vis.line(X=np.array([epoch]), Y=np.array([wh_loss.item()]), win='wh', update=None if epoch == 1 else 'append',
                 opts={'title': 'wh'})
        vis.line(X=np.array([epoch]), Y=np.array([off_loss.item()]), win='offset', update=None if epoch == 1 else 'append',
                 opts={'title': 'offset'})
        vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='map', update=None if epoch == 1 else 'append',
                 opts={'title': 'map'})
        if eval_map > mAP:
            mAP = eval_map
            torch.save(model.state_dict(), 'weights/map_%s.pt' % mAP)
