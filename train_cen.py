import os
import torch
import time
import numpy as np
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Attn_Dataset
from models.archs import UNext, UNext_S
from models.attention_unet import AttentionUNet
from models.TransUnet import TransUnet
from models.vnet_cui import VNet_cui
from utils.loss import dice_loss, focal_loss
from utils.utils import dice_coeff
from utils import common
from evaluate import for_mine
import sys
from utils.logger import Print_Logger
import torch.nn.functional as F

# from apex import amp

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 1e-4
    epochs = 1000
    n_labels = 2
    model = AttentionUNet(in_channel=1, num_class=n_labels).to(device)
    # model = UNext_S(in_channels=1, num_classes=2).to(device)

    # model = VNet_cui(n_channels=1, n_classes=n_labels, normalization='batchnorm', has_dropout=True).to(device)
    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'just_cen_off-{model_name}-{type_time}-log.log')
    criterion2 = nn.SmoothL1Loss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    # model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Attn_Dataset(train_mode=True, datapath="data")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=False)

    print("Start Training...")
    for epoch in range(epochs):
        model.train()

        for step, (_, vol, seg, cen_off) in enumerate(train_loader):  # seg_sub1, seg_sub2,
            vol = vol.to(device)
            seg = seg.to(device)
            # seg_sub1 = seg_sub1.to(device)
            # seg_sub2 = seg_sub2.to(device)
            cen_off = cen_off.to(device)
            optim.zero_grad()
            pred_seg, pred_off = model(vol)
            # loss_seg1 = F.cross_entropy(pred_seg_sub1, seg_sub1)
            # loss_seg2 = F.cross_entropy(pred_seg_sub2, seg_sub2)
            loss_seg = F.cross_entropy(pred_seg, seg)
            outputs_soft = F.softmax(pred_seg, dim=1)
            # loss_seg = loss_seg3 + (loss_seg2 + loss_seg1) * 0.5
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], seg == 1)
            loss_off = criterion2(pred_off[:, :, seg[0, :, :, :] == 1], cen_off[:, :, seg[0, :, :, :] == 1])
            loss_value = 0.5 * (loss_seg + loss_seg_dice) + loss_off
            pred_img = torch.argmax(pred_seg.detach().cpu(), dim=1)
            pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
            seg = common.to_one_hot_3d(seg.cpu(), n_classes=n_labels)

            # with amp.scale_loss(loss_value, optim) as scaled_loss:
            #     scaled_loss.backward()
            loss_value.backward()
            optim.step()
            print(
                'Epoch :{} ,Step :{}, LR: {},  loss_off:{:.3f} , loss_dice:{:.3f}, loss_ce:{:.3f}, loss:{:.3f}  \n '
                'DICE:{} '.format(
                    epoch, step, lr, loss_off.item(), loss_seg_dice, loss_seg, loss_value.item(),
                    np.around(dice_coeff(pred_img,
                                         seg).numpy(), 4)))
        print('=' * 12 + "Test" + '=' * 12)
        for_mine(model, device, n_labels)
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('output',
                                    'just_cen_off_{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
