import os

import numpy as np
import torch
import time
from torch import nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Attn_Dataset
# from models.unetr import UNETR
from models.unetr_official import UNETR_Official
from models.TransUnet import TransUnet
from models.attention_unet import AttentionUNet
from utils.utils import dice_coeff
from utils import common
from evaluate import centroid_mine, for_mine_attn
import sys
from utils.logger import New_Print_Logger, Print_Logger
import torch.nn.functional as F
from apex import amp
from utils.loss import DiceLoss, dice_loss


def train(model, train_loader, optim, device, epoch):
    model.train()
    for step, (idx, vol, seg, _) in enumerate(train_loader):
        vol = vol.to(device)
        _seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        seg = seg.to(device)
        # cen_off = cen_off.to(device)
        optim.zero_grad()
        pred_seg = model(vol)

        loss_seg = F.cross_entropy(pred_seg, seg)
        outputs_soft = F.softmax(pred_seg, dim=1)
        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], seg == 1)
        # loss_off = criterion3(pred_off[:, :, seg[0, :, :, :] > 0], cen_off[:, :, seg[0, :, :, :] > 0])
        loss_value = 0.5 * (loss_seg + loss_seg_dice)
        pred_img = torch.argmax(pred_seg.detach().cpu(), dim=1)
        pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)

        # with amp.scale_loss(loss_value, optim) as scaled_loss:
        #     scaled_loss.backward()
        loss_value.backward()
        optim.step()
        print(
            f"Epoch :{epoch} ,Step :{step},Index:{idx[0]}, LR: {lr}, CE Loss:{loss_seg.item():.3f} loss_seg_dice:{loss_seg_dice:.3f} "
            f"Loss:{loss_value.item():.3f} \n Dice:{np.around(dice_coeff(pred_img, _seg).numpy(), 4)}")


def validation(model, valid_loader, device):
    for step, (index, img, seg, cen_off) in enumerate(valid_loader):
        img, seg = img.to(device), seg.long()
        # cen_off = cen_off.to(device)
        pred_seg = model(img)
        # loss_off = criterion2(pred_off[:, :, seg[0, :, :, :] == 1], cen_off[:, :, seg[0, :, :, :] == 1])
        _seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred_seg = pred_seg.detach().cpu()
        seg_pred = torch.argmax(pred_seg, dim=1)
        pred_img = common.to_one_hot_3d(seg_pred, n_classes=n_labels)
        # print(f"off_loss:{loss_off.item():.3f}")
        print(f"{index[0]} dice: {np.around(dice_coeff(pred_img, _seg), 4)}")


if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 1e-4
    max_epochs = 400
    n_labels = 2
    model = UNETR_Official(in_channels=1, out_channels=2, img_size=(128, 128, 128), pos_embed='conv',
                           dropout_rate=0.2).to(device)
    start_epoch = 0
    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'cross_fold-{model_name}-{type_time}-log.log')
    criterion3 = nn.L1Loss().to(device)

    print("=" * 20)
    dataset = Attn_Dataset(datapath="data")
    total_size = len(dataset)
    kfold = 5
    seg_size = int(total_size / kfold)
    for i in range(kfold):
        print("=" * 12 + f"{i}th CrossFold" + "=" * 12)
        trll = 0
        trlr = i * seg_size
        vall = trlr
        valr = i * seg_size + seg_size
        trrl = valr
        trrr = total_size
        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))
        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
        train_loader = DataLoader(train_set, batch_size=1, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4)
        model = UNETR_Official(in_channels=1, out_channels=2, img_size=(128, 128, 128), pos_embed='conv',
                               dropout_rate=0.2).to(device)
        model.apply(model.reset_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        print("Start Training...")
        for epoch in range(start_epoch, max_epochs):
            train(model, train_loader, optimizer, device, epoch)
            print('=' * 12 + "Test" + '=' * 12)
            validation(model, val_loader, device)
            print('=' * 26)
