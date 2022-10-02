import os

import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
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

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 1e-4
    max_epochs = 1000
    n_labels = 2  # Dataset部分需要做修改
    configs = {
        'hidden_dim': 768,
        'input_size': (32, 32, 32),  # (64, 64, 64),
        'patch_size': (4, 4, 4),
        'trans_layers': 12,
        'mlp_dim': 512,
        'embedding_dprate': 0.1,
        'head_num': 12,
        'emb_in_channels': 256,
        'mlp_dprate': 0.2,
        'n_class': 2,
        'last_in_ch': 64,
        'n_skip': 3,
        'head_channels': 256,
        'decoder_channels': [(256, 256, 256), (256, 128, 128), (128, 64, 64)]
    }
    # model = UNETR().to(device)
    model = UNETR_Official(in_channels=1, out_channels=2, img_size=(128, 128, 128), pos_embed='conv', dropout_rate=0.2).to(
        device)
    # model = AttentionUNet(in_channel=1, num_class=n_labels).to(device)

    # attn_checkpoint = torch.load("../output/AttentionUNet_2022-09-03_15:11:32_epoch_399.pth", map_location='cpu')
    # model.load_state_dict(attn_checkpoint)
    start_epoch = 0

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'{model_name}-{type_time}-log.log')

    criterion3 = nn.L1Loss().to(device)
    optim = optim.AdamW(model.parameters(), lr=lr)
    # model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Attn_Dataset(datapath="data")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(start_epoch, max_epochs):
        model.train()

        for step, (_, vol, seg, _) in enumerate(train_loader):
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
                "Epoch :{} ,Step :{}, LR: {}, CE Loss:{:.3f} loss_seg_dice:{:.3f}  Loss:{:.3f} \n "
                "Dice:{}".format(
                    epoch, step, lr, loss_seg.item(), loss_seg_dice, loss_value.item(),
                    np.around(dice_coeff(pred_img,
                                         _seg).numpy(), 4)))

        print('=' * 12 + "Test" + '=' * 12)
        if (epoch + 1) % 100 == 0:
            for_mine_attn(model, device, n_labels, all_test=True, datapath="data")
            torch.save(model.state_dict(),
                       os.path.join('output',
                                    '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
        else:
            for_mine_attn(model, device, n_labels, datapath="data")
        print('=' * 26)
