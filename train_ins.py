import os
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from datasets.dataset_cen_train import Ins_Dataset
from datasets.dataset_patch_train import Train_Dataset
from models.vnet_ins_seg import VNet_singleTooth
from utils.utils import dice_coeff_all, dice_coeff
from utils import common
from evaluate import for_mine_patches
import sys
from utils.logger import Print_Logger
import torch.nn.functional as F
# from apex import amp
from utils.loss import dice_loss

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
    n_labels = 2  # 33
    model = VNet_singleTooth(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'ins-{model_name}-{type_time}-log.log')
    criterion1 = nn.CrossEntropyLoss().to(device)

    optim = optim.Adam(model.parameters(), lr=lr)
    # model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Ins_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(epochs):
        # common.adjust_learning_rate(optim, epoch, lr)
        model.train()

        for step, (_, vol, seg) in enumerate(train_loader):
            vol = vol.to(device)
            seg = seg.squeeze(0)
            # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
            seg = seg.to(device)
            optim.zero_grad()
            # img_patch1 = vol[:, :10]
            # img_patch2 = vol[:, 10:20]
            # img_patch3 = vol[:, 20:]
            # seg_patch1 = model(img_patch1)
            # seg_patch2 = model(img_patch2)
            # seg_patch3 = model(img_patch3)
            # seg_patch = torch.cat((seg_patch1, seg_patch2), 0)
            # pred = torch.cat((seg_patch, seg_patch3), 0)
            pred = model(vol)
            # pred = F.softmax(pred, dim=1)
            loss_seg = F.cross_entropy(pred, seg)
            outputs_soft = F.softmax(pred, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], seg == 1)
            loss_value = 0.5 * (loss_seg + loss_seg_dice)
            # loss_value = criterion1(pred, seg)
            pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            # pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)

            # with amp.scale_loss(loss_value, optim) as scaled_loss:
            #     scaled_loss.backward()
            loss_value.backward()
            optim.step()
            print(
                "Epoch :{} ,Step :{}, LR: {} ,loss :{:.3f} ,dice:{}".format(
                    epoch, step, lr,
                    loss_value.item(),
                    np.around(dice_coeff_all(
                        pred_img,
                        seg.cpu()).numpy(), 4)))
        print('=' * 12 + "Test" + '=' * 12)
        for_mine_patches(model, device, n_labels)
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('output',
                                    'ins_{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
