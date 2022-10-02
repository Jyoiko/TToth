import os

import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Cen_Dataset
from models.posenet import PoseNet
from models.resnet import ResNet
from models.attention_unet import AttentionUNet
from models.vnet_cui import VNet_cui
from models.vnet_res import VNet_res
from utils.utils import dice_coeff
from utils import common
from evaluate import centroid_mine
import sys
from utils.logger import New_Print_Logger
import torch.nn.functional as F
from apex import amp
from utils.loss import FocalLoss, HeatMSELoss

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 1e-4
    max_epochs = 1000
    batch_size = 1
    # n_labels = 2  # 33  # Dataset部分需要做修改
    # model = VNet_res(n_channels=1, n_classes=32, normalization='batchnorm', has_dropout=True).to(device)
    model = PoseNet(inp_dim=1, oup_dim=32).to(device)

    # attn_checkpoint = torch.load("../output/AttentionUNet_2022-05-21_12:58:15_epoch_399.pth", map_location='cpu')
    # model.load_state_dict(attn_checkpoint)
    start_epoch = 0  # 400

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = New_Print_Logger(filename=f'{model_name}-{type_time}-log.log')

    criterion_l1 = FocalLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    # model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Cen_Dataset()
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(start_epoch, max_epochs):
        model.train()
        # common.adjust_learning_rate(optim, epoch, lr)
        for step, (_, vol, heatmap) in enumerate(train_loader):
            vol = vol.to(device)
            heatmap = heatmap.to(device)
            optim.zero_grad()
            pred = model(vol)
            loss1 = criterion_l1(pred[:, 0], heatmap)
            loss2 = criterion_l1(pred[:, 1], heatmap)
            loss_value = 0.5 * (loss1 + loss2)
            loss_value.backward()
            optim.step()

            print(
                f"Epoch :{epoch} ,Step :{step}/[batch_size: {batch_size}], LR: {lr} loss1:{loss1} loss2:{loss2} loss :{loss_value.item()} ")

        print('=' * 12 + "Test" + '=' * 12)
        centroid_mine(model, device, datapath="../data")
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('../output',
                                    '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
