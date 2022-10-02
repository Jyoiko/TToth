import os

import numpy as np
import torch
import time
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Train_Cen_Dataset, Train_Miss_Cen_Dataset
from models.resnet import ResNet
from models.resnet_MME import ResNet_MME
from utils.utils import dice_coeff, precision
from utils import common
from evaluate import miss_mine
import sys
from utils.logger import New_Print_Logger
import torch.nn.functional as F
from apex import amp
from utils.loss import DiceLoss,ENLoss

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
    batch_size = 4
    n_labels = 2  # 33  # Dataset部分需要做修改
    model = ResNet_MME(n_classes=32).to(device)

    start_epoch = 0  # 400

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = New_Print_Logger(filename=f'{model_name}-{type_time}-log.log')

    criterion_miss = ENLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    sche=StepLR(optim,100,0.1)
    print("=" * 20)
    trainset = Train_Miss_Cen_Dataset()

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=False)

    print("Start Training...")
    for epoch in range(start_epoch, max_epochs):
        model.train()
        # common.adjust_learning_rate(optim, epoch, lr)
        # lr = lr * (0.1 ** (epoch // 100))
        # for param_group in optim.param_groups:
        #     param_group['lr'] = lr
        for step, (_, vol, miss) in enumerate(train_loader):
            vol = vol.to(device)
            miss = miss.to(device)

            optim.zero_grad()
            pred = model(vol)

            loss_value = criterion_miss(pred, miss)
            loss_value.backward()
            optim.step()

            print("Epoch :{} ,Step :{}/[batch_size: {}], LR: {} loss :{:.3f} Precision :{:.3f}".format(epoch, step, batch_size, lr, loss_value.item(), precision(pred, miss).cpu().numpy()))
        sche.step()
        print('=' * 12 + "Test" + '=' * 12)
        miss_mine(model, device, datapath="../data")
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('../output',
                                    '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
