import os

import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Train_Attn_Dataset
from models.attention_unet import AttentionUNet
from utils.utils import dice_coeff
from utils import common
from evaluate import centroid_offset_mine, for_mine_attn
import sys
from utils.logger import New_Print_Logger
import torch.nn.functional as F
from apex import amp

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
    n_labels = 33  # Dataset部分需要做修改
    model = AttentionUNet(in_channel=1, num_class=n_labels).to(device)

    attn_checkpoint = torch.load("../output/AttentionUNet_2022-05-21_12:58:15_epoch_399.pth", map_location='cpu')
    model.load_state_dict(attn_checkpoint)
    start_epoch=400

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = New_Print_Logger(filename=f'{model_name}-{type_time}-log.log')
    criterion1 = nn.CrossEntropyLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Train_Attn_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(start_epoch,max_epochs):
        model.train()

        for step, (_, vol, seg) in enumerate(train_loader):
            vol = vol.to(device)
            seg = seg.to(device)

            optim.zero_grad()
            pred = model(vol)

            loss_value = criterion1(pred, seg)

            pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
            seg = common.to_one_hot_3d(seg.cpu(), n_classes=n_labels)
            with amp.scale_loss(loss_value, optim) as scaled_loss:
                scaled_loss.backward()
            # loss_value.backward()
            optim.step()
            print(
                "Epoch :{} ,Step :{}, LR: {},loss :{:.3f}\n dice:{}".format(epoch, step, lr, loss_value.item(),
                                                                            np.around(
                                                                                dice_coeff(pred_img, seg).numpy(), 4)))
            print('=' * 26)
        print('=' * 12 + "Test" + '=' * 12)
        for_mine_attn(model, device, n_labels,datapath="../data")
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('../output',
                                    '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
