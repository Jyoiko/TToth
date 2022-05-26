import os
import torch
import time
import numpy as np
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Train_Dataset, Train_Attn_Dataset
from models.vnet_cui import VNet_cui
from utils.utils import dice_coeff
from utils import common
from evaluate import for_mine
import sys
from utils.logger import Print_Logger
import torch.nn.functional as F
from apex import amp

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 1e-4
    epochs = 1000
    n_labels = 33
    model = VNet_cui(n_channels=1, n_classes=n_labels, normalization='batchnorm', has_dropout=True).to(device)
    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'just_cen_off-{model_name}-{type_time}-log.log')
    criterion2 = nn.CrossEntropyLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Train_Attn_Dataset(datapath="data")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(epochs):
        model.train()

        for step, (_, vol, seg) in enumerate(train_loader):
            vol = vol.to(device)
            seg = seg.to(device)
            # cen_map = cen_map.to(device)
            optim.zero_grad()
            pred = model(vol)
            loss_value = criterion2(pred, seg)
            pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
            seg = common.to_one_hot_3d(seg.cpu(), n_classes=n_labels)

            with amp.scale_loss(loss_value, optim) as scaled_loss:
                scaled_loss.backward()
            # loss_value.backward()
            optim.step()
            print(
                "Epoch :{} ,Step :{}, LR: {}, CE Loss:{:.3f} \n Dice:{}".format(epoch, step, lr, loss_value.item(),
                                                                                np.around(dice_coeff(pred_img,
                                                                                          seg).numpy(), 4)))
        for_mine(model, device, n_labels)
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('output',
                                    'just_cen_off_{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
