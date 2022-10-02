import os

import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Attn_Dataset
from models.TransUnet import TransUnet
from utils.utils import dice_coeff
from utils import common
from evaluate import for_mine_attn
import sys
from utils.logger import New_Print_Logger
import torch.nn.functional as F
from apex import amp
from utils.loss import DiceLoss

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    lr = 4e-6
    max_epochs = 5000
    n_labels = 33  # Dataset部分需要做修改
    configs = {
        'hidden_dim': 768,
        'input_size': (48, 48, 48),  # (64, 64, 64),
        'patch_size': (4, 4, 4),
        'trans_layers': 12,
        'mlp_dim': 512,
        'embedding_dprate': 0.1,
        'head_num': 12,
        'emb_in_channels': 32,
        'mlp_dprate': 0.2,
        'n_class': 33,
        'last_in_ch': 16,
        'n_skip': 3,
        'head_channels': 128,
        'decoder_channels': [(128, 64, 32), (64, 32, 16), (32, 16, 8)]
    }
    model = TransUnet(configs).to(device)
    attn_checkpoint = torch.load("../output/TransUnet_2022-06-02_09:53:11_epoch_2099.pth", map_location='cpu')
    model.load_state_dict(attn_checkpoint)
    start_epoch = 2100  # 400

    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = New_Print_Logger(filename=f'{model_name}-{type_time}-log.log')
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = DiceLoss(n_classes=n_labels).to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    # model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Attn_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(start_epoch, max_epochs):
        model.train()
        # common.adjust_learning_rate(optim, epoch, lr)
        for step, (_, vol, seg) in enumerate(train_loader):
            vol = vol.to(device)
            _seg = common.to_one_hot_3d(seg, n_classes=n_labels)
            seg = seg.to(device)

            optim.zero_grad()
            pred = model(vol)

            loss1 = criterion1(pred, seg)
            outputs_soft = F.softmax(pred, dim=1)

            loss2 = criterion2(outputs_soft, seg)
            loss_value = 0.5 * (loss2 + loss1)
            pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)

            # with amp.scale_loss(loss_value, optim) as scaled_loss:
            #     scaled_loss.backward()
            loss_value.backward()
            optim.step()
            print(
                "Epoch :{} ,Step :{}, LR: {},CE Loss:{:.3f}, DiceLoss:{:.3f},loss :{:.3f}\n dice:{}".format(
                    epoch, step, lr, loss1.item(), loss2.item(), loss_value.item(),
                    np.around(
                        dice_coeff(pred_img, _seg).numpy(), 4)))
            print('=' * 26)
        print('=' * 12 + "Test" + '=' * 12)
        for_mine_attn(model, device, n_labels, datapath="../data")
        print('=' * 26)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join('../output',
                                    '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
