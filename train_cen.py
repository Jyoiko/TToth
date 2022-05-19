import os
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Train_Dataset
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
    n_labels = 2  # 33
    model = VNet_cui(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    sys.stdout = Print_Logger(filename=f'cen_off-{model_name}-{type_time}-log.log')
    criterion2 = nn.SmoothL1Loss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    model, optim = amp.initialize(model, optim, opt_level="O1")

    print("=" * 20)
    trainset = Train_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=False)

    print("Start Training...")
    for epoch in range(epochs):
        model.train()

        for step, (_, vol, _, cen_map) in enumerate(train_loader):

            vol = vol.to(device)
            # seg = seg.to(device)
            cen_map = cen_map.to(device)
            optim.zero_grad()
            _, pred_off = model(vol)
            loss2 = criterion2(pred_off, cen_map)
            # pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            # pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
            # seg = common.to_one_hot_3d(seg.cpu(), n_classes=n_labels)
            loss_value = loss2 * 10
            with amp.scale_loss(loss_value, optim) as scaled_loss:
                scaled_loss.backward()
            # loss_value.backward()
            optim.step()
            print(
                "Epoch :{} ,Step :{}, LR: {}, cen_SmoothL1Loss:{:.3f}".format(epoch, step, lr, loss2.item()))
            # print('=' * 12 + "Test" + '=' * 12)
            # for_mine(model, device, n_labels)
            print('=' * 26)
            if (epoch + 1) % 100 == 0:
                torch.save(model.state_dict(),
                           os.path.join('output',
                                        'cen_off_{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
