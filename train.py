import os
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_lits_train import Train_Dataset
from datasets.dataset import TrainDataset
from models.vnet4dout import VNet
from models.vnet_dilation import DVNet
from utils.utils import dice_coeff
from models.UNet import UNet
from models.unet3d import Unet
from models.unet3d_dilated import DUnet
from models.DMFNet_16x import DMFNet
from utils import loss, common, metrics
from utils.loss import HybridLoss
from evaluate import test_for_mine
import sys
from utils.logger import Print_Logger

from apex import amp

if __name__ == '__main__':
    start = time.time()
    local_time = time.localtime()
    type_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)
    sys.stdout = Print_Logger(filename=f'{type_time}-log.log')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # torch.backends.cudnn.enabled = False
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)
    # device = torch.device('cuda:0')
    # torch.cuda.empty_cache()
    torch.manual_seed(123)

    lr = 1e-4
    epochs = 50
    # model = VNet(elu=False, nll=False).to(device)
    n_labels = 2  # 33
    model = DVNet(nll=False, elu=False).to(device)
    # model = DMFNet(c=1, num_classes=2).to(device)
    # model = DUnet().to(device)
    model_name = str(model)
    model_name = model_name[:model_name.index('(')]
    criterion1 = nn.CrossEntropyLoss().to(device)

    # criterion2 = nn.SmoothL1Loss().to(device)
    # loss = loss.DiceLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    model, optim = amp.initialize(model, optim, opt_level="O1")
    alpha = 0.4

    print("=" * 20)
    trainset = Train_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=False)

    # testloader

    print("Start Training...")
    for epoch in range(epochs):
        common.adjust_learning_rate(optim, epoch, lr)
        model.train()

        for step, (vol, seg) in enumerate(train_loader):
            vol, seg = vol.float(), seg.long()
            # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
            # print(torch.sum(seg, (0, 2, 3, 4)))
            vol = vol.to(device)
            seg = seg.to(device)

            # seg = seg.view(-1, 2)
            # cen = cen.to(device)

            optim.zero_grad()
            pred = model(vol)
            loss_value = criterion1(pred, seg)
            pred_img = torch.argmax(pred.detach().cpu(), dim=1)
            pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
            seg = common.to_one_hot_3d(seg.cpu(), n_classes=n_labels)

            # loss_value = loss3 #+ alpha * (loss0 + loss2 + loss1)
            # loss2 = criterion2(pred2, cen)
            # loss = loss2 * 10 + loss1

            # print('loss: ',loss)
            with amp.scale_loss(loss_value, optim) as scaled_loss:
                scaled_loss.backward()
            # loss_value.backward()
            optim.step()
            # train_dice.update(pred, seg)

            print("Epoch :{} ,Step :{}, loss :{:.3f} , dice:{} ".format(epoch, step, loss_value.item(),
                                                                        dice_coeff(pred_img, seg).numpy()))
        # print("Train Dice Avg: ", train_dice.avg)

        if (epoch + 1) % 1 == 0:

            test_for_mine(model,device,n_labels)
            # torch.save(model.state_dict(),
            #            os.path.join('output', '{}_{}_epoch_{}.pth'.format(model_name, type_time, epoch)))  # 打印模型名称
