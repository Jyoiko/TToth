import os
import torch
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
# from datasets.dataset_lits_train import Train_Dataset
from datasets.dataset import TrainDataset
from models.vnet4dout import VNet
from utils.utils import dice_coeff
from models.UNet import UNet
from models.unet3d import Unet
from utils import loss, common, metrics
from utils.loss import HybridLoss

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.empty_cache()
    torch.manual_seed(123)

    lr = 1e-4
    epochs = 100
    # model = VNet(elu=False, nll=False).to(device)
    n_labels = 2  # 33
    model = UNet(out_channel=n_labels).to(device)
    criterion1 = loss.DiceLoss().to(device)  # nn.CrossEntropyLoss().to(device)
    # criterion2 = nn.SmoothL1Loss().to(device)
    # loss = loss.DiceLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    alpha = 0.4
    trainset = TrainDataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=False)
    time = time.time()

    # testloader
    print("Start Training...")
    for epoch in range(epochs):
        common.adjust_learning_rate(optim, epoch, lr)
        model.train()
        # train_dice = metrics.DiceAverage(n_labels)
        for step, (index, vol, seg) in enumerate(train_loader):
            print("Index:", index)
            vol, seg = vol.float(), seg.long()
            seg = common.to_one_hot_3d(seg, n_classes=n_labels)
            # print(torch.sum(seg, (0, 2, 3, 4)))
            vol = vol.to(device)
            seg = seg.to(device)

            # seg = seg.view(-1, 2)
            # cen = cen.to(device)

            optim.zero_grad()
            pred = model(vol)
            # print(pred.dtype,seg.dtype)

            # loss0 = loss(pred[0], seg)
            # loss1 = loss(pred[1], seg)
            # loss2 = loss(pred[2], seg)
            loss_value = criterion1(pred, seg)
            # print("Test voxel: ", pred[0].detach().cpu().numpy())
            # loss_value = loss3 #+ alpha * (loss0 + loss2 + loss1)
            # loss2 = criterion2(pred2, cen)
            # loss = loss2 * 10 + loss1

            # print('loss: ',loss)

            loss_value.backward()
            optim.step()
            # train_dice.update(pred, seg)

            print("Epoch :{} ,Step :{}, loss :{:.3f} , dice:{} ".format(epoch, step, loss_value.item(),
                                                                        dice_coeff(pred, seg).detach().cpu().numpy()))
        # print("Train Dice Avg: ", train_dice.avg)

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join('output', '{}_epoch_{}.pth'.format(time, epoch)))
