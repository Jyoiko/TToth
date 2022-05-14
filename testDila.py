import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from datasets.dataset_cen_train import Train_Dataset
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils import common
import torch.nn.functional as F
import os
from models.DMFNet_16x import MFunit, DilatedConv3DBlock, DMFNet, DMFUnit


def passthrough(x, **kwargs):
    return x


class Net(nn.Module):
    def __init__(self, c=1, channels=16, num_class=2, groups=1, norm='bn'):
        super(Net, self).__init__()

        self.encoder_block1 = nn.Conv3d(c, channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.encoder_block2 = DilatedConv3DBlock(num_in=channels, num_out=channels, stride=2, kernel_size=(3, 3, 3),
                                                 d=(3, 3, 3))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2, channels, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2, channels, g=groups, stride=1, norm=norm)
        self.seg = nn.Conv3d(channels, num_class, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.encoder_block1(x)  # H//2 down
        x2 = self.encoder_block2(x1)  # H//4 down
        # Decoder
        y1 = self.upsample1(x2)  # H//8
        y1 = torch.cat([x1, y1], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4
        y2 = self.seg(y2)
        y2 = self.softmax(y2)
        return y2


class OutputTransition(nn.Module):
    def __init__(self, inChans, ):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.softmax(out, dim=1)

        return out


class ArtiNet(nn.Module):

    def __init__(self, c=1, channels=16, num_class=2, groups=1, norm='bn'):
        super(ArtiNet, self).__init__()
        self.encoder_block1 = nn.Conv3d(c, channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.encoder_block2 = nn.Conv3d(channels, channels, kernel_size=3, padding=3, stride=1, dilation=3, bias=False)
        # self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_conv = nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2)
        self.out = OutputTransition(channels//2)

    def forward(self, x):
        out = self.encoder_block1(x)
        out = self.encoder_block2(out)
        out = self.up_conv(out)
        out = self.out(out)
        return out


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    # cudnn.deterministic = True
    # model = Net(1).to(device)
    model = ArtiNet().to(device)
    trainset = Train_Dataset()
    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=False)
    criterion1 = nn.CrossEntropyLoss().to(device)
    optim = optim.Adam(model.parameters(), lr=1e-4)
    n_labels = 2

    # for step, (vol, seg) in enumerate(train_loader):
    loader = iter(train_loader)
    vol, seg = next(loader)
    vol, seg = vol.float(), seg.long()
    # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
    vol = vol.to(device)
    seg = seg.to(device)
    optim.zero_grad()
    pred = model(vol)
    loss_value = criterion1(pred, seg)
    loss_value.backward()
    optim.step()

    print("Step :{}, loss :{:.3f} ".format(0, loss_value.item(), ))
