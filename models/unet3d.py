import torch.nn as nn
import torch.nn.functional as F
import math
import torch


# adapt from https://github.com/MIC-DKFZ/BraTS2017

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        # self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, padding=1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel, 1, padding, bias=False)
        self.bn1 = normalization(out_channel, norm)

    def forward(self, x):
        x = self.bn1(self.conv(x))
        return x


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = ConvBlock(inplanes, planes, 3)

        self.conv2 = ConvBlock(planes, planes, 3)

        self.conv3 = ConvBlock(planes, planes, 3)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.conv1(x)
        y = self.relu(self.conv2(x))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.conv3(x)
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = ConvBlock(2 * planes, planes, 3)

        self.conv2 = ConvBlock(planes, planes // 2, 1, 0)

        self.conv3 = ConvBlock(planes, planes, 3, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.conv1(x))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.conv2(y))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.conv3(y))

        return y


class Unet(nn.Module):
    def __init__(self, c=1, n=16, dropout=0.5, norm='gn', num_classes=2):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c, n, dropout, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, )
        self.convd3 = ConvD(2 * n, 4 * n, dropout)
        self.convd4 = ConvD(4 * n, 8 * n, dropout)
        self.convd5 = ConvD(8 * n, 16 * n, dropout)

        self.convu4 = ConvU(16 * n, True)
        self.convu3 = ConvU(8 * n)
        self.convu2 = ConvU(4 * n)
        self.convu1 = ConvU(2 * n)

        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)

        return y1


# import torch
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# cuda0 = torch.device('cuda:0')
# x = torch.rand((2, 4, 32, 32, 32), device=cuda0)
# model = Unet()
# model.cuda()
# y = model(x)
# print(y.shape)
if __name__ == '__main__':
    model = Unet()
    name=str(model)
    index=name.index('(')
    print(name[:name.index('(')])
