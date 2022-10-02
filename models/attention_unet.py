import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1, padding=2):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class DilaConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(DilaConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=2, dilation=2),
            nn.Conv3d(out_channel, out_channel, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    """
    upsample, concat and conv
    """

    def __init__(self, in_channel, inter_channel, out_channel):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channel, inter_channel),
            nn.Upsample(scale_factor=2)
        )
        self.conv = ConvBlock(2 * inter_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out


class AttentionGate(nn.Module):
    """
    filter the features propagated through the skip connections
    """

    def __init__(self, in_channel, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv3d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv3d(in_channel, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv3d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.interpolate(out, size=x.size()[2:], mode='trilinear', align_corners=True)
        out = x * out
        return out


class AttentionUNet(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel, num_class, filters=[32, 64, 128, 256]):
        super(AttentionUNet, self).__init__()

        f1, f2, f3, f4 = filters
        f5 = 16
        # self.softmax = F.softmax

        self.down1 = ConvBlock(in_channel, f1)

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f1, f2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f2, f3)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f3, f4)
        )
        # self.down5 = nn.Sequential(
        #     nn.MaxPool3d(kernel_size=(2, 2, 2)),
        #     ConvBlock(f4, f4)
        # )
        # self.ag1 = AttentionGate(f3, f4, f3)
        # self.ag2 = AttentionGate(f2, f2, f2)
        # self.ag3 = AttentionGate(f1, f1, f1)
        #
        # self.up1 = Upsample(f4, f3, f2)
        # self.up2 = Upsample(f2, f2, f1)
        # self.up3 = Upsample(f1, f1, f5)
        self.ag1_seg = AttentionGate(f3, f4, f3)
        self.ag2_seg = AttentionGate(f2, f2, f2)
        self.ag3_seg = AttentionGate(f1, f1, f1)

        self.up1_seg = Upsample(f4, f3, f2)
        self.up2_seg = Upsample(f2, f2, f1)
        self.up3_seg = Upsample(f1, f1, f5)

        self.ag1_off = AttentionGate(f3, f4, f3)
        self.ag2_off = AttentionGate(f2, f2, f2)
        self.ag3_off = AttentionGate(f1, f1, f1)

        self.up1_off = Upsample(f4, f3, f2)
        self.up2_off = Upsample(f2, f2, f1)
        self.up3_off = Upsample(f1, f1, f5)

        self.out_seg = nn.Conv3d(f5, num_class, 1, padding=0)
        self.out_off = nn.Conv3d(f5, 3, 1, padding=0)
        # self.dropout = nn.Dropout3d(p=0.2, inplace=False)
        # self.apply(self.initialize_weights)

    def forward(self, x):  # 128
        down1 = self.down1(x)  # 128
        down2 = self.down2(down1)  # 64
        down3 = self.down3(down2)  # 32
        down4 = self.down4(down3)  # 16

        ag1_seg = self.ag1_seg(down3, down4)  # 32
        up1_seg = self.up1_seg(down4, ag1_seg)
        ag2_seg = self.ag2_seg(down2, up1_seg)
        up2_seg = self.up2_seg(up1_seg, ag2_seg)
        ag3_seg = self.ag3_seg(down1, up2_seg)
        up3_seg = self.up3_seg(up2_seg, ag3_seg)

        ag1_off = self.ag1_off(down3, down4)  # 32
        up1_off = self.up1_off(down4, ag1_off)
        ag2_off = self.ag2_off(down2, up1_off)
        up2_off = self.up2_off(up1_off, ag2_off)
        ag3_off = self.ag3_off(down1, up2_off)
        up3_off = self.up3_off(up2_off, ag3_off)

        off = self.out_off(up3_off)
        seg = self.out_seg(up3_seg)
        # ag1 = self.ag1(down3, down4)  # 32
        # up1 = self.up1(down4, ag1)
        # ag2 = self.ag2(down2, up1)
        # up2 = self.up2(up1, ag2)
        # ag3 = self.ag3(down1, up2)
        # up3 = self.up3(up2, ag3)
        # off = self.out_off(up3)
        # seg = self.out_seg(up3)
        # up3 = self.softmax(up3, dim=1)
        return seg, off

    # def initialize_weights(self, m):
    #     if isinstance(m, nn.Conv3d):
    #         torch.nn.init.xavier_normal_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.zero_()


if __name__ == '__main__':
    model = AttentionUNet(in_channel=1, num_class=2)
    print(model)
