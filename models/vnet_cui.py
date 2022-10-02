import torch
from torch import nn
import torch.nn.functional as F
from models.VanBlock import Block, SpatialAttention


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DilaConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(DilaConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
                # ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            else:
                input_channel = n_filters_out
            ops.append(SpatialAttention(input_channel))
            # ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            # ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=3, padding=2, dilation=2))
            # ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in

            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


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


class Concat(nn.Module):
    """
     concat and conv
    """

    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1, padding=2):
        super(Concat, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out


class VNet_cui(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=True):
        super(VNet_cui, self).__init__()
        self.has_dropout = has_dropout
        # self.fcn = nn.Sequential(
        #     ConvBlock(2, n_channels, 32),
        #     nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0),
        #     nn.BatchNorm3d(64)
        # )
        # self.ag1 = AttentionGate(n_filters * 2, n_filters * 4, n_filters * 2)
        # self.ag2 = AttentionGate(n_filters * 4, n_filters * 8, n_filters * 4)
        # self.ag3 = AttentionGate(n_filters * 8, n_filters * 16, n_filters * 8)
        # self.con1 = Concat(n_filters * 4, n_filters * 2)
        # self.con2 = Concat(n_filters * 8, n_filters * 4)
        # self.con3 = Concat(n_filters * 16, n_filters * 8)
        # self.ag1 = DilaConvBlock(3, n_filters * 2, n_filters * 2, normalization=normalization)
        # self.ag2 = DilaConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        # self.ag3 = DilaConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        # self.ag4 = DilaConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.block_one = ConvBlock(1, n_channels, 8, normalization=normalization)
        # self.block_one = ConvBlock(1, 64, 8, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(8, n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters, 8, normalization=normalization)

        # self.out_conv_seg = nn.Conv3d(8, n_classes, 1, padding=0)
        # self.out_conv_sub1 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
        # self.out_conv_sub2 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
        self.out_conv_off = nn.Conv3d(8, 3, 1, padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = F.softmax
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.apply(self.initialize_weights)

    def encoder(self, input):
        # input = self.fcn(input)
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x2 = features[0]
        x3 = features[1]
        x4 = features[2]
        x5 = features[3]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out = self.block_eight_up(x8)

        # out_seg = self.out_conv_seg(out)
        out_off = self.out_conv_off(out)
        return out_off

    def forward(self, input):
        features = self.encoder(input)
        out_off = self.decoder(features)

    # if self.training is True:
        return out_off
    # else:
    #     return out_seg,  out_off
# def initialize_weights(self, m):
#     if isinstance(m, nn.Conv3d):
#         torch.nn.init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             m.bias.data.zero_()
