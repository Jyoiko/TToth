import torch
import torch.nn.functional as F
import torch.nn as nn
from models.VanBlock import Block
__all__ = ['UNext']

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
import math


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv1 = DWConv(hidden_features)
        self.dwconv2 = DWConv(hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W, D):
        # pdb.set_trace()
        B, N, C = x.shape  # N=H*W*D C=embedding

        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv1(x, H, W, D)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc3(x_shift_c)

        x = self.dwconv2(x, H, W, D)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x_shift_d = x_s.transpose(1, 2)

        x = self.fc2(x_shift_d)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W, D


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, in_channels=1, embed_dims=[128, 160, 256], drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super().__init__()

        self.encoder1 = nn.Conv3d(in_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm3d(16)
        self.ebn2 = nn.BatchNorm3d(32)
        self.ebn3 = nn.BatchNorm3d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [shiftedBlock(dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.block2 = nn.ModuleList(
            [shiftedBlock(dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.dblock1 = nn.ModuleList(
            [shiftedBlock(dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.dblock2 = nn.ModuleList(
            [shiftedBlock(dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv3d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(160)
        self.dbn2 = nn.BatchNorm3d(128)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)

        self.seg = nn.Conv3d(16, num_classes, kernel_size=1)
        self.off = nn.Conv3d(16, 3, kernel_size=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W, D = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W, D)
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W, D = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W, D)
        out = self.norm4(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))

        out = torch.add(out, t4)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W, D)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W, D)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))

        return self.seg(out), self.off(out)


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


class UNext_S(nn.Module):
    def __init__(self, num_classes, in_channels=1, embed_dims=[128, 160, 256], drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super().__init__()

        self.encoder1 = nn.Conv3d(in_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(32, 128, 3, stride=1, padding=1)
        #  channels:[16 32 128 160 256]
        self.ag1 = AttentionGate(16, 32, 16)
        self.ag2 = AttentionGate(32, embed_dims[0], 32)
        self.ag3 = AttentionGate(embed_dims[0], embed_dims[1], embed_dims[0])
        self.ag4 = AttentionGate(embed_dims[1], embed_dims[2], embed_dims[1])

        self.con1 = Concat(32, 16)
        self.con2 = Concat(64, 32)
        self.con3 = Concat(embed_dims[0]*2, embed_dims[0])
        self.con4 = Concat(embed_dims[1]*2, embed_dims[1])

        self.ebn1 = nn.BatchNorm3d(16)
        self.ebn2 = nn.BatchNorm3d(32)
        self.ebn3 = nn.BatchNorm3d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [Block(dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])])

        self.block2 = nn.ModuleList(
            [Block(dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1])])

        self.dblock1 = nn.ModuleList(
            [Block(dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])])

        self.dblock2 = nn.ModuleList(
            [Block(dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1])])

        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv3d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(160)
        self.dbn2 = nn.BatchNorm3d(128)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)

        self.seg = nn.Conv3d(16, num_classes, kernel_size=1)
        self.off = nn.Conv3d(16, 3, kernel_size=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W, D = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W, D)
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W, D = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W, D)
        out = self.norm4(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### Stage 4
        d1 = self.ag4(t4, out)
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))

        out = self.con4(d1, out)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W, D)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        d2 = self.ag3(t3, out)
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.con3(out, d2)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W, D)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        d3 = self.ag2(t2, out)
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.con2(out, d3)

        d4 = self.ag1(t1, out)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.con1(out, d4)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))

        return self.seg(out), self.off(out)
