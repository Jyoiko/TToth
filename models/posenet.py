import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
import torch.nn.functional as F

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, in_dim=1, oup_dim=32,nstack=2,  bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        inp_dim=256
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(in_dim, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack
        # self.softmax = F.softmax
    def forward(self, x):
        ## our posenet
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            # preds=self.softmax(preds,dim=1)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)
