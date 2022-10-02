"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import common


def focal_loss(input, target):
    # gt = common.to_one_hot_3d(gt, n_classes=2).long().cuda()
    alpha = .25
    gamma = 2
    alpha = torch.tensor([alpha, 1 - alpha]).cuda()
    input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
    input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
    input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    target = target.view(-1, 1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.exp()
    at = alpha.gather(0, target.view(-1))
    loss = -1*at * (1 - pt) ** gamma * logpt
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes=32):

        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.theta = 1 / self.num_classes

    def _sigmoid(self, x):
        out = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)
        return out

    def _neg_loss(self, pred, gt):
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        loss = 0.0
        eps = 1e-5

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        # pos_pred[pos_pred < 0] = 0
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)

        # neg_pred[neg_pred > 1] = 1
        neg_loss_s1 = torch.log(1 - neg_pred)
        neg_loss_s2 = neg_loss_s1 * torch.pow(neg_pred, 2)
        neg_loss_s3 = neg_loss_s2 * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss_s3.sum()
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        if math.isnan(loss):
            print("num_pos: ", num_pos)
            print("pos_loss: ", pos_loss)
            print("neg_loss: ", neg_loss)
            print("neg_loss_s1: ", neg_loss_s1.sum())
            print("neg_loss_s2: ", neg_loss_s2.sum())
            # print("neg_loss_s3: ", neg_loss_s3)
            a = torch.where(neg_pred >= 1)
            print("neg_pred: ", neg_pred[a])
            sys.exit(1)
        return loss

    def forward(self, inputs, target):
        loss = 0.0
        for i in range(0, self.num_classes):
            focal = self._neg_loss(self._sigmoid(inputs[:, i]), target[:, i])
            loss += focal
        return loss / self.num_classes


class HeatMSELoss(nn.Module):
    def __init__(self, num_classes=32):
        super(HeatMSELoss, self).__init__()
        self.n_class = num_classes

    def forward(self, pred, gt):
        l = ((pred - gt) ** 2)
        l = l.mean(dim=2).mean(dim=2).mean(dim=2).sum()
        return l


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class ENLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def ova_loss(self, out_open, label):
        assert len(out_open.size()) == 3
        assert out_open.size(1) == 2

        out_open = F.softmax(out_open, 1)

        label_n = 1 - label
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * label, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                        1e-8) * label_n, 1)[0])
        return open_loss_pos, open_loss_neg

    def forward(self, inputs, target):
        open_loss_pos, open_loss_neg = self.ova_loss(inputs, target)
        return 0.5 * (open_loss_neg + open_loss_pos)


class CubeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance(self, pred, target):
        """
        (batch,32,3)
        :param pred:
        :param target:
        :return:
        """
        cha = pred - target
        mi = torch.pow(cha, 2)
        he = torch.sum(mi, dim=2)
        kaifang = torch.sqrt(he)
        return torch.mean(kaifang)
        # return torch.sqrt(torch.sum(torch.pow(pred - target, 2), dim=2))

    def forward(self, pred, target):
        return self.distance(pred, target)


class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(
                dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    smooth + target[:, i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                    0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                            (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class MultiDiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]
