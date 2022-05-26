"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


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
