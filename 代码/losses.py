# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:46
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : losses.py
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from LovaszSoftmax.pytorch import lovasz_losses as L


class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(
        self,
        weight=None,
        ignore_index=255,
        reduction="mean",
        thresh=0.6,
        min_kept=256,
        down_ratio=1,
    ):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio

        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, weight=weight, ignore_index=ignore_index
        )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)  ## ne: not equal
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Labels: {}".format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)  #
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # print('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, lb_ignore=255, ignore_simple_sample_factor=16):
        """
        Args:
            thresh: 阈值，超过该值则被算法简单样本 -> 不参与Loss计算
            lb_ignore: 忽略的像素值(一般255代表背景), 不参与损失的计算
            ignore_simple_sample_factor: 忽略简单样本的系数
                                                该系数越大，最少计算的像素点个数越少
                                                该系数越小，最少计算的像素点个数越多
        """
        super(OhemCELoss, self).__init__()

        """
            这里的 thresh 和 self.thresh 不是一回儿事儿
                ①预测概率 > thresh -> 简单样本
                ①预测概率 < thresh -> 困难样本
                ②损失值 > self.thresh -> 困难样本
                ②损失值 < self.thresh -> 简单

                ①和②其实是一回儿事儿，但 thresh 和 self.thresh 不是一回儿事儿
        """
        self.thresh = -torch.log(
            input=torch.tensor(thresh, requires_grad=False, dtype=torch.float)
        )
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction="none")
        self.ignore_simple_sample_factor = ignore_simple_sample_factor

        """
            reduction 参数用于控制损失的计算方式和输出形式。它有三种可选的取值：
                1. 'none'：当设置为 'none' 时，损失将会逐个样本计算，返回一个与输入张量相同形状的损失张量。
                           这意味着输出的损失张量的形状与输入的标签张量相同，每个位置对应一个样本的损失值。
                2. 'mean'：当设置为 'mean' 时，损失会对逐个样本计算的损失进行求均值，得到一个标量值。
                           即计算所有样本的损失值的平均值。
                3. 'sum' : 当设置为 'sum'  时，损失会对逐个样本计算的损失进行求和，得到一个标量值。
                           即计算所有样本的损失值的总和。

            在语义分割任务中，通常使用 ignore_index 参数来忽略某些特定标签，例如背景类别。
            当计算损失时，将会忽略这些特定标签的损失计算，以避免这些标签对损失的影响。
            如果设置了 ignore_index 参数，'none' 的 reduction 参数会很有用，因为它可以让你获取每个样本的损失，包括被忽略的样本。

            总之，reduction 参数允许在计算损失时控制输出形式，以满足不同的需求。
        """

    def forward(self, logits, labels):
        # 1. 计算 n_min(至少算多少个像素点)
        n_min = (
            labels[labels != self.lb_ignore].numel() // self.ignore_simple_sample_factor
        )

        # 2. 使用 CrossEntropy 计算损失, 之后再将其展平
        loss = self.criteria(logits, labels).view(-1)

        # 3. 选出所有loss中大于self.thresh的像素点 -> 困难样本
        loss_hard = loss[loss > self.thresh]

        # 4. 如果总数小于 n_min, 那么肯定要保证有 n_min 个像素点的 loss
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        # 5. 如果参与的像素点的个数 > n_min 个，那么这些点都参与计算
        loss_hard_mean = torch.mean(loss_hard)

        # 6. 返回损失的均值
        return loss_hard_mean


class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss


class LovaszLossHinge(nn.Module):
    def __init__(self):
        super(LovaszLossHinge, self).__init__()

    def forward(self, input, target):
        loss = L.lovasz_hinge(input, target)
        return loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = (
                grad_output
                * 2
                * (target * self.union - self.inter)
                / (self.union * self.union)
            )
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
