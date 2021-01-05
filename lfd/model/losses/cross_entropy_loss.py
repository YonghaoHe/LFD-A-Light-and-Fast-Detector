# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss

__all__ = ['CrossEntropyLoss']


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None
                ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor
        )
        return loss_cls
