"""
Author: Huibin Lin (huibinlin@outlook.com)
Date: March 20, 2023
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='cosine'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        return self.loss_func(x, y)


class ProxyPLoss(nn.Module):
    '''
    pass
    '''

    def __init__(self, num_classes, scale=12):
        super(ProxyPLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale

    def forward(self, feature, target, proxy, hard_negatives=None):
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C) dot product
        label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
        pred = torch.masked_select(pred.transpose(1, 0), label)  # N,

        pred = pred.unsqueeze(1)  # (N, 1)  # 正样本对

        neg_feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
        neg_feature = neg_feature * ~label_matrix  # get negative matrix

        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, p=2, dim=1)
            # hard_neg = (feature * torch.cat([hard_negatives, hard_negatives], dim=0)).sum(dim=1)  # Dot Product
            hard_neg = (feature * hard_negatives).sum(dim=1)  # Dot Product
            hard_neg = hard_neg.unsqueeze(1)
            negatives = torch.cat([neg_feature, hard_neg], dim=1)
        else:
            negatives = neg_feature
        negatives = negatives.masked_fill(negatives < 1e-6, 0)  # (N, N)
        logits = torch.cat([pred, negatives], dim=1)  # (N, N+hard_count+1)
        label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)  # 预测为标签0
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
        return loss


