#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn

class AnchorLoss(nn.Module):
    def __init__(self, dim_emb=768):
        super().__init__()
        self.factor = math.sqrt(dim_emb)

    def forward(self, anchors):
        n_classes, k, _ = anchors.shape
        anchors = anchors.view(n_classes, -1)
        distances = (anchors.unsqueeze(0) - anchors.unsqueeze(1)) ** 2
        distances = distances / self.factor
        loss = -distances.sum() / k / k
        return loss


class CenterLoss(nn.Module):
    def __init__(self, n_classes=8, reduction='mean', dim_emb=768):
        super().__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.factor = math.sqrt(dim_emb)

    def forward(self, distances, labels, confidence):
        # distances are [batch, n_class, n_anchors]
        distances = distances[range(len(labels)), labels]  # [batch, n_anchors]
        # pick the closest
        distances = torch.min(distances, 1).values  # [batch]
        # loss
        loss = distances * confidence.view(-1) / self.factor
        if self.reduction == 'mean':
            return loss.sum() / len(loss)
        return loss.sum()


class DistLoss(nn.Module):
    """
    NLL Loss applied to probability distribution.
    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, x, l):
        return self.loss(torch.log(x), l)
