#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class Anchors(nn.Module):
    def __init__(self, size_emb=768, n_classes=8, n_anchors=10):
        """
        Anchors of embeddings.

        Args:
            emb_size: int
                dimension of input embedding
            n_classes: int
                number of classes
            n_anchors: int
                number of anchoring embeddings
        """
        super().__init__()

        self.size_emb = size_emb
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        anchors = torch.zeros((self.n_classes, self.n_anchors, self.size_emb))
        self.anchors = nn.Parameter(anchors)

    def forward(self, x):
        """
        find similarity between for each pair of embeddings in x and anchors
        """
        # x -> [batch, emb]
        x = x.view(x.shape[0], 1, 1, x.shape[1])  # [batch, 1, 1, emb]
        anchors = self.anchors.unsqueeze(0)  # [1, classes, anchors, emb]
        distances = (anchors - x) ** 2  # [batch, classes, anchors, emb]
        distances = distances.sum(-1)  # [batch, classes, anchors]
        distances = torch.sqrt(distances)  # [batch, classes, anchors]
        return distances

    def get_anchors(self):
        return self.anchors
