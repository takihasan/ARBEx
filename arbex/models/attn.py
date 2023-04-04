#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


class SelfAttn(nn.Module):
    def __init__(self, n_anchors=1, n_classes=8, size_emb=768, nhead=1,):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        size_out = n_classes * n_anchors
        self.Q = nn.Linear(size_emb, size_out)
        self.K = nn.Linear(size_emb, size_out)
        self.V = nn.Linear(size_emb, size_out)
        self.attn_fn = nn.MultiheadAttention(size_out, num_heads=nhead)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        attn_scores, _ = self.attn_fn(q, k, v)
        attn_scores = torch.softmax(attn_scores, -1)
        return attn_scores.view(-1, self.n_classes, self.n_anchors).sum(-1)
