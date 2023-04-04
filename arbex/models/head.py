#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, size_in=768, size_out=8, size_hidden=[384],
                 dropout=0.5, batch_norm=False):
        """
        Multi-layer perceptron classification head.
        Each layer execept the last one is follow by the ReLU activation
        and batch normalization.

        Args:
            size_in: int
                dimension of input embedding
            size_out: int
                number of classes
            size_hidden: list
                sizes of hidden layers
            dropout: float
                what dropout value to apply between layers
            batch_norm: bool
                use batch normalization layers or not
        """
        super().__init__()

        self.size_in = size_in
        self.size_out = size_out
        self.size_hidden = size_hidden
        self.dropout = dropout
        self.batch_norm = batch_norm

        net = []
        for h in self.size_hidden:
            net.append(nn.Linear(size_in, h))
            net.append(nn.ReLU())
            if self.batch_norm:
                net.append(nn.BatchNorm1d(h))
            net.append(nn.Dropout(self.dropout))
            size_in = h

        net.append(nn.Linear(h, self.size_out))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
