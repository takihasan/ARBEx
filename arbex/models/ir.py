#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, channel_in, depth, stride):
        super().__init__()
        if channel_in == depth:
            self.shortcut = nn.MaxPool2d(1, stride)
        else:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(channel_in, depth,
                              (1, 1), stride, bias=False),
                    nn.BatchNorm2d(depth),
                    )

        self.res = nn.Sequential(
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, depth,
                          (3, 3), (1, 1), 1, bias=False),
                nn.PReLU(depth),
                nn.Conv2d(depth, depth,
                          (3, 3), stride, 1, bias=False),
                nn.BatchNorm2d(depth),
                )


    def forward(self, x):
        return self.res(x) + self.shortcut(x)


class Block(nn.Module):
    def __init__(self, channel_in, depth, layers, stride=2):
        super().__init__()
        net = [Bottleneck(channel_in, depth, stride)]
        net += [Bottleneck(depth, depth, 1) for _ in range(layers - 1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class IR50(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Sequential(
                nn.Conv2d(3, 64,
                          3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
                )

        self.net_1 = Block(64, 64, 3)
        self.net_2 = Block(64, 128, 4)
        self.net_3 = Block(128, 256, 14)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=112)
        x = self.input_layer(x)
        out_1 = self.net_1(x)
        out_2 = self.net_2(out_1)
        out_3 = self.net_3(out_2)
        return out_1, out_2, out_3
