#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1):
        """
        Conv2d -> BatchNorm -> PReLU
        """
        super().__init__()

        self.net = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False,),
                nn.BatchNorm2d(channel_out),
                nn.PReLU(channel_out),
                )

    def forward(self, x):
        return self.net(x)


class LinearBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1):
        super().__init__()
        """
        Conv2d -> BatchNorm
        """

        self.net = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False,),
                nn.BatchNorm2d(channel_out),
                )

    def forward(self, x):
        return self.net(x)


class DepthWise(nn.Module):
    def __init__(self, channel_in, channel_out,
                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 groups=1, residual=False):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
                ConvBlock(channel_in, groups,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          ),
                ConvBlock(groups, groups,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          groups=groups),
                LinearBlock(groups, channel_out,
                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                            ),
                )

    def forward(self, x):
        x_ = x
        x = self.net(x)
        if self.residual:
            x += x_
        return x


class Residual(nn.Module):
    def __init__(self, channel, blocks, groups,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        net = [DepthWise(channel, channel,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, groups=groups, residual=True)
               for _ in range(blocks)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MobileFaceNet(nn.Module):
    def __init__(self):
        """
        MobileFaceNet module
        """
        super().__init__()
        self.path_1 = nn.Sequential(
                ConvBlock(3, 64,
                          kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                ConvBlock(64, 64,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          groups=64),
                DepthWise(64, 64,
                          kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                          groups=128),
                Residual(64, 4, 128,
                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                         ),
                )

        self.path_2 = nn.Sequential(
                DepthWise(64, 128,
                          kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                          groups=256),
                Residual(128, 6, 256,
                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                         ),
                )

        self.path_3 = nn.Sequential(
                DepthWise(128, 128,
                          kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                          groups=512),
                Residual(128, 2, 256,
                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                         ),
                ConvBlock(128, 512,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          ),
                )


    def forward(self, x):
        out_1 = self.path_1(x)
        out_2 = self.path_2(out_1)
        out_3 = self.path_3(out_2)
        return [out_1, out_2, out_3]
