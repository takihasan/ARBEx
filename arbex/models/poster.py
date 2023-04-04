#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from .vit import VisionTransformer as ViT
from .ir import IR50
from .mobilefacenet import MobileFaceNet

class PatchEmbed(nn.Module):
    def __init__(self, size_emb, channel_in):
        super().__init__()
        self.proj = nn.Conv2d(channel_in, size_emb, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Window(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        batch, height, width, channel = x.shape
        x = self.norm(x)
        x_ = x  # residual
        x = x.view(batch, height // self.size, self.size,
                   width // self.size, self.size, channel)
        x = x.permute(0, 1, 3, 2, 4, 5)  # batch, segment h, segment w, patch, patch, channel
        x = x.contiguous()
        x = x.view(-1, self.size, self.size, channel)  # collapse batch and segment
        x = x.view(-1, self.size * self.size, channel)  # collapse patch and patch
        return x, x_


class GlobalWindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = self.dim // self.heads
        self.window = (window, window)
        self.scale = self.dim_head ** (-0.5)
        self.table = nn.Parameter(
                torch.zeros((2 * self.window[0] - 1) * (2 * self.window[1] - 1),
                            self.heads)
                )
        hs = torch.arange(self.window[0])
        ws = torch.arange(self.window[1])
        coords = torch.flatten(torch.stack(torch.meshgrid([hs, ws], indexing='ij')), 1)
        coords = coords.unsqueeze(2) - coords.unsqueeze(1)
        coords = coords.permute(1, 2, 0).contiguous()
        coords[:, :, 0] += self.window[0] - 1
        coords[:, :, 1] += self.window[1] - 1
        coords[:, :, 0] *= 2 * self.window[1] - 1
        index = coords.sum(-1)
        self.register_buffer('index_relative', index)
        self.qkv = nn.Linear(self.dim, self.dim *2, bias=True)
        self.linear = nn.Linear(self.dim, self.dim)
        self.drop_attn = nn.Dropout(0.0)
        self.drop_proj = nn.Dropout(0.0)
        trunc_normal_(self.table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q):
        batch_x, n_x, channel_x = x.shape
        batch_q = q.shape[0]
        head_dim = channel_x // self.heads
        kv = self.qkv(x)
        kv = kv.reshape(batch_x, n_x, 2, self.heads, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0].transpose(-2, -1), kv[1]
        q = q.repeat(1, batch_x // batch_q, 1, 1, 1)
        q = q.reshape(batch_x, self.heads, n_x, head_dim)
        q = q * self.scale
        attn = q @ k
        bias = self.table[self.index_relative.view(-1)]
        bias = bias.view(self.window[0] * self.window[1],
                         self.window[0] * self.window[1], -1)
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + bias
        attn = self.softmax(attn)
        attn = self.drop_attn(attn)
        x = attn @ v
        x = x.transpose(1, 2)
        x = x.reshape(batch_x, n_x, channel_x)
        x = self.linear(x)
        x = self.drop_proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, size,
                 ratio=4.0, activation=nn.GELU,
                 drop=0.0, drop_path=0.0,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.size = size
        self.ratio = ratio
        self.activation = activation
        self.drop = drop
        if layer_scale is not None:
            self.gamma_1 = nn.Parameter(layer_scale * torch.ones(self.dim))
            self.gamma_2 = nn.Parameter(layer_scale * torch.ones(self.dim))
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

        self.mlp = nn.Sequential(
                nn.Linear(self.dim, int(self.dim * ratio)),
                self.activation(),
                nn.Dropout(self.drop),
                nn.Linear(int(self.dim * ratio), self.dim),
                nn.Dropout(self.drop),
                )

        self.norm = nn.LayerNorm(self.dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


    def forward(self, window, shortcut):
        batch, height, width, channel = shortcut.shape
        B = int(window.shape[0] * self.size * self.size /  height / width)
        x = window.view(B, height // self.size, width // self.size,
                        self.size, self.size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous()
        x = x.view(B, height, width, -1)
        x = self.drop_path(self.gamma_1 * x)
        x = x + shortcut
        norm = self.norm(x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(norm))
        return x


class PosterV2(nn.Module):
    def __init__(self, landmark_extractor, ir,
                 windows=[28, 14, 7],
                 heads=[2, 4, 8],
                 dims=[64, 128, 256],
                 dim_emb=768,
                 depth=2,
                 num_heads=8,
                 ):
        super().__init__()

        self.heads = heads
        self.windows = windows
        self.patch_size = [i*i for i in self.windows]
        self.dims = dims
        self.dim_head = [d // h for d, h in zip(self.dims, self.heads)]
        self.dim_emb = dim_emb

        self.landmark = landmark_extractor
        self.ir = ir
        self.conv = nn.Conv2d(512, 256,
                              3, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1)
            for d in self.dims])

        self.window_module = nn.ModuleList([
            Window(p, d) for p, d in zip(self.windows, self.dims)
            ])

        self.attns = nn.ModuleList([
            GlobalWindowAttention(d, h, s) for
            d, h, s in zip(self.dims, self.heads, self.windows)
            ])

        self.ffns = nn.ModuleList([
            FeedForward(d, w, layer_scale=1e-5, drop_path=drop) for
            d, w, drop in zip(self.dims, self.windows, [0.0, 0.125, 0.25])
            ])

        self.embs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.dims[0], self.dim_emb,
                                    kernel_size=3, stride=2, padding=1),
                          nn.Conv2d(self.dim_emb, self.dim_emb,
                                    kernel_size=3, stride=2, padding=1)),
            nn.Conv2d(self.dims[1], self.dim_emb,
                      kernel_size=3, stride=2, padding=1,),
            PatchEmbed(channel_in=256, size_emb=self.dim_emb,),
            ])

        self.vit = ViT(img_size=14,
                       patch_size=14,
                       depth=depth,
                       num_heads=num_heads,
                       embed_dim=self.dim_emb,)
        self.vit.heads = nn.Identity()  # remove classification head


    def forward(self, x):
        # facenet
        x_ = nn.functional.interpolate(x, size=112)
        xf = self.landmark(x_)
        xf[-1] = self.conv(xf[-1])
        # move channel to last place
        xf = [l_i.permute(0, 2, 3, 1) for l_i in xf]
        # queries
        qs = [x_f.reshape(x_f.shape[0], p, h, d).permute(0, 2, 1, 3).unsqueeze(1)
             for x_f, p, h, d in
             zip(xf, self.patch_size, self.heads, self.dim_head)]
        # ir
        xi = self.ir(x)
        xi = [c(i) for c, i in zip(self.convs, xi)]
        w_s = [w(i) for w, i in zip(self.window_module, xi)]  # w, s
        os = [attn(w, q_i) for attn, (w, _), q_i in zip(self.attns, w_s, qs)]
        os = [ffn(o, s) for ffn, o, (_, s) in zip(self.ffns, os, w_s)]
        os = [o.permute(0, 3, 1, 2) for o in os]
        os = [emb(o) for emb, o in zip(self.embs, os)]
        os[0] = os[0].flatten(2).transpose(1, 2)
        os[1] = os[1].flatten(2).transpose(1, 2)
        o = torch.cat(os, dim=1)
        out = self.vit(o)
        return out


def get_poster(path_landmark=None, path_ir=None, dim_emb=768):
    landmark_extractor = MobileFaceNet()
    ir = IR50()

    if path_landmark is not None:
        # load
        landmark_extractor.load_state_dict(torch.load(path_landmark))
        # freeze
        for m in landmark_extractor.parameters():
            m.requires_grad = False

    if path_ir is not None:
        ir.load_state_dict(torch.load(path_ir))
        for m in ir.parameters():
            m.requires_grad = False

    poster = PosterV2(landmark_extractor, ir, dim_emb=dim_emb)
    return poster


def get_poster_pretrained(
        path_landmark='pretrained/mobilefacenet.pth',
        path_ir='pretrained/ir50.pth',
        ):
    loc = '/'.join(__file__.split('/')[:-1])
    path_landmark = loc + '/' + path_landmark
    path_ir = loc + '/' + path_ir
    poster = get_poster(path_landmark, path_ir)
    return poster
