#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def normalized_entropy(x, eps=1e-9):
    """
    Calculate normalized entropy of a discrete probability distribution.

    Args:
        x: torch.tensor
            logit
    """
    norm = torch.log(torch.tensor([len(x[0])])).item()
    h = -torch.sum(torch.log(x) * x, -1) / norm
    return h


def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)
