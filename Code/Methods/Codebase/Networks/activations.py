#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
""" Torch """
import torch
import torch.nn as nn


class Tanhplus(nn.Module):
    def __init__(self):
        super(Tanhplus, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 0.5 + self.tanh(x) * 0.5


torch_activations = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "celu": nn.CELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanhplus": Tanhplus(),
}


def getActivation(str_):
    return torch_activations[str_]
