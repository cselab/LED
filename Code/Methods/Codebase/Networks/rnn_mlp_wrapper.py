#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch
import torch.nn as nn
from . import activations


class RNN_MLP_wrapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, act, act_output):
        super(RNN_MLP_wrapper, self).__init__()

        self.hidden_sizes = [input_size] + hidden_sizes + [output_size]
        self.act = activations.getActivation(act)
        self.act_output = activations.getActivation(act_output)
        self.layers = nn.ModuleList()

        # print(self.hidden_sizes)
        for ln in range(len(self.hidden_sizes) - 1):
            self.layers.append(
                nn.Linear(self.hidden_sizes[ln],
                          self.hidden_sizes[ln + 1],
                          bias=True))
            if ln < len(self.hidden_sizes) - 2:
                self.layers.append(self.act)
            else:
                self.layers.append(self.act_output)
        # print(self.layers)

    def forward(self, input_, state, is_train=False):
        for layer in self.layers:
            output = layer(input_)
            input_ = output
        return output, state
