#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch
import torch.nn as nn


class ZoneoutLayer(nn.Module):
    def __init__(self, RNN_cell, zoneout_prob):
        super(ZoneoutLayer, self).__init__()
        self.RNN_cell = RNN_cell
        self.zoneout_prob = zoneout_prob

    def forward(self, input, state, is_train):
        state_new = self.RNN_cell(input, state)
        if isinstance(self.RNN_cell, nn.LSTMCell):
            h_new, c_new = state_new
            if is_train and self.zoneout_prob < 1.0:
                h, c = state
                mask = torch.bernoulli(
                    self.zoneout_prob *
                    torch.ones_like(h_new)) / self.zoneout_prob
                h_new = mask * (h_new - h) + h
                c_new = mask * (c_new - c) + c
                state_new = tuple([h_new, c_new])
            output = h_new
        else:
            if is_train and self.zoneout_prob < 1.0:
                mask = torch.bernoulli(
                    self.zoneout_prob *
                    torch.ones_like(state_new)) / self.zoneout_prob
                state_new = mask * (state_new - state) + state
            output = state_new
        return output, state_new
