#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np


class viewEliminateChannels(nn.Module):
    def __init__(self, params):
        super(viewEliminateChannels, self).__init__()
        self.Dx = params["Dx"]
        self.Dy = params["Dy"]
        self.Dz = params["Dz"]
        self.channels = params["channels"]
        self.input_dim = params["input_dim"]

        if self.channels == 1:
            self.shape_ = (self.input_dim, self.Dx)
            self.total_dim = self.input_dim * self.Dx
        elif self.channels == 2:
            self.shape_ = (self.input_dim, self.Dx, self.Dy)
            self.total_dim = self.input_dim * self.Dx * self.Dy
        elif self.channels == 3:
            self.shape_ = (self.input_dim, self.Dx, self.Dy, self.Dz)
            self.total_dim = self.input_dim * self.Dx * self.Dy * self.Dz
        else:
            raise ValueError("Not implemented.")

    def forward(self, x):
        # print("# viewEliminateChannels() #")
        # K remains unchanged
        # print(x.size())
        shape_ = (x.size(0), self.total_dim)
        temp = x.view(shape_)
        # print(temp.size())
        return temp


class viewAddChannels(nn.Module):
    def __init__(self, params):
        super(viewAddChannels, self).__init__()
        self.Dx = params["Dx"]
        self.Dy = params["Dy"]
        self.Dz = params["Dz"]
        self.channels = params["channels"]
        self.input_dim = params["input_dim"]

        if self.channels == 1:
            self.shape_ = (self.input_dim, self.Dx)
            self.total_dim = self.input_dim * self.Dx
        elif self.channels == 2:
            self.shape_ = (self.input_dim, self.Dx, self.Dy)
            self.total_dim = self.input_dim * self.Dx * self.Dy
        elif self.channels == 3:
            self.shape_ = (self.input_dim, self.Dx, self.Dy, self.Dz)
            self.total_dim = self.input_dim * self.Dx * self.Dy * self.Dz
        else:
            raise ValueError("Not implemented.")

    def forward(self, x):
        # print("# viewAddChannels() #")
        # K remains unchanged
        # print(x.size())
        shape_ = (x.size(0), *self.shape_)
        temp = x.view(shape_)
        # print(temp.size())
        return temp


class dummy(nn.Module):
    def __init__(self, params, model):
        super(dummy, self).__init__()
        self.parent = model
        self.buildNetwork()

        self.module_list = [
            self.ENCODER,
            self.DECODER,
        ]

    def buildNetwork(self):
        self.ENCODER = nn.ModuleList()
        self.ENCODER.append(viewEliminateChannels(self.parent.params))

        self.DECODER = nn.ModuleList()
        self.DECODER.append(viewAddChannels(self.parent.params))

    def printModuleList(self):
        self.print("[crnn_model] module_list :")
        module_list_str = str(self.module_list)
        module_list_str_lines = module_list_str.split("\n")
        for line in module_list_str_lines:
            self.print("[crnn_model] " + line)
        return 0

    def countTrainableParams(self):
        temp = 0
        for layers in self.module_list:
            for layer in layers:
                temp += sum(p.numel() for p in layer.parameters()
                            if p.requires_grad)
        return temp

    def countParams(self):
        temp = 0
        for layers in self.module_list:
            for layer in layers:
                temp += sum(p.numel() for p in layer.parameters())
        return temp

    def getParams(self):
        params = list()
        named_params = list()
        for layers in self.module_list:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params

    def forwardEncoder(self, inputs):
        # print("# forwardEncoder() #")
        outputs = inputs
        shape_ = outputs.size()
        T, K = shape_[0], shape_[1]
        outputs = torch.reshape(outputs, (T * K, *shape_[2:]))
        for l in range(len(self.ENCODER)):
            outputs = self.ENCODER[l](outputs)
        shape_ = outputs.size()
        outputs = torch.reshape(outputs, (T, K, *shape_[1:]))
        return outputs

    def forwardDecoder(self, inputs):
        # print("# forwardDecoder() #")
        outputs = inputs
        shape_ = outputs.size()
        T, K = shape_[0], shape_[1]
        outputs = torch.reshape(outputs, (T * K, *shape_[2:]))
        for l in range(len(self.DECODER)):
            outputs = self.DECODER[l](outputs)
        shape_ = outputs.size()
        outputs = torch.reshape(outputs, (T, K, *shape_[1:]))
        return outputs

    def sendModelToCuda(self):
        pass
