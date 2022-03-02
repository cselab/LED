#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
""" Torch """
import torch
import torch.nn as nn

from . import activations
# import activations


class MLPViewModule(nn.Module):
    def __init__(self, input_dim, channels, Dx=1, Dy=1):
        super(MLPViewModule, self).__init__()
        self.channels = channels

        if self.channels == 1:
            self.shape = input_dim * Dx

        elif self.channels == 2:
            self.shape = input_dim * Dx * Dy

        else:
            raise ValueError("Not implemented.")

    def forward(self, x):
        # K remains unchanged
        shape_ = (*np.shape(x)[:1], self.shape)
        return x.view(shape_)


class LinearResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearResidualLayer, self).__init__()

        if input_dim == output_dim:
            self.is_residual = True
        else:
            self.is_residual = False

        self.layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        if self.is_residual:
            return self.layer(x) + x
        else:
            return self.layer(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        channels=None,
        input_dim=None,
        Dx=1,
        Dy=1,
        output_dim=None,
        activation=None,
        activation_output=None,
        layers_size=None,
        dropout_keep_prob=1.0,
        torch_dtype=torch.DoubleTensor,
    ):
        super(MLPEncoder, self).__init__()

        self.channels = channels
        self.Dx = Dx
        self.Dy = Dy
        self.input_dim = input_dim

        if self.channels == 1:
            self.input_dim_expanded = input_dim * Dx

        elif self.channels == 2:
            self.input_dim_expanded = input_dim * Dx * Dy

        else:
            raise ValueError("Not implemented.")

        self.output_dim = output_dim
        self.activation = activation
        self.activation_output = activation_output
        self.layers_size = layers_size
        self.dropout_keep_prob = dropout_keep_prob

        self.layers_size.append(self.output_dim)
        self.layers_size.insert(0, self.input_dim_expanded)

        print("[MLPEncoder] layers {:}".format(self.layers_size))

        self.layers_idx = []
        self.layers_num = 0

        self.layers = nn.ModuleList()

        self.layers.append(
            MLPViewModule(self.input_dim, self.channels, self.Dx, self.Dy))
        self.layers_num += 1
        self.layers_idx.append(self.layers_num - 1)

        for ln in range(len(self.layers_size) - 1):
            # self.layers.append(nn.Linear(self.layers_size[ln], self.layers_size[ln + 1], bias=True))
            self.layers.append(
                LinearResidualLayer(self.layers_size[ln],
                                    self.layers_size[ln + 1],
                                    bias=True))
            self.layers_num += 1
            if ln < len(self.layers_size) - 2:
                self.layers.append(activations.getActivation(self.activation))
                self.layers.append(nn.Dropout(p=1 - self.dropout_keep_prob))
                self.layers_num += 1
            else:
                self.layers.append(
                    activations.getActivation(self.activation_output))
                self.layers_num += 1

            self.layers_idx.append(self.layers_num - 1)

        self.layers = nn.ModuleList(self.layers)

        # Adding the MLP layer
        sizes = self.getDimensions()
        self.printDimensions()

    def forward(self, input):
        # print("# ENCODER: forward() #")
        # print(input.size())
        for layer in self.layers:
            # print("--------------")
            # print(layer)
            input = layer(input)
            # print(input.size())
        return input

    def getDimensions(self):
        if self.channels == 1:
            input = torch.randn(
                1,
                self.input_dim,
                self.Dx,
            )
        elif self.channels == 2:
            input = torch.randn(
                1,
                self.input_dim,
                self.Dx,
                self.Dy,
            )
        else:
            raise ValueError("Not implemented. channels = {:}".format(
                self.channels))

        sizes_all = []
        input_size = list(input.size()[1:])
        for layer in self.layers:
            # print(layer)
            # print(input.size())
            # input = layer(input)
            sizes_all.append(list(input.size()[1:]))
        sizes = []
        layer_num = 0
        for size in sizes_all:
            if layer_num in self.layers_idx:
                sizes.append(size)
            layer_num += 1

        sizes.insert(0, input_size)
        return sizes

    def printDimensions(self):
        print("[Encoder] -------------------")
        print("[Encoder] printDimensions() #")
        sizes = self.getDimensions()
        for size in sizes:
            print("[Encoder] {:}".format(size))
        print("[Encoder] -------------------")
        return sizes


class MLPReViewModule(nn.Module):
    def __init__(self, output_dim, channels, Dx=1, Dy=1):
        super(MLPReViewModule, self).__init__()
        self.output_dim = output_dim
        self.Dx = Dx
        self.Dy = Dy
        self.channels = channels

        if self.channels == 1:
            self.shape_ = (self.output_dim, self.Dx)
        elif channels == 2:
            self.shape_ = (self.output_dim, self.Dx, self.Dy)
        else:
            raise ValueError("Not implemented.")

    def forward(self, x):
        shape_ = (*np.shape(x)[:1], *self.shape_)
        return x.view(shape_)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        channels=None,
        input_dim=None,
        Dx=1,
        Dy=1,
        output_dim=None,
        activation=None,
        activation_output=None,
        layers_size=None,
        dropout_keep_prob=1.0,
        torch_dtype=torch.DoubleTensor,
    ):
        super(MLPDecoder, self).__init__()

        self.channels = channels
        self.Dx = Dx
        self.Dy = Dy
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.channels == 1:
            self.output_dim_expanded = output_dim * Dx

        elif self.channels == 2:
            self.output_dim_expanded = output_dim * Dx * Dy

        else:
            raise ValueError("Not implemented.")

        self.activation = activation
        self.activation_output = activation_output
        self.layers_size = layers_size
        self.dropout_keep_prob = dropout_keep_prob

        self.layers_size.insert(0, self.input_dim)
        self.layers_size.append(self.output_dim_expanded)

        print("[MLPDecoder] layers {:}".format(self.layers_size))

        self.layers_idx = []
        self.layers_num = 0

        self.layers = nn.ModuleList()

        for ln in range(len(self.layers_size) - 1):
            # self.layers.append(nn.Linear(self.layers_size[ln], self.layers_size[ln + 1], bias=True))
            self.layers.append(
                LinearResidualLayer(self.layers_size[ln],
                                    self.layers_size[ln + 1],
                                    bias=True))
            self.layers_num += 1
            if ln < len(self.layers_size) - 2:
                self.layers.append(activations.getActivation(self.activation))
                self.layers.append(nn.Dropout(p=1 - self.dropout_keep_prob))
                self.layers_num += 1
            else:
                self.layers.append(
                    activations.getActivation(self.activation_output))
                self.layers_num += 1

            self.layers_idx.append(self.layers_num - 1)

        self.layers.append(
            MLPReViewModule(self.output_dim, self.channels, self.Dx, self.Dy))
        self.layers_num += 1
        self.layers_idx.append(self.layers_num - 1)

        self.layers = nn.ModuleList(self.layers)

        # Adding the MLP layer
        sizes = self.getDimensions()
        self.printDimensions()

    def forward(self, input):
        # print("# ENCODER: forward() #")
        # print(input.size())
        for layer in self.layers:
            # print("--------------")
            # print(layer)
            input = layer(input)
            # print(input.size())
        return input

    def getDimensions(self):
        input = torch.randn(
            1,
            self.input_dim,
        )
        sizes_all = []
        input_size = list(input.size()[1:])
        for layer in self.layers:
            input = layer(input)
            sizes_all.append(list(input.size()[1:]))
        sizes = []
        layer_num = 0
        for size in sizes_all:
            if layer_num in self.layers_idx:
                sizes.append(size)
            layer_num += 1

        sizes.insert(0, input_size)
        return sizes

    def printDimensions(self):
        print("[Decoder] -------------------")
        print("[Decoder] printDimensions() #")
        sizes = self.getDimensions()
        for size in sizes:
            print("[Decoder] {:}".format(size))
        print("[Decoder] -------------------")
        return sizes


if __name__ == '__main__':

    import activations

    input_chanels = 2
    activation = "celu"
    activation_output = "identity"

    latent_state_dim = 4

    system_name = "KSGP64L22"
    channels = 1
    input_dim = 2
    Dx = 101
    output_dim = latent_state_dim

    layers_size = [10, 10, 10]
    dropout_keep_prob = 1.0

    input = torch.randn(1, input_dim, Dx)

    encoder = MLPEncoder(
        channels=channels,
        Dx=Dx,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        activation_output=activation_output,
        layers_size=layers_size.copy(),
        dropout_keep_prob=dropout_keep_prob,
    )
    sizes = encoder.printDimensions()

    print("###############")
    print("input.size()")
    print(input.size())
    latent = encoder.forward(input)
    print("latent = encoder.forward(input)")
    print(latent.size())

    output_dim = input_dim
    input_dim = latent_state_dim
    decoder = MLPDecoder(
        channels=channels,
        Dx=Dx,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        activation_output=activation_output,
        layers_size=layers_size.copy(),
        dropout_keep_prob=dropout_keep_prob,
    )
    sizes = decoder.printDimensions()

    print("###############")
    print("latent.size()")
    print(latent.size())
    output = decoder.forward(latent)
    print("output = decoder.forward(latent)")
    print(output.size())

    print(encoder.layers)
    print(decoder.layers)
