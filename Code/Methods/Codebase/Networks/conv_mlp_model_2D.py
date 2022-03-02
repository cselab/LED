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
from . import interpolation_layer
# import activations
# import interpolation_layer


class ViewModule(nn.Module):
    def __init__(self, shape):
        super(ViewModule, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def getAutoencoderArchitecture(model):
    system_name = model.params["system_name"]
    conv_architecture = model.params["AE_conv_architecture"]
    latent_state_dim = model.params["latent_state_dim"]
    params_dict = {}

    if system_name in ["NSC-Re600"]:

        if conv_architecture == "conv_latent_1":
            print("[CONV-NET]: Loading architecture: {:}".format(
                conv_architecture))
            conv_layers_kernel_sizes = [11, 9, 7, 5]
            conv_layers_channels = [5, 10, 20, 1]
            conv_layers_channels = list(model.params["AE_size_factor"] *
                                        np.array(conv_layers_channels))
            conv_layers_strides = [1, 1, 1, 1]
            pool_kernel_sizes = [2, 2, 2, 2]
            pool_kernel_strides = [2, 2, 2, 2]
            upsampling_factors = [2, 2, 2, 2]
            upsampling_mode = "bilinear"
            interpolated_subsampling_layer_factor = None

        else:
            raise ValueError(
                "(conv_mlp_model_2D.py) Architecture name {:} of system name {:} not recognized. Do not know how to build CNN-MLP Autoencoder."
                .format(conv_architecture, system_name))

    elif system_name in ["Dummy", "DummyStructured"]:

        if conv_architecture == "conv_latent_1":
            print("[CONV-NET]: Loading architecture: {:}".format(
                conv_architecture))
            conv_layers_kernel_sizes = [3, 3]
            conv_layers_channels = [2, 5]
            conv_layers_channels = list(model.params["AE_size_factor"] *
                                        np.array(conv_layers_channels))
            conv_layers_strides = [1, 1]
            pool_kernel_sizes = [2, 2]
            pool_kernel_strides = [2, 2]
            upsampling_factors = [2, 2]
            upsampling_mode = "bilinear"
            interpolated_subsampling_layer_factor = None

        else:
            raise ValueError(
                "(conv_mlp_model_2D.py) Architecture name {:} of system name {:} not recognized. Do not know how to build CNN-MLP Autoencoder."
                .format(conv_architecture, system_name))

    elif system_name in [
            "cylRe100",
            "cylRe1000",
            "cylRe2500",
    ]:
        if conv_architecture == "conv_latent_1":
            print("[CONV-NET]: Loading architecture: {:}".format(
                conv_architecture))

            conv_layers_kernel_sizes = [11, 9, 7, 5, 3]
            conv_layers_channels = [8, 16, 16, 32, 8]
            conv_layers_channels = list(model.params["AE_size_factor"] *
                                        np.array(conv_layers_channels))
            conv_layers_strides = [1, 1, 1, 1, 1]
            pool_kernel_sizes = [2, 4, 2, 2, 2]
            pool_kernel_strides = [2, 4, 2, 2, 2]
            upsampling_factors = [2, 4, 2, 2, 2]

            # conv_layers_kernel_sizes = [11, 9, 7, 5, 3, 3]
            # conv_layers_channels = [8, 16, 16, 32, 32, 8]
            # conv_layers_channels = list(model.params["AE_size_factor"]*np.array(conv_layers_channels))
            # conv_layers_strides = [1, 1, 1, 1, 1, 1]
            # pool_kernel_sizes = [2, 4, 2, 2, 2, 2]
            # pool_kernel_strides = [2, 4, 2, 2, 2, 2]
            # upsampling_factors = [2, 4, 2, 2, 2, 2]

            upsampling_factors = upsampling_factors[::-1]
            upsampling_mode = "bilinear"

            interpolated_subsampling_layer_factor = None

    elif system_name in [
            "cylRe100HR", "cylRe1000HR", "cylRe100HR_demo", "cylRe1000HR_demo",
            "cylRe100HRDt005", "cylRe1000HRDt005", "cylRe100_vortPres_veryLowRes",
            "cylRe1000HRLarge",
    ]:
        # if conv_architecture == "conv_latent_1":
        #     print("[CONV-NET]: Loading architecture: {:}".format(conv_architecture))

        #     conv_layers_kernel_sizes = [13, 13, 13, 13, 13, 13]
        #     conv_layers_channels = [8, 8, 8, 8, 8, 1]
        #     conv_layers_channels = list(model.params["AE_size_factor"]*np.array(conv_layers_channels))
        #     conv_layers_strides = [1, 1, 1, 1, 1, 1]
        #     pool_kernel_sizes = [2, 2, 2, 2, 2, 2]
        #     pool_kernel_strides = pool_kernel_sizes
        #     upsampling_factors = [2, 2, 2, 2, 2, 2]
        #     upsampling_mode = "bilinear"
        #     upsampling_factors = upsampling_factors[::-1]

        if conv_architecture == "conv_latent_1":

            NUM_LAYERS = 4
            kernel_size = 5

        elif conv_architecture == "conv_latent_2":

            NUM_LAYERS = 4
            kernel_size = 7

        elif conv_architecture == "conv_latent_3":

            NUM_LAYERS = 4
            kernel_size = 13

        elif conv_architecture == "conv_latent_4":

            NUM_LAYERS = 6
            kernel_size = 5

        elif conv_architecture == "conv_latent_5":

            NUM_LAYERS = 6
            kernel_size = 7

        elif conv_architecture == "conv_latent_6":

            NUM_LAYERS = 6
            kernel_size = 13
            
        else:
            raise ValueError(
                "(conv_mlp_model_2D.py) Architecture name {:} of system name {:} not recognized. Do not know how to build CNN-MLP Autoencoder."
                .format(conv_architecture, system_name))

        print(
            "[CONV-NET]: Loading architecture: {:}".format(conv_architecture))

        channel_size = 10
        conv_layers_kernel_sizes = list(kernel_size *
                                        np.ones(NUM_LAYERS).astype(int))
        # print(conv_layers_kernel_sizes)
        conv_layers_channels = list(
            channel_size * np.ones(NUM_LAYERS - 1).astype(int)) + [1]
        conv_layers_channels = list(model.params["AE_size_factor"] *
                                    np.array(conv_layers_channels))
        conv_layers_strides = list(1 * np.ones(NUM_LAYERS).astype(int))
        pool_kernel_sizes = list(2 * np.ones(NUM_LAYERS).astype(int))
        pool_kernel_strides = pool_kernel_sizes
        upsampling_factors = list(2 * np.ones(NUM_LAYERS).astype(int))
        upsampling_mode = "bilinear"
        upsampling_factors = upsampling_factors[::-1]

        # elif conv_architecture == "conv_latent_4":
        #     print("[CONV-NET]: Loading architecture: {:}".format(conv_architecture))

        #     conv_layers_kernel_sizes = [13, 13, 13, 13]
        #     conv_layers_channels = [8, 8, 8, 1]
        #     conv_layers_channels = list(model.params["AE_size_factor"]*np.array(conv_layers_channels))
        #     conv_layers_strides = [1, 1, 1, 1]
        #     pool_kernel_sizes = [4, 4, 4, 4]
        #     pool_kernel_strides = pool_kernel_sizes
        #     upsampling_factors = [4, 4, 4, 4]
        #     upsampling_mode = "bilinear"
        #     upsampling_factors = upsampling_factors[::-1]

        # elif conv_architecture == "conv_latent_5":
        #     print("[CONV-NET]: Loading architecture: {:}".format(conv_architecture))

        #     conv_layers_kernel_sizes = [5, 5, 5, 5, 5, 5]
        #     conv_layers_channels = [8, 8, 8, 8, 8, 1]
        #     conv_layers_channels = list(model.params["AE_size_factor"]*np.array(conv_layers_channels))
        #     conv_layers_strides = [1, 1, 1, 1, 1, 1]
        #     pool_kernel_sizes = [2, 2, 2, 2, 2, 2]
        #     pool_kernel_strides = pool_kernel_sizes
        #     upsampling_factors = [2, 2, 2, 2, 2, 2]
        #     upsampling_mode = "bilinear"
        #     upsampling_factors = upsampling_factors[::-1]

        # elif conv_architecture == "conv_latent_6":
        #     print("[CONV-NET]: Loading architecture: {:}".format(conv_architecture))

        #     conv_layers_kernel_sizes = [5, 5, 5, 5]
        #     conv_layers_channels = [8, 8, 8, 1]
        #     conv_layers_channels = list(model.params["AE_size_factor"]*np.array(conv_layers_channels))
        #     conv_layers_strides = [1, 1, 1, 1]
        #     pool_kernel_sizes = [4, 4, 4, 4]
        #     pool_kernel_strides = pool_kernel_sizes
        #     upsampling_factors = [4, 4, 4, 4]
        #     upsampling_mode = "bilinear"
        #     upsampling_factors = upsampling_factors[::-1]
        # else:
        # raise ValueError("(conv_mlp_model_2D.py) Architecture name {:} of system name {:} not recognized. Do not know how to build CNN-MLP Autoencoder.".format(conv_architecture, system_name))
    else:
        raise ValueError(
            "System name {:} not recognized. Do not know how to build CNN-MLP Autoencoder."
            .format(system_name))

    params_dict["latent_state_dim"] = latent_state_dim
    islf = model.params["AE_interp_subsampling_input"]
    params_dict["interpolated_subsampling_layer_factor"] = (islf, islf)

    params_dict["conv_layers_kernel_sizes"] = conv_layers_kernel_sizes
    params_dict["conv_layers_channels"] = conv_layers_channels
    params_dict["conv_layers_strides"] = conv_layers_strides
    params_dict["pool_kernel_sizes"] = pool_kernel_sizes
    params_dict["pool_kernel_strides"] = pool_kernel_strides
    params_dict["upsampling_factors"] = upsampling_factors
    params_dict["upsampling_mode"] = upsampling_mode

    params_dict["batch_norm_affine"] = model.params["AE_batch_norm_affine"]
    params_dict["batch_norm"] = model.params["AE_batch_norm"]
    params_dict["conv_transpose"] = model.params["AE_conv_transpose"]
    params_dict["pool_type"] = model.params["AE_pool_type"]

    params_dict["activation"] = model.params["activation_str_general"]
    params_dict["activation_output"] = model.params["activation_str_output"]
    params_dict["input_chanels"] = model.params["input_dim"]
    params_dict["input_image_height"] = model.params["Dy"]
    params_dict["input_image_width"] = model.params["Dx"]

    params_dict["dropout_keep_prob"] = model.params["dropout_keep_prob"]
    return params_dict


def getEncoderModel(model):
    # print("# getEncoderModel() #")
    params_dict = getAutoencoderArchitecture(model)
    # Be careful: The output activation of the autoencoder should not be the output activation of the network that depends on the selection of the scaler. (here, set to the general activation)
    encoder = ConvMLPEncoder(
        latent_state_dim=params_dict["latent_state_dim"],
        conv_layers_kernel_sizes=params_dict["conv_layers_kernel_sizes"],
        conv_layers_channels=params_dict["conv_layers_channels"],
        conv_layers_strides=params_dict["conv_layers_strides"],
        pool_kernel_sizes=params_dict["pool_kernel_sizes"],
        pool_kernel_strides=params_dict["pool_kernel_strides"],
        pool_type=params_dict["pool_type"],
        activation=params_dict["activation"],
        # activation_output         = params_dict["activation_output"],
        activation_output=params_dict["activation"],
        batch_norm=params_dict["batch_norm"],
        batch_norm_affine=params_dict["batch_norm_affine"],
        input_chanels=params_dict["input_chanels"],
        input_image_height=params_dict["input_image_height"],
        input_image_width=params_dict["input_image_width"],
        dropout_keep_prob=params_dict["dropout_keep_prob"],
        interpolated_subsampling_layer_factor=params_dict[
            "interpolated_subsampling_layer_factor"],
    )
    # encoder.printDimensions()
    return encoder


def getDecoderModel(model, encoder):
    # print("# getDecoderModel() #")
    params_dict = getAutoencoderArchitecture(model)
    state_dim_after_mlp = encoder.state_dim_before_mlp
    decoder = ConvMLPDecoder(
        latent_state_dim=params_dict["latent_state_dim"],
        state_dim_after_mlp=state_dim_after_mlp,
        conv_layers_kernel_sizes=params_dict["conv_layers_kernel_sizes"],
        conv_layers_channels=params_dict["conv_layers_channels"],
        conv_layers_strides=params_dict["conv_layers_strides"],
        pool_kernel_strides_encoder=params_dict["pool_kernel_strides"],
        upsampling_factors=params_dict["upsampling_factors"],
        upsampling_mode=params_dict["upsampling_mode"],
        activation=params_dict["activation"],
        activation_output=params_dict["activation_output"],
        batch_norm=params_dict["batch_norm"],
        batch_norm_affine=params_dict["batch_norm_affine"],
        conv_transpose=params_dict["conv_transpose"],
        input_chanels=params_dict["input_chanels"],
        input_image_height=params_dict["input_image_height"],
        input_image_width=params_dict["input_image_width"],
        dropout_keep_prob=params_dict["dropout_keep_prob"],
        interpolated_subsampling_layer_factor=params_dict[
            "interpolated_subsampling_layer_factor"],
    )
    return decoder


def getSamePadding(stride, image_size, filter_size):
    # Input image (W_i,W_i)
    # Output image (W_o,W_o) with W_o = (W_i - F + 2P)/S + 1
    # W_i == W_o -> P = ((S-1)W + F - S)/2
    S = stride
    W = image_size  # width or height
    F = filter_size
    half_pad = int((S - 1) * W - S + F)
    if half_pad % 2 == 1:
        raise ValueError(
            "(S-1) * W  - S + F has to be divisible by two ({:}-1)*{:} - {:} + {:} = {:}"
            .format(S, W, S, F, half_pad))
    else:
        pad = int(half_pad / 2)
    if (pad > image_size / 2):
        raise ValueError(
            "Very large padding P={:}, compared to input width {:}. Reduce the strides."
            .format(pad, image_size))
    return pad


class ConvMLPEncoder(nn.Module):
    def __init__(
        self,
        latent_state_dim=None,
        conv_layers_kernel_sizes=None,
        conv_layers_channels=None,
        conv_layers_strides=None,
        pool_kernel_sizes=None,
        pool_kernel_strides=None,
        pool_type="avg",
        activation=None,
        activation_output=None,
        batch_norm=None,
        batch_norm_affine=False,
        input_chanels=None,
        input_image_height=None,
        input_image_width=None,
        torch_dtype=torch.DoubleTensor,
        zero_padding=True,
        dropout_keep_prob=1.0,
        interpolated_subsampling_layer_factor=None,
    ):
        super(ConvMLPEncoder, self).__init__()

        assert (latent_state_dim)
        self.latent_state_dim = latent_state_dim
        self.dropout_keep_prob = dropout_keep_prob

        self.input_chanels = input_chanels
        self.activation = activation
        self.activation_output = activation_output
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width
        self.interpolated_subsampling_layer_factor = interpolated_subsampling_layer_factor

        self.conv_layers_kernel_sizes = conv_layers_kernel_sizes
        self.conv_layers_channels = conv_layers_channels
        self.conv_layers_strides = conv_layers_strides
        self.zero_padding = zero_padding

        self.batch_norm = batch_norm
        self.batch_norm_affine = batch_norm_affine

        if not (self.input_image_height is not None):
            raise ValueError(
                "In case of CNN-MLP-AE reducing the dimensionality, the height of the input image has to be provided."
            )
        if not (self.input_image_width is not None):
            raise ValueError(
                "In case of CNN-MLP-AE reducing the dimensionality, the width of the input image has to be provided."
            )

        self.layers = []
        self.layers_idx = []
        self.layers_num = 0

        if not (self.interpolated_subsampling_layer_factor == None):
            input_image_height = int(
                self.input_image_height /
                self.interpolated_subsampling_layer_factor[0])
            input_image_width = int(
                self.input_image_width /
                self.interpolated_subsampling_layer_factor[1])

            size_sub = (input_image_height, input_image_width)
            self.layers.append(
                interpolation_layer.interpolationLayer(size_sub))
            self.layers_num += 1
            self.layers_idx.append(self.layers_num - 1)

        if self.zero_padding:
            print("[CNN-MLP-AE] with padding to keep the dimensionality.")
            self.conv_layers_zero_padding = []
            for i in range(len(self.conv_layers_kernel_sizes)):
                pad_x = getSamePadding(
                    self.conv_layers_strides[i],
                    input_image_width,
                    self.conv_layers_kernel_sizes[i],
                )
                pad_y = getSamePadding(
                    self.conv_layers_strides[i],
                    input_image_height,
                    self.conv_layers_kernel_sizes[i],
                )
                # (padding_left, padding_right, padding_top, padding_bottom)
                self.conv_layers_zero_padding.append(
                    tuple([pad_x, pad_x, pad_y, pad_y]))

        self.conv_layers_channels.insert(0, self.input_chanels)

        self.pool_kernel_sizes = pool_kernel_sizes
        self.pool_kernel_strides = pool_kernel_strides
        self.pool_type = pool_type

        if self.pool_type not in ["max", "avg"]:
            raise ValueError("Invalid pooling operation type {:}.".format(
                self.pool_type))

        for i in range(len(self.conv_layers_kernel_sizes)):
            if self.zero_padding:
                self.layers.append(
                    nn.ZeroPad2d(padding=self.conv_layers_zero_padding[i]))
                self.layers_num += 1
            self.layers.append(
                nn.Conv2d(
                    kernel_size=self.conv_layers_kernel_sizes[i],
                    in_channels=self.conv_layers_channels[i],
                    out_channels=self.conv_layers_channels[i + 1],
                    stride=self.conv_layers_strides[i],
                ))
            self.layers_num += 1

            if self.batch_norm:
                if i < len(self.conv_layers_kernel_sizes) - 1:
                    self.layers.append(
                        nn.BatchNorm2d(self.conv_layers_channels[i + 1],
                                       affine=self.batch_norm_affine,
                                       momentum=0.1,
                                       track_running_stats=True))
                    self.layers_num += 1

            # POOLING OPERATION
            if self.pool_type == "max":
                self.layers.append(
                    nn.MaxPool2d(self.pool_kernel_sizes[i],
                                 stride=self.pool_kernel_strides[i]))
                self.layers_num += 1
            elif self.pool_type == "avg":
                self.layers.append(
                    nn.AvgPool2d(self.pool_kernel_sizes[i],
                                 stride=self.pool_kernel_strides[i]))
                self.layers_num += 1
            else:
                raise ValueError("Invalid pooling operation.")

            act_ = activations.getActivation(self.activation_output) if (
                i == len(self.conv_layers_kernel_sizes) -
                1) else activations.getActivation(self.activation)
            self.layers.append(act_)
            self.layers_num += 1

            if self.dropout_keep_prob < 1.0:
                self.layers.append(nn.Dropout(p=1 - self.dropout_keep_prob))
                self.layers_num += 1

            self.layers_idx.append(self.layers_num - 1)

        self.layers = nn.ModuleList(self.layers)

        # Adding the MLP layer
        sizes = self.getDimensions()
        temp = sizes[-1]
        self.state_dim_before_mlp = temp

        in_features = np.prod(self.state_dim_before_mlp)
        out_features = self.latent_state_dim

        self.layers.append(torch.nn.Flatten(start_dim=-3, end_dim=-1))
        self.layers_num += 1

        self.layers.append(
            torch.nn.Linear(in_features, out_features, bias=True))
        self.layers_num += 1
        self.layers_idx.append(self.layers_num - 1)

        act_ = activations.getActivation(self.activation_output)
        self.layers.append(act_)
        self.layers_num += 1

        sizes = self.getDimensions()

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
            self.input_chanels,
            self.input_image_height,
            self.input_image_width,
        )
        sizes_all = []
        input_size = list(input.size()[1:])
        for layer in self.layers:
            # print(layer)
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
        print("[Encoder] -------------------")
        print("[Encoder] printDimensions() #")
        sizes = self.getDimensions()
        for size in sizes:
            print("[Encoder] {:}".format(size))
        print("[Encoder] -------------------")
        return sizes


class ConvMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_state_dim=None,
        state_dim_after_mlp=None,
        conv_layers_kernel_sizes=None,
        conv_layers_channels=None,
        conv_layers_strides=None,
        pool_kernel_strides_encoder=None,
        upsampling_factors=None,
        upsampling_mode=None,
        activation=None,
        activation_output=None,
        batch_norm=None,
        batch_norm_affine=False,
        conv_transpose=None,
        input_chanels=None,
        input_image_height=None,
        input_image_width=None,
        torch_dtype=torch.DoubleTensor,
        zero_padding=True,
        dropout_keep_prob=1.0,
        interpolated_subsampling_layer_factor=None,
    ):
        super(ConvMLPDecoder, self).__init__()

        assert (latent_state_dim)
        assert (state_dim_after_mlp)
        self.latent_state_dim = latent_state_dim
        self.state_dim_after_mlp = state_dim_after_mlp

        self.dropout_keep_prob = dropout_keep_prob

        self.input_chanels = input_chanels
        self.activation = activation
        self.activation_output = activation_output
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width

        self.conv_layers_kernel_sizes = conv_layers_kernel_sizes[::-1]
        self.conv_layers_channels = conv_layers_channels[::-1]
        self.conv_layers_strides = conv_layers_strides[::-1]
        self.pool_kernel_strides_encoder = pool_kernel_strides_encoder[::-1]

        self.upsampling_factors = upsampling_factors
        self.upsampling_mode = upsampling_mode
        self.interpolated_subsampling_layer_factor = interpolated_subsampling_layer_factor

        self.zero_padding = zero_padding

        self.batch_norm = batch_norm
        self.batch_norm_affine = batch_norm_affine
        self.conv_transpose = conv_transpose

        print("[CNN-Decoder] with padding to keep the dimensionality.")
        if not (self.input_image_height is not None):
            raise ValueError(
                "In case of AE reducing the dimensionality, the height of the input image has to be provided."
            )
        if not (self.input_image_width is not None):
            raise ValueError(
                "In case of AE reducing the dimensionality, the width of the input image has to be provided."
            )

        if not (self.interpolated_subsampling_layer_factor == None):
            input_image_height = int(
                self.input_image_height /
                self.interpolated_subsampling_layer_factor[0])
            input_image_width = int(
                self.input_image_width /
                self.interpolated_subsampling_layer_factor[1])
            # Upsampling to the original dimension
            size_sub = (self.input_image_height, self.input_image_width)

        if self.zero_padding:
            self.conv_layers_zero_padding = []
            for i in range(len(self.conv_layers_kernel_sizes)):
                pad_x = getSamePadding(
                    self.conv_layers_strides[i],
                    input_image_width,
                    self.conv_layers_kernel_sizes[i],
                )
                pad_y = getSamePadding(
                    self.conv_layers_strides[i],
                    input_image_height,
                    self.conv_layers_kernel_sizes[i],
                )
                # (padding_left, padding_right, padding_top, padding_bottom)
                self.conv_layers_zero_padding.append(
                    tuple([pad_x, pad_x, pad_y, pad_y]))

        self.conv_layers_channels.append(self.input_chanels)

        # First padding layer
        self.layers = []
        self.layers_idx = []
        self.layers_num = 0

        # Add the linear layer
        in_features = self.latent_state_dim
        out_features = np.prod(self.state_dim_after_mlp)
        self.layers.append(
            torch.nn.Linear(in_features, out_features, bias=True))
        self.layers_num += 1

        act_ = activations.getActivation(self.activation)
        self.layers.append(act_)
        self.layers_num += 1

        temp = [-1] + list(self.state_dim_after_mlp)
        self.layers.append(ViewModule(temp))
        self.layers_num += 1
        self.layers_idx.append(self.layers_num - 1)

        for i in range(len(self.conv_layers_kernel_sizes)):

            if not self.conv_transpose:
                """ No upsampling layers in transpose convolution """
                if isinstance(self.upsampling_factors[i], list):
                    upsampling_factors = tuple(self.upsampling_factors[i])
                else:
                    upsampling_factors = self.upsampling_factors[i]

                self.layers.append(
                    nn.Upsample(scale_factor=upsampling_factors,
                                mode=self.upsampling_mode,
                                align_corners=True))
                self.layers_num += 1

            if self.conv_transpose:
                # Transforming the padding
                if self.zero_padding:
                    padding = self.getConv2dPadding(
                        self.conv_layers_zero_padding[i])
                else:
                    padding = 0

                assert (self.conv_layers_strides[0] == 1)
                self.layers.append(
                    nn.ConvTranspose2d(
                        kernel_size=self.conv_layers_kernel_sizes[i],
                        in_channels=self.conv_layers_channels[i],
                        out_channels=self.conv_layers_channels[i + 1],
                        stride=self.pool_kernel_strides_encoder[i],
                        padding=padding,
                        output_padding=1,
                        # padding=self.conv_layers_zero_padding[i],
                    ))
                self.layers_num += 1

            else:
                if self.zero_padding:
                    padding = self.getConv2dPadding(
                        self.conv_layers_zero_padding[i])
                else:
                    padding = 0

                self.layers.append(
                    nn.Conv2d(
                        kernel_size=self.conv_layers_kernel_sizes[i],
                        in_channels=self.conv_layers_channels[i],
                        out_channels=self.conv_layers_channels[i + 1],
                        stride=self.conv_layers_strides[i],
                        padding=padding,
                        # padding=self.conv_layers_zero_padding[i],
                    ))
                self.layers_num += 1

            if self.batch_norm:
                if i < len(self.conv_layers_kernel_sizes) - 1:
                    self.layers.append(
                        nn.BatchNorm2d(self.conv_layers_channels[i + 1],
                                       affine=self.batch_norm_affine,
                                       momentum=0.1,
                                       track_running_stats=False))
                    self.layers_num += 1

            # Adding the interpolation layer, before the activation

            if (i == len(self.conv_layers_kernel_sizes) - 1):
                # Last layer
                if not (self.interpolated_subsampling_layer_factor == None):
                    self.layers.append(
                        interpolation_layer.interpolationLayer(size_sub))
                    self.layers_num += 1
                    self.layers_idx.append(self.layers_num - 1)

            act_ = activations.getActivation(self.activation_output) if (
                i == len(self.conv_layers_kernel_sizes) -
                1) else activations.getActivation(self.activation)
            self.layers.append(act_)
            self.layers_num += 1

            if (i < len(self.conv_layers_kernel_sizes) - 1):
                if self.dropout_keep_prob < 1.:
                    self.layers.append(nn.Dropout(p=1 -
                                                  self.dropout_keep_prob))
                    self.layers_num += 1

            self.layers_idx.append(self.layers_num - 1)

        self.layers = nn.ModuleList(self.layers)

    def getConv2dPadding(self, padding):
        padding_left, padding_right, padding_top, padding_bottom = padding
        return [padding_left, padding_top]

    def forward(self, input):
        # print("# DECODER: forward() #")
        # print(input.size())
        for layer in self.layers:
            # print("--------------")
            # print(layer)
            input = layer(input)
            # print(input.size())
        return input

    def getDimensions(self, encoder_sizes):
        latent_size = encoder_sizes[-1]
        input = torch.randn(1, *latent_size)
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

    def printDimensions(self, encoder_sizes):
        print("[Decoder] -------------------")
        print("[Decoder] # printDimensions() #")
        sizes = self.getDimensions(encoder_sizes)
        for size in sizes:
            print("[Decoder] {:}".format(size))
        print("[Decoder] -------------------")
        return sizes


if __name__ == '__main__':

    import activations

    input_chanels = 2
    activation = "celu"
    activation_output = "identity"

    latent_state_dim = 3

    # system_name = "CUP2D-cylRe-100"
    # input_image_height = 128
    # input_image_width = 256
    # conv_layers_kernel_sizes = [11, 9, 7]
    # conv_layers_channels = [4, 8, 16]
    # conv_layers_strides = [1, 1, 1]
    # pool_kernel_sizes = [2, 2, 2]
    # pool_kernel_strides = [2, 2, 2]
    # upsampling_factors = [2, 2, 2]
    # upsampling_factors = upsampling_factors[::-1]
    # upsampling_mode = "bilinear"
    # zero_padding = True

    # system_name = "CUP2D-cylRe-100-Vort"
    # input_image_height = 128
    # input_image_width = 256

    # conv_layers_kernel_sizes = [11, 9, 7, 5, 3]
    # conv_layers_channels = [4, 8, 16, 32, 1]
    # conv_layers_strides = [1, 1, 1, 1, 1]
    # pool_kernel_sizes = [2, 4, 2, 2, 2]
    # pool_kernel_strides = [2, 4, 2, 2, 2]
    # upsampling_factors = [2, 4, 2, 2, 2]

    # system_name = "CUP2D-cylRe-100-Vort"
    # input_image_height = 256
    # input_image_width = 512
    # AE_size_factor = 2

    # conv_layers_kernel_sizes = [11, 9, 7, 5, 3]
    # conv_layers_channels = [4, 8, 16, 32, 1]
    # conv_layers_channels = list(AE_size_factor*np.array(conv_layers_channels))
    # conv_layers_strides = [1, 1, 1, 1, 1]
    # pool_kernel_sizes = [2, 4, 2, 2, 2]
    # pool_kernel_strides = [2, 4, 2, 2, 2]
    # upsampling_factors = [2, 4, 2, 2, 2]

    # system_name = "Dummy"
    # input_image_height = 8
    # input_image_width = 8
    # AE_size_factor = 2

    # conv_layers_kernel_sizes = [3, 3, 3]
    # conv_layers_channels = [2, 2, 2]
    # conv_layers_channels = list(AE_size_factor*np.array(conv_layers_channels))
    # conv_layers_strides = [1, 1, 1]
    # pool_kernel_sizes = [2, 2, 2]
    # pool_kernel_strides = [2, 2, 2]
    # upsampling_factors = [2, 2, 2]

    # upsampling_factors = upsampling_factors[::-1]
    # upsampling_mode = "bilinear"
    # zero_padding = True
    # input = torch.randn(1, input_chanels, input_image_height, input_image_width)

    # encoder = ConvMLPEncoder(
    #     latent_state_dim=latent_state_dim,
    #     conv_layers_kernel_sizes=conv_layers_kernel_sizes,
    #     conv_layers_channels=conv_layers_channels,
    #     conv_layers_strides=conv_layers_strides,
    #     pool_kernel_sizes=pool_kernel_sizes,
    #     pool_kernel_strides=pool_kernel_strides,
    #     activation=activation,
    #     activation_output=activation_output,
    #     input_chanels=input_chanels,
    #     input_image_height=input_image_height,
    #     input_image_width=input_image_width,
    #     zero_padding=zero_padding,
    # )
    # sizes = encoder.printDimensions()

    # print("###############")
    # print("input.size()")
    # print(input.size())
    # latent = encoder.forward(input)
    # print("latent = encoder.forward(input)")
    # print(latent.size())

    # decoder = ConvMLPDecoder(
    #     latent_state_dim=latent_state_dim,
    #     state_dim_after_mlp=encoder.state_dim_before_mlp,
    #     conv_layers_kernel_sizes=conv_layers_kernel_sizes,
    #     conv_layers_channels=conv_layers_channels,
    #     conv_layers_strides=conv_layers_strides,
    #     upsampling_factors=upsampling_factors,
    #     upsampling_mode=upsampling_mode,
    #     activation=activation,
    #     activation_output=activation_output,
    #     input_chanels=input_chanels,
    #     input_image_height=input_image_height,
    #     input_image_width=input_image_width,
    # )
    # sizes = decoder.printDimensions(sizes)

    # print("###############")
    # print("latent.size()")
    # print(latent.size())
    # output = decoder.forward(latent)
    # print("output = decoder.forward(latent)")
    # print(output.size())

    # system_name = "Dummy"
    # input_image_height = 512
    # input_image_width = 1024
    # AE_size_factor = 1

    # conv_layers_kernel_sizes = [11, 11, 7, 5]
    # conv_layers_channels = [8, 16, 16, 8]
    # conv_layers_channels = list(AE_size_factor*np.array(conv_layers_channels))
    # conv_layers_strides = [1, 1, 1, 1]
    # pool_kernel_sizes = [4, 4, 4, 4]
    # pool_kernel_strides = pool_kernel_sizes

    # upsampling_factors = [4, 4, 4, 4]
    # upsampling_mode = "bilinear"

    system_name = "Dummy"
    input_image_height = 512
    input_image_width = 1024
    AE_size_factor = 1

    conv_layers_kernel_sizes = [13, 13, 13, 13, 13, 13]
    conv_layers_channels = [8, 8, 8, 8, 8, 1]
    conv_layers_channels = list(np.array(conv_layers_channels))
    conv_layers_strides = [1, 1, 1, 1, 1, 1]
    pool_kernel_sizes = [2, 2, 2, 2, 2, 2]
    pool_kernel_strides = pool_kernel_sizes
    upsampling_factors = [2, 2, 2, 2, 2, 2]
    upsampling_mode = "bilinear"
    upsampling_factors = upsampling_factors[::-1]
    interpolated_subsampling_layer_factor = (2, 2)

    # conv_layers_kernel_sizes = [13, 13, 13]
    # conv_layers_channels = [4, 4, 4, 1]
    # conv_layers_channels = list(AE_size_factor*np.array(conv_layers_channels))
    # conv_layers_strides = [1, 1, 1, 1]
    # pool_kernel_sizes = [4, 4, 4, 4]
    # pool_kernel_strides = pool_kernel_sizes
    # upsampling_factors = [4, 4, 4, 4]
    # upsampling_mode = "bilinear"
    # interpolated_subsampling_layer_factor = (2, 2)

    zero_padding = True
    input = torch.randn(1, input_chanels, input_image_height,
                        input_image_width)

    encoder = ConvMLPEncoder(
        latent_state_dim=latent_state_dim,
        conv_layers_kernel_sizes=conv_layers_kernel_sizes,
        conv_layers_channels=conv_layers_channels,
        conv_layers_strides=conv_layers_strides,
        pool_kernel_sizes=pool_kernel_sizes,
        pool_kernel_strides=pool_kernel_strides,
        activation=activation,
        activation_output=activation_output,
        input_chanels=input_chanels,
        input_image_height=input_image_height,
        input_image_width=input_image_width,
        zero_padding=zero_padding,
        interpolated_subsampling_layer_factor=
        interpolated_subsampling_layer_factor,
    )
    sizes = encoder.printDimensions()

    print("###############")
    print("input.size()")
    print(input.size())
    latent = encoder.forward(input)
    print("latent = encoder.forward(input)")
    print(latent.size())

    decoder = ConvMLPDecoder(
        latent_state_dim=latent_state_dim,
        state_dim_after_mlp=encoder.state_dim_before_mlp,
        conv_layers_kernel_sizes=conv_layers_kernel_sizes,
        conv_layers_channels=conv_layers_channels,
        conv_layers_strides=conv_layers_strides,
        upsampling_factors=upsampling_factors,
        upsampling_mode=upsampling_mode,
        activation=activation,
        activation_output=activation_output,
        input_chanels=input_chanels,
        input_image_height=input_image_height,
        input_image_width=input_image_width,
        interpolated_subsampling_layer_factor=
        interpolated_subsampling_layer_factor,
    )
    sizes = decoder.printDimensions(sizes)

    print("###############")
    print("latent.size()")
    print(latent.size())
    output = decoder.forward(latent)
    print("output = decoder.forward(latent)")
    print(output.size())

    print(encoder.conv_layers_zero_padding)
    print(decoder.conv_layers_zero_padding)
