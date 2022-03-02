#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
""" Torch """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.distributions as tdist

import numpy as np

from . import zoneoutlayer
from . import conv_mlp_model_1D
from . import conv_mlp_model_2D
from . import mlp_autoencoders
from . import activations
from . import conv_rnn_cell_1D
from . import conv_rnn_cell_2D
from . import beta_vae
from . import dummy
""" Printing """
from functools import partial

print = partial(print, flush=True)

import time

from .. import Utils as utils

#################################
# Backpropagation Through Time
# Autoregressive module - Normal import
from . import bptta_module
#################################


class dimred_rnn_model(nn.Module):
    def __init__(self, params, model):
        super(dimred_rnn_model, self).__init__()
        self.parent = model

        if not self.parent.RNN_convolutional:
            # Determining cell type
            if self.parent.params["RNN_cell_type"] == "lstm":
                self.RNN_cell = nn.LSTMCell
            elif self.parent.params["RNN_cell_type"] == "gru":
                self.RNN_cell = nn.GRUCell
            elif self.parent.params["RNN_cell_type"] == "mlp":
                """ in case of dummy MLP propagator """
                pass
            else:
                raise ValueError("Invalid RNN_cell_type {:}".format(
                    params["RNN_cell_type"]))

        self.activation_str_general = params["activation_str_general"]

        if self.activation_str_general not in [
                "relu", "celu", "elu", "selu", "tanh"
        ]:  # WITH AUTOENCODER
            raise ValueError("Invalid general activation {:}".format(
                self.activation_str_general))
        self.buildNetwork()
        """ Changing the type of the network modules """
        if (self.parent.torch_dtype
                == torch.DoubleTensor) or (self.parent.torch_dtype
                                           == torch.cuda.DoubleTensor):
            for modules in self.module_list:
                modules.double()

        self.has_latent_scaler = self.hasLatentStateScaler()

        # Define latent state scaler
        if self.has_latent_scaler:
            self.defineLatentStateParams()

        if self.parent.has_autoencoder:
            self.defineOutputResidualScalerParams()

        if self.parent.params["precision"] == "double":
            self.double()
            self.setModelToPrecision(self.parent.params["precision"])
        elif self.parent.params["precision"] == "single":
            self.float()  # model.half()
            self.setModelToPrecision(self.parent.params["precision"])
        else:
            raise ValueError("Invalid precision {:}.".format(
                self.parent.params["precision"]))

        if self.parent.gpu:
            self.sendModelToCuda()
            self.cuda()

    def setModelToPrecision(self, precision="double"):
        self.print(
            "[dimred_rnn_model] setModelToPrecision({:})".format(precision))
        # for modules in [self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT]:
        for modules in self.module_list:
            for layer in modules:
                # print(layer)
                if precision == "single":
                    layer.float()
                else:
                    layer.double()
                # if isinstance(layer, nn.BatchNorm2d):
                #     print("Detected Batchnorm layer.")
        return 0

    def hasLatentStateScaler(self):
        # if self.parent.has_rnn and self.parent.has_autoencoder and self.parent.AE_convolutional and (not self.parent.RNN_convolutional) and self.parent.load_trained_AE:
        if self.parent.has_rnn and (self.parent.has_autoencoder
                                    or self.parent.has_dimred) and (
                                        not self.parent.RNN_convolutional
                                    ) and self.parent.load_trained_AE:
            return True
        else:
            return False

    def print(self, str_):
        if self.parent.multinode:
            print("{:}{:}".format(self.parent.rank_str, str_))
        else:
            print(str_)

    def ifAnyIn(self, list_, name):
        for element in list_:
            if element in name:
                return True
        return False

    def augmentRNNwithTrainableInitialHiddenState(self):
        for ln in range(len(self.RNN)):
            hidden_channels = self.RNN[ln].hidden_channels
            self.RNN[
                ln].initial_hidden_state = self.defineRNNInitialHiddenStateLayer(
                    hidden_channels)

    def getZeroState(self, hidden_units):
        if self.parent.channels == 2:
            return torch.zeros(hidden_units, self.parent.Dx, self.parent.Dy)
        elif self.parent.channels == 1:
            return torch.zeros(hidden_units, self.parent.Dx)

    def defineRNNInitialHiddenStateLayer(self, hidden_units):
        if self.RNN_cell_type == "mlp": return torch.zeros(1)
        if self.parent.RNN_convolutional:
            hx = torch.nn.Parameter(data=self.getZeroState(hidden_units),
                                    requires_grad=True)
            if self.parent.params["RNN_cell_type"] == "lstm":
                cx = torch.nn.Parameter(data=self.getZeroState(hidden_units),
                                        requires_grad=True)
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.parent.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(
                    self.parent.params["RNN_cell_type"]))
        else:
            hx = torch.nn.Parameter(data=torch.zeros(hidden_units),
                                    requires_grad=True)
            if self.parent.params["RNN_cell_type"] == "lstm":
                cx = torch.nn.Parameter(data=torch.zeros(hidden_units),
                                        requires_grad=True)
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.parent.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(
                    self.parent.params["RNN_cell_type"]))

    def initializeWeights(self):
        self.print("[dimred_rnn_model] Initializing parameters.")
        for modules in self.module_list:
            for module in modules:
                for name, param in module.named_parameters():
                    """ Initializing RNN, GRU, LSTM cells """
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)

                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn([
                            "Wxi.weight",
                            "Wxf.weight",
                            "Wxc.weight",
                            "Wxo.weight",
                            "Bci",
                            "Bcf",
                            "Bco",
                            "Bcc",
                    ], name):
                        torch.nn.init.xavier_uniform_(param.data)

                    elif self.ifAnyIn([
                            "Wco", "Wcf", "Wci", "Whi.weight", "Whf.weight",
                            "Whc.weight", "Who.weight"
                    ], name):
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn([
                            "Whi.bias", "Wxi.bias", "Wxf.bias", "Whf.bias",
                            "Wxc.bias", "Whc.bias", "Wxo.bias", "Who.bias"
                    ], name):
                        param.data.fill_(0.001)

                    elif 'weight' in name:
                        ndim = len(list(param.data.size()))
                        if ndim > 1:
                            torch.nn.init.xavier_uniform_(param.data)
                        else:
                            self.print(
                                "[dimred_rnn_model] Module {:}, Params {:} default initialization."
                                .format(module, name))

                    elif 'bias' in name:
                        param.data.fill_(0.001)

                    elif 'initial_hidden_state' in name:
                        param.data.fill_(0.00001)

                    else:
                        raise ValueError(
                            "[dimred_rnn_model] NAME {:} NOT FOUND!".format(
                                name))

        self.print("[dimred_rnn_model] Parameters initialized.")
        return 0

    def sendModelToCPU(self):
        self.print("[dimred_rnn_model] Sending model to CPU.")
        for modules in self.module_list:
            for model in modules:
                model.cpu()
        return 0

    def sendModelToCuda(self):
        self.print("[dimred_rnn_model] Sending model to CUDA.")
        for modules in self.module_list:
            for model in modules:
                model.cuda()
        return 0

    def buildNetwork(self):
        # self.DROPOUT = nn.ModuleList()
        # self.DROPOUT.append(nn.Dropout(p=1 - self.parent.dropout_keep_prob))

        if self.parent.has_autoencoder:
            # Building the layers of the encoder
            if not self.parent.AE_convolutional:
                encoder = mlp_autoencoders.MLPEncoder(
                    channels=self.parent.channels,
                    input_dim=self.parent.input_dim,
                    Dx=self.parent.Dx,
                    Dy=self.parent.Dy,
                    output_dim=self.parent.latent_state_dim_auto,
                    activation=self.parent.params["activation_str_general"],
                    activation_output=self.parent.
                    params["activation_str_general"],
                    layers_size=self.parent.layers_encoder,
                    dropout_keep_prob=self.parent.params["dropout_keep_prob"],
                    torch_dtype=self.parent.torch_dtype,
                )
                self.ENCODER = encoder.layers

            elif self.parent.AE_convolutional:

                if self.parent.channels == 1:
                    encoder = conv_mlp_model_1D.getEncoderModel(self.parent)
                elif self.parent.channels == 2:
                    encoder = conv_mlp_model_2D.getEncoderModel(self.parent)
                else:
                    raise ValueError("Not implemented.")

                self.ENCODER = encoder.layers

        elif self.parent.has_dummy_autoencoder:
            self.ENCODER = nn.ModuleList()
            self.ENCODER.append(dummy.viewEliminateChannels(
                self.parent.params))
        else:
            self.ENCODER = nn.ModuleList()

        self.BETA_VAE = nn.ModuleList()
        if self.parent.has_autoencoder and self.parent.beta_vae:
            temp = beta_vae.beta_vae(embedding_size=self.latent_state_dim_auto)
            self.BETA_VAE.append(temp)

        self.RNN = nn.ModuleList()
        if self.parent.has_rnn:
            if self.parent.RNN_cell_type == "mlp":
                """ dummy wrapper to implement an MLP ignoring the hidden state """
                from . import rnn_mlp_wrapper
                input_size = self.parent.params["RNN_state_dim"]
                self.RNN.append(
                    rnn_mlp_wrapper.RNN_MLP_wrapper(
                        input_size=input_size,
                        output_size=input_size,
                        hidden_sizes=self.parent.layers_rnn,
                        act=self.parent.RNN_activation_str,
                        act_output=self.parent.RNN_activation_str_output,
                    ))

            else:
                """ normal recurrent neural network """
                if not self.parent.RNN_convolutional:
                    # Parsing the layers of the RNN
                    # The input to the RNN is an embedding (effective dynamics) vector, or latent vector
                    input_size = self.parent.params["RNN_state_dim"]
                    for ln in range(len(self.parent.layers_rnn)):
                        self.RNN.append(
                            zoneoutlayer.ZoneoutLayer(
                                self.RNN_cell(
                                    input_size=input_size,
                                    hidden_size=self.parent.layers_rnn[ln]),
                                self.parent.zoneout_keep_prob))
                        input_size = self.parent.layers_rnn[ln]
                else:
                    input_size = self.parent.params["RNN_state_dim"]
                    # Convolutional RNN Cell
                    for ln in range(len(self.parent.layers_rnn)):
                        if self.parent.channels == 1:
                            self.RNN.append(
                                conv_rnn_cell_1D.ConvRNNCell(
                                    input_size,
                                    self.parent.layers_rnn[ln],
                                    self.parent.RNN_kernel_size,
                                    activation=self.parent.RNN_activation_str,
                                    cell_type=self.parent.RNN_cell_type,
                                    torch_dtype=self.parent.torch_dtype))
                        elif self.parent.channels == 2:
                            self.RNN.append(
                                conv_rnn_cell_2D.ConvRNNCell(
                                    input_size,
                                    self.parent.layers_rnn[ln],
                                    self.parent.RNN_kernel_size,
                                    activation=self.parent.RNN_activation_str,
                                    cell_type=self.parent.RNN_cell_type,
                                    torch_dtype=self.parent.torch_dtype))
                        input_size = self.parent.layers_rnn[ln]

        # Output MLP of the RNN
        self.RNN_OUTPUT = nn.ModuleList()
        if self.parent.has_rnn:

            if self.parent.RNN_cell_type == "mlp":

                self.RNN_OUTPUT.extend([nn.Identity()])

            else:
                if not self.parent.RNN_convolutional:
                    self.RNN_OUTPUT.extend([
                        nn.Linear(
                            self.parent.layers_rnn[-1],
                            self.parent.params["RNN_state_dim"],
                            bias=True,
                        )
                    ])
                    # Here RNN is in the latent space (general activation string)
                    self.RNN_OUTPUT.extend([
                        activations.getActivation(
                            self.parent.RNN_activation_str_output)
                    ])
                else:
                    if self.parent.channels == 2:
                        # padding_same = int((self.parent.RNN_kernel_size - 1) / 2)
                        self.RNN_OUTPUT.extend([
                            # nn.Conv2d(self.parent.RNN_layers_size, self.parent.input_dim, self.parent.RNN_kernel_size, stride=1, padding=padding_same, bias=True)
                            nn.Conv2d(self.parent.RNN_layers_size,
                                      self.parent.input_dim,
                                      1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
                        ])
                    elif self.parent.channels == 1:
                        self.RNN_OUTPUT.extend([
                            nn.Conv1d(self.parent.RNN_layers_size,
                                      self.parent.input_dim,
                                      1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
                        ])
                    # Here RNN is at the output (output activation string)
                    self.RNN_OUTPUT.extend([
                        activations.getActivation(
                            self.parent.RNN_activation_str_output)
                    ])

        if self.parent.has_autoencoder:
            # Building the layers of the decoder (additional input is the latent noise)
            if not self.parent.AE_convolutional:
                decoder = mlp_autoencoders.MLPDecoder(
                    channels=self.parent.channels,
                    input_dim=self.parent.latent_state_dim_auto,
                    Dx=self.parent.Dx,
                    Dy=self.parent.Dy,
                    output_dim=self.parent.input_dim,
                    activation=self.parent.params["activation_str_general"],
                    activation_output=self.parent.
                    params["activation_str_output"],
                    layers_size=self.parent.layers_decoder,
                    dropout_keep_prob=self.parent.params["dropout_keep_prob"],
                    torch_dtype=self.parent.torch_dtype,
                )
                self.DECODER = decoder.layers

            elif self.parent.AE_convolutional:

                if self.parent.channels == 1:
                    decoder = conv_mlp_model_1D.getDecoderModel(
                        self.parent, encoder)
                elif self.parent.channels == 2:
                    decoder = conv_mlp_model_2D.getDecoderModel(
                        self.parent, encoder)
                else:
                    raise ValueError("Not implemented.")
                self.DECODER = decoder.layers
        elif self.parent.has_dummy_autoencoder:
            self.DECODER = nn.ModuleList()
            self.DECODER.append(dummy.viewAddChannels(self.parent.params))
        else:
            self.DECODER = nn.ModuleList()

        # Initial RNN hidden states (in case of trainable)
        if self.parent.has_rnn and self.parent.RNN_trainable_init_hidden_state:
            self.augmentRNNwithTrainableInitialHiddenState()

        self.module_list = [
            # self.DROPOUT, self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT
            self.ENCODER,
            self.BETA_VAE,
            self.DECODER,
            self.RNN,
            self.RNN_OUTPUT
        ]

        if self.parent.has_autoencoder and self.parent.AE_convolutional:
            # self.print("[dimred_rnn_model] # AUTOENCODER #")
            # sizes = encoder.printDimensions()
            # sizes = decoder.printDimensions(sizes)
            # sizes = encoder.printDimensions()
            self.parent.params[
                "conv_layers_channels"] = encoder.conv_layers_channels
            self.parent.params[
                "conv_layers_kernel_sizes"] = encoder.conv_layers_kernel_sizes
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

    def getAutoencoderParams(self):
        params = list()
        named_params = list()
        for layers in [self.ENCODER, self.DECODER]:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params

    def getRNNParams(self):
        params = list()
        named_params = list()
        for layers in [self.RNN, self.RNN_OUTPUT]:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params

    def printModuleList(self):
        self.print("[dimred_rnn_model] module_list :")
        module_list_str = str(self.module_list)
        module_list_str_lines = module_list_str.split("\n")
        for line in module_list_str_lines:
            self.print("[dimred_rnn_model] " + line)
        return 0

    def eval(self):
        for modules in [
                # self.DROPOUT, self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT,
                self.ENCODER,
                self.BETA_VAE,
                self.DECODER,
                self.RNN,
                self.RNN_OUTPUT,
        ]:
            for layer in modules:
                layer.eval()
        return 0

    def train(self):
        for modules in [
                # self.DROPOUT, self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT,
                self.ENCODER,
                self.BETA_VAE,
                self.DECODER,
                self.RNN,
                self.RNN_OUTPUT,
        ]:
            for layer in modules:
                layer.train()
        return 0

    def getRnnHiddenState(self, batch_size):
        hidden_state = []
        for ln in range(len(self.RNN)):
            hidden_state_layer = self.RNN[ln].initial_hidden_state
            if self.parent.RNN_convolutional and self.parent.params[
                    "RNN_cell_type"] == "lstm":
                hx, cx = hidden_state_layer

                hx = hx.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                cx = cx.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                hidden_state_layer = torch.stack([hx, cx])

            elif self.parent.RNN_convolutional and self.parent.params[
                    "RNN_cell_type"] == "gru":
                hidden_state_layer = hidden_state_layer.unsqueeze(0).repeat(
                    batch_size, 1, 1, 1)
            else:
                raiseValueError("Not implemented.")
            hidden_state.append(hidden_state_layer)
        hidden_state = torch.stack(hidden_state)
        hidden_state = self.transposeHiddenState(hidden_state)
        return hidden_state

    def transposeHiddenState(self, hidden_state):
        # Transpose hidden state from batch_first to Layer first
        # (gru)  [K, L, H]    -> [L, K, H]
        # (lstm) [K, 2, L, H] -> [L, 2, K, H]
        if self.parent.params["RNN_cell_type"] == "gru":
            hidden_state = hidden_state.transpose(0, 1)  #
        elif "lstm" in self.parent.params["RNN_cell_type"]:
            hidden_state = hidden_state.transpose(0, 2)  # (lstm)
        elif self.parent.params["RNN_cell_type"] == "mlp":
            pass
        else:
            raise ValueError("RNN_cell_type {:} not recognized".format(
                self.parent.params["RNN_cell_type"]))
        return hidden_state

    def bptta_loss(self, input_batch, target_batch, latent_states,
                   initial_hidden_states, is_train):
        return bptta_module.bptta_loss(self, input_batch, target_batch,
                                       latent_states, initial_hidden_states,
                                       is_train)

    def forward(
        self,
        inputs,
        init_hidden_state,
        is_train=False,
        is_iterative_forecasting=False,
        iterative_forecasting_prob=0,
        iterative_forecasting_gradient=1,
        horizon=None,
        input_is_latent=False,
        iterative_propagation_is_latent=False,
        detach_output=False,
        # teacher_forcing_forecasting=0,
    ):
        # ARGUMENTS:
        #   inputs:                        The input to the RNN
        #   init_hidden_state:            The initial hidden state
        #   is_train:                    Whether it is training or evaluation
        #   is_iterative_forecasting:   Whether to feed the predicted output back in the input (iteratively)
        #   input_is_latent:            Whether the input is latent state/original state

        # ONLY RELEVANT FOR ITERATIVE FORECASTING (DURING THE TESTING PHASE)
        #   horizon:                            The iterative forecasting horizon
        #   teacher_forcing_forecasting:        The number of time-steps to teacher-force (warm-up) in iterative prediction
        #   iterative_propagation_is_latent:    To iteratively propagate the latent state or the output

        if is_iterative_forecasting:
            return self.forecast(
                inputs,
                init_hidden_state,
                horizon,
                is_train=is_train,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                input_is_latent=input_is_latent,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
            )
        else:
            assert (input_is_latent == iterative_propagation_is_latent)
            return self.forward_(
                inputs,
                init_hidden_state,
                is_train=is_train,
                is_latent=input_is_latent,
                detach_output=detach_output,
            )

    def forward_(
        self,
        inputs,
        init_hidden_state,
        is_train=True,
        is_latent=False,
        detach_output=False,
    ):

        # for modules in [self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT]:
        #     for module in modules:
        #         for name, param in module.named_parameters():
        #             print("DEVICE {:} - PARAM {:} DEVICE {:}".format(init_hidden_state.get_device(), name, param.get_device()))

        # TRANSPOSE FROM BATCH FIRST TO LAYER FIRST
        if self.parent.has_rnn:
            init_hidden_state = self.transposeHiddenState(init_hidden_state)

        if is_train:
            self.train()
        else:
            self.eval()

        if self.parent.has_rnn and self.parent.train_RNN:
            self.setBatchNormalizationLayersToEvalMode()

        # Time spent in propagation of the latent state
        time_latent_prop = 0.0
        with torch.set_grad_enabled(is_train):

            inputsize = inputs.size()
            K, T, D = inputsize[0], inputsize[1], inputsize[2]

            # Swapping the inputs to RNN [K,T,LD]->[T, K, LD] (time first) LD=latent dimension
            inputs = inputs.transpose(1, 0)

            if (K != self.parent.batch_size and is_train == True
                    and (not self.parent.device_count > 1)):
                raise ValueError(
                    "Batch size {:d} does not match {:d} and model not in multiple GPUs."
                    .format(K, self.parent.batch_size))

            if (self.parent.has_autoencoder
                    or self.parent.has_dummy_autoencoder
                    or self.parent.has_dimred) and not is_latent:
                if D != self.parent.input_dim:
                    raise ValueError(
                        "Input dimension {:d} does not match {:d}.".format(
                            D, self.parent.input_dim))

                # Forward the encoder only in the original space
                encoder_output = self.forwardEncoder(inputs)
            else:
                encoder_output = inputs

            if detach_output:
                del inputs
                encoder_output = encoder_output.detach()

            if self.parent.beta_vae:
                latent_state_shape = encoder_output.size()
                n_dims = len(list(encoder_output.size()))
                # print(encoder_output.size())
                encoder_output_flat = encoder_output.flatten(start_dim=2,
                                                             end_dim=n_dims -
                                                             1)
                # print(encoder_output_flat.size())
                z, beta_vae_mu, beta_vae_logvar = self.BETA_VAE[0](
                    encoder_output_flat)

                if (not is_train) or self.parent.has_rnn:
                    # In case of testing, or RNN training, use the mean value
                    decoder_input = torch.reshape(beta_vae_mu,
                                                  latent_state_shape)
                else:
                    decoder_input = torch.reshape(z, latent_state_shape)

            else:
                decoder_input = encoder_output
                beta_vae_mu = None
                beta_vae_logvar = None

            latent_states = encoder_output

            # print("##")
            # print(encoder_output.size())
            # print(ark)
            # print(decoder_input.size())
            # print(latent_states.size())
            # decoder_input = 0.0 * torch.ones_like(decoder_input)

            if not detach_output:
                if self.parent.has_autoencoder or self.parent.has_dummy_autoencoder or self.parent.has_dimred:
                    inputs_decoded = self.forwardDecoder(decoder_input)
                else:
                    inputs_decoded = decoder_input
            else:
                inputs_decoded = []

            # print(inputs_decoded.size())
            # print(ark)

            if self.parent.has_rnn:
                time0 = time.time()

                if not self.parent.beta_vae:
                    rnn_input = encoder_output
                else:
                    rnn_input = beta_vae_mu

                # print(rnn_input.size())
                # print(torch.amin(rnn_input, dim=(0,1)))
                # print(torch.amax(rnn_input, dim=(0,1)))

                # print(self.latent_state_min)
                # print(self.latent_state_max)

                if self.has_latent_scaler:
                    rnn_input = self.scaleLatentState(rnn_input)

                # print(torch.amax(rnn_input, dim=(0,1)))
                # print(torch.amin(rnn_input, dim=(0,1)))
                # # print(rnn_input)
                # print(ark)

                # Latent states are the autoencoded states BEFORE being past through the RNN
                RNN_outputs, next_hidden_state = self.forwardRNN(
                    rnn_input, init_hidden_state, is_train)

                # for name, param in self.RNN_OUTPUT[0].named_parameters():
                #     print("DEVICE {:} - PARAM {:} DEVICE {:}".format(init_hidden_state.get_device(), name, param.get_device()))

                # Output of the RNN passed through MLP
                # print(RNN_outputs)
                # outputs = self.RNN_OUTPUT[0](RNN_outputs)
                latent_states_pred = self.forwardRNNOutput(RNN_outputs)

                if self.has_latent_scaler:
                    latent_states_pred = self.descaleLatentState(
                        latent_states_pred)

                # print(latent_states_pred.size())
                # print(latent_states_pred.min())
                # print(latent_states_pred.max())

                time1 = time.time()
                time_latent_prop += (time1 - time0)

                # The predicted latent states are the autoencoded states AFTER being past through the RNN, before beeing decoded
                decoder_input_pred = latent_states_pred

                # print(decoder_input_pred.size())
                # print(latent_states_pred.size())
                # TRANSPOSING BATCH_SIZE WITH TIME
                latent_states_pred = latent_states_pred.transpose(
                    1, 0).contiguous()
                # print(latent_states_pred.size())
                # print(aek)
                RNN_outputs = RNN_outputs.transpose(1, 0).contiguous()

                # TRANSPOSE BACK FROM LAYER FIRST TO BATCH FIRST
                next_hidden_state = self.transposeHiddenState(
                    next_hidden_state)

                # Output of the RNN (after the MLP) has dimension
                # [T, K, latend_dim]

                if not detach_output:
                    if self.parent.has_autoencoder or self.parent.has_dummy_autoencoder or self.parent.has_dimred:
                        outputs = self.forwardDecoder(decoder_input_pred)
                        # print(outputs.size())
                        # print(decoder_input_pred.size())
                        # print(outputs.max())
                        # print(outputs.min())
                        outputs = outputs.transpose(1, 0).contiguous()
                    else:
                        outputs = latent_states_pred
                else:
                    outputs = []

            else:
                outputs = []
                RNN_outputs = []
                latent_states_pred = []
                next_hidden_state = []

            latent_states = latent_states.transpose(1, 0).contiguous()
            if not detach_output:
                inputs_decoded = inputs_decoded.transpose(1, 0).contiguous()
        # print("#####")
        # if torch.is_tensor(outputs): print("outputs.size() = {:}".format(outputs.size()))
        # if torch.is_tensor(latent_states): print("latent_states.size() = {:}".format(latent_states.size()))
        # if torch.is_tensor(latent_states_pred): print("latent_states_pred.size() = {:}".format(latent_states_pred.size()))
        # if torch.is_tensor(RNN_outputs): print("RNN_outputs.size() = {:}".format(RNN_outputs.size()))
        # if torch.is_tensor(inputs_decoded): print("inputs_decoded.size() = {:}".format(inputs_decoded.size()))
        # print("#####")
        # print(ark)

        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar

    def forecast(
        self,
        inputs,
        init_hidden_state,
        horizon=None,
        is_train=False,
        iterative_forecasting_prob=0,
        iterative_forecasting_gradient=1,
        iterative_propagation_is_latent=False,
        input_is_latent=False,
    ):
        if is_train:
            self.train()
        else:
            self.eval()

        if self.parent.has_rnn and self.parent.train_RNN:
            self.setBatchNormalizationLayersToEvalMode()

        if input_is_latent and not iterative_propagation_is_latent:
            raise ValueError(
                "input_is_latent and not iterative_propagation_is_latent Not implemented."
            )

        with torch.set_grad_enabled(is_train):
            # inputs is either the inputs of the encoder or the latent state when input_is_latent=True
            outputs = []
            inputs_decoded = []

            latent_states = []
            latent_states_pred = []
            RNN_internal_states = []
            RNN_outputs = []

            inputsize = inputs.size()
            K, T, D = inputsize[0], inputsize[1], inputsize[2]

            # print("inputs.size()")
            # print(inputs.size())

            if (horizon is not None):
                if (not (K == 1)) or (not (T == 1)):
                    raise ValueError(
                        "Forward iterative called with K!=1 or T!=1 and a horizon. This is not allowed! K={:}, T={:}, D={:}"
                        .format(K, T, D))
                else:
                    # Horizon is not None and T=1, so forecast called in the testing phase
                    pass
            else:
                horizon = T

            if iterative_forecasting_prob == 0:
                assert T == horizon, "If iterative forecasting, with iterative_forecasting_prob={:}>0, the provided time-steps T cannot be {:}, but have to be horizon={:}.".format(
                    iterative_forecasting_prob, T, horizon)

            # When T>1, only inputs[:,0,:] is taken into account. The network is propagating its own predictions.
            input_t = inputs[:, 0].view(K, 1, *inputs.size()[2:])
            # print("start")
            # print("horizon")
            # print(horizon)
            # print(T)
            assert (T > 0)
            assert (horizon > 0)
            time_latent_prop = 0.0
            for t in range(horizon):
                # print("t")
                # print(t)
                # print(input_is_latent)
                # print(input_t.size())
                # BE CAREFULL: input may be the latent input!
                output, next_hidden_state, latent_state, latent_state_pred, RNN_output, input_decoded, time_latent_prop_t, _, _ = self.forward_(
                    input_t,
                    init_hidden_state,
                    is_train=is_train,
                    is_latent=input_is_latent)
                # print(output.min())
                time_latent_prop += time_latent_prop_t

                # Settting the next input if t < horizon - 1
                if t < horizon - 1:

                    if iterative_forecasting_prob > 0.0:
                        # Iterative forecasting:
                        # with probability iterative_forecasting_prob propagate the state
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        temp = torch.rand(1).data[0].item()
                    else:
                        temp = 0.0

                    if temp < (1 - iterative_forecasting_prob):
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        input_t = inputs[:,
                                         t + 1].view(K, 1,
                                                     *inputs.size()[2:])
                        input_is_latent = False
                    else:
                        # with probability iterative_forecasting_prob propagate the state
                        if iterative_propagation_is_latent:
                            # Changing the propagation to latent
                            input_is_latent = True
                            # input_t = latent_state_pred

                            if iterative_forecasting_gradient:
                                # Forecasting the prediction as a tensor in graph
                                input_t = latent_state_pred
                            else:
                                # Deatching, and propagating the prediction as data
                                input_t = latent_state_pred.detach()
                        else:
                            if iterative_forecasting_gradient:
                                # Forecasting the prediction as a tensor in graph
                                input_t = output
                            else:
                                # Deatching, and propagating the prediction as data
                                # input_t = output.detach()
                                input_t = Variable(output.data)

                outputs.append(output[:, 0])
                inputs_decoded.append(input_decoded[:, 0])

                latent_states.append(latent_state[:, 0])
                latent_states_pred.append(latent_state_pred[:, 0])
                RNN_internal_states.append(next_hidden_state)
                RNN_outputs.append(RNN_output[:, 0])

                init_hidden_state = next_hidden_state

            outputs = torch.stack(outputs)
            outputs = outputs.transpose(1, 0)
            inputs_decoded = torch.stack(inputs_decoded)
            inputs_decoded = inputs_decoded.transpose(1, 0)

            latent_states = torch.stack(latent_states)
            latent_states_pred = torch.stack(latent_states_pred)
            RNN_outputs = torch.stack(RNN_outputs)

            latent_states = latent_states.transpose(1, 0)
            latent_states_pred = latent_states_pred.transpose(1, 0)
            RNN_outputs = RNN_outputs.transpose(1, 0)

        # Two additional dummy outputs for the case of Beta-VAE
        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop, None, None

    def transform2Tuple(self, hidden_state):
        hx, cx = hidden_state
        hidden_state = tuple([hx, cx])
        return hidden_state

    def forwardRNN(self, inputs, init_hidden_state, is_train):
        # The inputs are the latent_states
        T = inputs.size()[0]
        RNN_outputs = []
        for t in range(T):
            input_t = inputs[t]
            next_hidden_state = []
            for ln in range(len(self.RNN)):
                hidden_state = init_hidden_state[ln]

                if self.parent.params[
                        "RNN_cell_type"] == "lstm" and not self.parent.RNN_convolutional:
                    hidden_state = self.transform2Tuple(init_hidden_state[ln])

                # print(input_t)
                # print(hidden_state)
                RNN_output, next_hidden_state_layer = self.RNN[ln].forward(
                    input_t, hidden_state, is_train=is_train)

                if self.parent.params[
                        "RNN_cell_type"] == "lstm" and not self.parent.RNN_convolutional:
                    hx, cx = next_hidden_state_layer
                    next_hidden_state_layer = torch.stack([hx, cx])

                next_hidden_state.append(next_hidden_state_layer)
                input_t = RNN_output

            init_hidden_state = next_hidden_state
            RNN_outputs.append(RNN_output)

        RNN_outputs = torch.stack(RNN_outputs)
        next_hidden_state = torch.stack(next_hidden_state)

        return RNN_outputs, next_hidden_state

    def forwardEncoder(self, inputs):
        # print("# forwardEncoder() #")
        # # PROPAGATING THROUGH THE ENCODER TO GET THE LATENT STATE
        # print("inputs.size()")
        # print(inputs.size())

        # print(torch.amax(inputs))
        # print(torch.amin(inputs))
        # print(inputs)
        # print(ark)
        """ Perform dimensionality reduction """
        if self.parent.has_dimred:
            if torch.is_tensor(inputs):
                latent_pca = inputs.detach().cpu().numpy()
            latent_pca = self.parent.model_autoencoder.applyDimRed(latent_pca)
            latent_pca = utils.transform2Tensor(self.parent, latent_pca)

        # print(latent_pca)
        # print(ark)
        if self.parent.has_autoencoder:
            latent_autoencoder = inputs
            shape_ = latent_autoencoder.size()
            T, K = shape_[0], shape_[1]
            latent_autoencoder = torch.reshape(latent_autoencoder,
                                               (T * K, *shape_[2:]))
            for l in range(len(self.ENCODER)):
                latent_autoencoder = self.ENCODER[l](latent_autoencoder)
            shape_ = latent_autoencoder.size()

            latent_autoencoder = torch.reshape(latent_autoencoder,
                                               (T, K, *shape_[1:]))

        if self.parent.has_dimred and self.parent.has_autoencoder:
            """ Concatenate over latent state dimension """
            latent_output = torch.cat((latent_autoencoder, latent_pca), axis=2)
            return latent_output

        elif self.parent.has_dimred:
            return latent_pca

        elif self.parent.has_autoencoder:
            return latent_autoencoder

        else:
            raise ValueError("Propagation through undefined encoder.")

    def forwardDecoder(self, inputs):
        """ Divide axis """

        # print("# forwardDecoder() #")
        # Dimension of inputs: [T, K, latend_dim + noise_dim]

        shape_ = inputs.size()
        T, K, LD = shape_[0], shape_[1], shape_[2]

        if self.parent.has_autoencoder and self.parent.has_dimred:
            LD = int(LD / 2)
            latent_autoencoder = inputs[:, :, :LD]
            latent_pca = inputs[:, :, LD:]

        elif (not self.parent.has_autoencoder) and self.parent.has_dimred:
            latent_pca = inputs
            if torch.is_tensor(latent_pca):
                latent_pca = latent_pca.detach().cpu().numpy()
        elif self.parent.has_autoencoder and (not self.parent.has_dimred):
            latent_autoencoder = inputs
        else:
            raise ValueError("Propagation through undefined decoder.")

        if self.parent.has_dimred:
            if torch.is_tensor(latent_pca):
                latent_pca = latent_pca.detach().cpu().numpy()
            outputs_pca = self.parent.model_autoencoder.applyInverseDimRed(
                latent_pca)
            outputs_pca = np.reshape(outputs_pca,
                                     (T, K, *self.parent.getOutputShape()))
            outputs_pca = utils.transform2Tensor(self.parent, outputs_pca)

        if self.parent.has_autoencoder:
            outputs_autoencoder = latent_autoencoder
            shape_ = outputs_autoencoder.size()
            outputs_autoencoder = torch.reshape(outputs_autoencoder,
                                                (T * K, *shape_[2:]))
            for l in range(len(self.DECODER)):
                outputs_autoencoder = self.DECODER[l](outputs_autoencoder)

            shape_ = outputs_autoencoder.size()
            outputs_autoencoder = torch.reshape(outputs_autoencoder,
                                                (T, K, *shape_[1:]))

        if self.parent.has_autoencoder and self.parent.has_dimred:
            # residual_output_scaling = self.residual_output_scaling
            # if self.parent.Dx > 0: residual_output_scaling = residual_output_scaling.unsqueeze(1)
            # if self.parent.Dy > 0: residual_output_scaling = residual_output_scaling.unsqueeze(1)
            # """ Adding the K dimension """
            # residual_output_scaling = residual_output_scaling.unsqueeze(0)
            # """ Adding the T dimension """
            # residual_output_scaling = residual_output_scaling.unsqueeze(0)

            # print("-"*10)
            # print("residual_output_scaling max, min, mean, std")
            # print(torch.amax(residual_output_scaling))
            # print(torch.amin(residual_output_scaling))
            # print(torch.mean(residual_output_scaling))
            # print(torch.std(residual_output_scaling))

            # print("outputs max, min, mean, std")
            # print(torch.amax(outputs))
            # print(torch.amin(outputs))
            # print(torch.mean(outputs))
            # print(torch.std(outputs))

            # print("pca_outputs max, min, mean, std")
            # print(torch.amax(pca_outputs))
            # print(torch.amin(pca_outputs))
            # print(torch.mean(pca_outputs))
            # print(torch.std(pca_outputs))
            outputs = pca_outputs + outputs  # * residual_output_scaling
            # outputs = pca_outputs + 0.000000001 * outputs

        elif (not self.parent.has_autoencoder) and self.parent.has_dimred:
            return outputs_pca

        elif self.parent.has_autoencoder and (not self.parent.has_dimred):
            return outputs_autoencoder

        else:
            raise ValueError("Propagation through undefined decoder.")

    def forwardRNNOutput(self, inputs):
        # print("forwardRNNOutput()")
        # print("inputs.size()")
        # print(inputs.size())
        outputs = []
        for input_t in inputs:
            output = input_t
            for l in range(len(self.RNN_OUTPUT)):
                output = self.RNN_OUTPUT[l](output)
            outputs.append(output)
        outputs = torch.stack(outputs)
        # print("outputs.size()")
        # print(outputs.size())
        return outputs

    def setBatchNormalizationLayersToEvalMode(self):
        # print("# setBatchNormalizationLayersToEvalMode() #")
        for layer in self.ENCODER:
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                # print("# encoder : found batch norm layer ! #")
                layer.eval()
                # print(layer)
                # print(layer.running_mean)
                # print(layer.running_var)
            elif isinstance(layer, nn.modules.Dropout):
                layer.eval()

        for layer in self.DECODER:
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                # print("# decoder : found batch norm layer ! #")
                layer.eval()
                # print(layer)
                # print(layer.weight)
                # print(layer.bias)
                # print(layer.running_mean)
                # print(layer.running_var)
                # pass
            elif isinstance(layer, nn.modules.Dropout):
                layer.eval()

        # print(ark)
        return 0

    def scaleLatentState(self, latent_state):
        K, T, D = latent_state.size()

        # print(latent_state)
        # print(self.latent_state_max)
        # print(ark)
        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            latent_state_min = self.repeatTensor(self.latent_state_min, K, T)
            latent_state_max = self.repeatTensor(self.latent_state_max, K, T)
            if self.parent.gpu:
                latent_state_min = latent_state_min.cuda()
                latent_state_max = latent_state_max.cuda()
            assert (latent_state.size() == latent_state_min.size())
            assert (latent_state.size() == latent_state_max.size())
            return (latent_state - latent_state_min) / (latent_state_max -
                                                        latent_state_min)
        elif self.parent.params["latent_space_scaler"] == "Standard":
            latent_state_mean = self.repeatTensor(self.latent_state_mean, K, T)
            latent_state_std = self.repeatTensor(self.latent_state_std, K, T)
            if self.parent.gpu:
                latent_state_mean = latent_state_mean.cuda()
                latent_state_std = latent_state_std.cuda()
            assert (latent_state.size() == latent_state_mean.size())
            assert (latent_state.size() == latent_state_std.size())
            return (latent_state - latent_state_mean) / latent_state_std
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(
                self.parent.params["latent_space_scaler"]))

    def descaleLatentState(self, latent_state):
        K, T, D = latent_state.size()
        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            assert self.parent.RNN_activation_str_output == "tanhplus", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(
                self.parent.params["latent_space_scaler"],
                self.parent.RNN_activation_str_output)
            latent_state_min = self.repeatTensor(self.latent_state_min, K, T)
            latent_state_max = self.repeatTensor(self.latent_state_max, K, T)
            if self.parent.gpu:
                latent_state_min = latent_state_min.cuda()
                latent_state_max = latent_state_max.cuda()

            assert (latent_state.size() == latent_state_min.size())
            assert (latent_state.size() == latent_state_max.size())
            return latent_state * (latent_state_max -
                                   latent_state_min) + latent_state_min
        elif self.parent.params["latent_space_scaler"] == "Standard":
            assert self.parent.RNN_activation_str_output == "identity", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(
                self.parent.params["latent_space_scaler"],
                self.parent.RNN_activation_str_output)
            latent_state_mean = self.repeatTensor(self.latent_state_mean, K, T)
            latent_state_std = self.repeatTensor(self.latent_state_std, K, T)
            if self.parent.gpu:
                latent_state_mean = latent_state_mean.cuda()
                latent_state_std = latent_state_std.cuda()
            assert (latent_state.size() == latent_state_mean.size())
            assert (latent_state.size() == latent_state_std.size())
            return latent_state * latent_state_std + latent_state_mean
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(
                self.parent.params["latent_space_scaler"]))

    def repeatTensor(self, temp, K, T):
        tensor = temp.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.repeat(K, T, 1)
        return tensor

    def setLatentStateBounds(self,
                             min_=None,
                             max_=None,
                             mean_=None,
                             std_=None,
                             slack=0.1):
        latent_state_min = self.parent.torch_dtype(min_)
        latent_state_max = self.parent.torch_dtype(max_)

        latent_state_mean = self.parent.torch_dtype(mean_)
        latent_state_std = self.parent.torch_dtype(std_)

        range_ = latent_state_max - latent_state_min
        latent_state_min = latent_state_min - slack * range_
        latent_state_max = latent_state_max + slack * range_
        del self.latent_state_min
        del self.latent_state_max
        del self.latent_state_mean
        del self.latent_state_std

        self.latent_state_min = torch.nn.Parameter(latent_state_min)
        self.latent_state_max = torch.nn.Parameter(latent_state_max)
        self.latent_state_min.requires_grad = False
        self.latent_state_max.requires_grad = False
        self.latent_state_mean = torch.nn.Parameter(latent_state_mean)
        self.latent_state_std = torch.nn.Parameter(latent_state_std)
        self.latent_state_mean.requires_grad = False
        self.latent_state_std.requires_grad = False
        return 0

    def setOutputResidualScalerParams(self, data, slack=0.1):
        data = self.parent.torch_dtype(data)
        """ Positive scaling """
        data = (1. + slack) * data
        del self.residual_output_scaling
        self.residual_output_scaling = torch.nn.Parameter(data)
        self.residual_output_scaling.requires_grad = False
        return 0

    def defineOutputResidualScalerParams(self):
        self.residual_output_scaling = torch.nn.Parameter(
            torch.zeros(self.parent.input_dim))
        return 0

    def defineLatentStateParams(self):
        print("[dimred_rnn_model] Network has {:} latent state scaler.".format(
            self.parent.params["latent_space_scaler"]))
        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            assert self.parent.RNN_activation_str_output == "tanhplus", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(
                self.parent.params["latent_space_scaler"],
                self.parent.RNN_activation_str_output)
        elif self.parent.params["latent_space_scaler"] == "Standard":
            assert self.parent.RNN_activation_str_output == "identity", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(
                self.parent.params["latent_space_scaler"],
                self.parent.RNN_activation_str_output)
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(
                self.parent.params["latent_space_scaler"]))

        self.latent_state_min = torch.nn.Parameter(
            torch.zeros(self.parent.latent_state_dim_cum))
        self.latent_state_max = torch.nn.Parameter(
            torch.zeros(self.parent.latent_state_dim_cum))
        self.latent_state_mean = torch.nn.Parameter(
            torch.zeros(self.parent.latent_state_dim_cum))
        self.latent_state_std = torch.nn.Parameter(
            torch.zeros(self.parent.latent_state_dim_cum))
        self.latent_state_min.requires_grad = False
        self.latent_state_max.requires_grad = False
        self.latent_state_mean.requires_grad = False
        self.latent_state_std.requires_grad = False
        return 0

    def encoder(self, inputs):
        print(np.shape(inputs))
        print(ark)
        # print("# forwardEncoder() #")
        # # PROPAGATING THROUGH THE ENCODER TO GET THE LATENT STATE
        # print("inputs.size()")
        # print(inputs.size())

        # print(torch.amax(inputs))
        # print(torch.amin(inputs))
        # print(inputs)
        # print(ark)
        """ Perform dimensionality reduction """
        if self.parent.has_dimred:
            if torch.is_tensor(inputs):
                latent_pca = inputs.detach().cpu().numpy()
            latent_pca = self.parent.model_autoencoder.applyDimRed(latent_pca)
            latent_pca = utils.transform2Tensor(self.parent, latent_pca)

        # print(latent_pca)
        # print(ark)
        if self.parent.has_autoencoder:
            latent_autoencoder = inputs
            shape_ = latent_autoencoder.size()
            T, K = shape_[0], shape_[1]
            latent_autoencoder = torch.reshape(latent_autoencoder,
                                               (T * K, *shape_[2:]))
            for l in range(len(self.ENCODER)):
                latent_autoencoder = self.ENCODER[l](latent_autoencoder)
            shape_ = latent_autoencoder.size()

            latent_autoencoder = torch.reshape(latent_autoencoder,
                                               (T, K, *shape_[1:]))

        if self.parent.has_dimred and self.parent.has_autoencoder:
            """ Concatenate over latent state dimension """
            latent_output = torch.cat((latent_autoencoder, latent_pca), axis=2)
            return latent_output

        elif self.parent.has_dimred:
            return latent_pca

        elif self.parent.has_autoencoder:
            return latent_autoencoder

        else:
            raise ValueError("Propagation through undefined encoder.")
