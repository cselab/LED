#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
""" Torch """
import torch
import sys
from torch.autograd import Variable
""" Libraries """
import numpy as np
import os
import random
import time
from tqdm import tqdm
from pathlib import Path
""" Utilities """
from .. import Utils as utils
from .. import Systems as systems
""" Networks """
from . import dimred_rnn_model
""" Common libraries """
from . import common_testing
from . import common_plot
""" Printing """
from functools import partial

print = partial(print, flush=True)
import warnings
""" Horovod """
try:
    import horovod.torch as hvd
    print("[dimred_rnn] Imported Horovod.")
except ImportError:
    hvd = None


class dimred_rnn():
    def __init__(self, params):
        super(dimred_rnn, self).__init__()
        # Starting the timer
        self.start_time = time.time()
        # The parameters used to define the model
        self.params = params.copy()

        # The system to which the model is applied on
        self.system_name = params["system_name"]
        # Checking the system name
        assert (systems.checkSystemName(self))
        # The save format
        self.save_format = params["save_format"]

        self.reference_train_time = utils.getReferenceTrainingTime(
            params["reference_train_time"], params["buffer_train_time"])

        # Checking whether the GPU is available and setting the default tensor datatype
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.torch_dtype = torch.cuda.DoubleTensor
            if self.params["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True

        else:
            self.torch_dtype = torch.DoubleTensor

        ##################################################################
        # RANDOM SEEDING
        ##################################################################
        self.random_seed = params["random_seed"]

        # Setting the random seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu: torch.cuda.manual_seed(self.random_seed)

        # Optimizer to use
        self.optimizer_str = params["optimizer_str"]

        ##################################################################
        # SETTING THE PATHS
        ##################################################################
        # The path of the training data
        self.data_path_train = params['data_path_train']
        # The path of the training data
        self.data_path_val = params['data_path_val']
        # General data path (scaler data/etc.)
        self.data_path_gen = params['data_path_gen']
        # The path of the test data
        self.data_path_test = params['data_path_test']
        # The path to save all the results
        self.saving_path = params['saving_path']
        # The directory to save the model (inside saving_path)
        self.model_dir = params['model_dir']
        # The directory to save the figures (inside saving_path)
        self.fig_dir = params['fig_dir']
        # The directory to save the data results (inside saving_path)
        self.results_dir = params['results_dir']
        # The directory to save the logfiles (inside saving_path)
        self.logfile_dir = params["logfile_dir"]
        # Whether to write a log-file or not
        self.write_to_log = params["write_to_log"]

        # Whether to display in the output (verbocity)
        self.display_output = params["display_output"]

        # The number of IC to test on
        self.num_test_ICS = params["num_test_ICS"]

        # The prediction horizon
        self.prediction_horizon = params["prediction_horizon"]

        # The activation string of the RNN
        self.RNN_activation_str = params['RNN_activation_str']

        # The activation string at the output of the RNN
        self.RNN_activation_str_output = params['RNN_activation_str_output']

        # The cell type of the RNN
        self.RNN_cell_type = params['RNN_cell_type']

        self.input_dim = params['input_dim']

        self.channels = params['channels']
        self.Dz, self.Dy, self.Dx = utils.getChannels(self.channels, params)

        # Zoneout probability for regularizing the RNN
        self.zoneout_keep_prob = params["zoneout_keep_prob"]
        # The sequence length
        self.sequence_length = params['sequence_length']
        # The prediction length of BPTT
        self.prediction_length = params['prediction_length']
        if self.prediction_length == 0:
            self.prediction_length = self.sequence_length
        # The number of warming-up steps during training
        self.n_warmup_train = params["n_warmup_train"]

        # The noise level in the training data
        self.noise_level = params['noise_level']

        ##################################################################
        # SCALER
        ##################################################################
        self.scaler = params["scaler"]

        ##################################################################
        # TRAINING PARAMETERS
        ##################################################################
        # Whether to retrain or not
        self.retrain = params['retrain']

        self.batch_size = params['batch_size']
        self.overfitting_patience = params['overfitting_patience']
        self.max_epochs = params['max_epochs']
        self.max_rounds = params['max_rounds']
        self.learning_rate = params['learning_rate']

        self.output_forecasting_loss = params["output_forecasting_loss"]
        self.latent_forecasting_loss = params["latent_forecasting_loss"]

        self.latent_state_dim = params["latent_state_dim"]

        self.layers_rnn = [self.params["RNN_layers_size"]
                           ] * self.params["RNN_layers_num"]
        if len(self.layers_rnn) > 0:
            # Parsing the RNN layers
            self.has_rnn = 1
            # VALID COMPINATIONS:
            if self.RNN_activation_str not in [
                    'tanh'
            ] and self.RNN_cell_type not in ['mlp']:
                raise ValueError('Error: Invalid RNN activation.')
            if self.RNN_cell_type not in ['lstm', 'gru', 'mlp']:
                raise ValueError('Error: Invalid RNN cell type.')
        else:
            self.has_rnn = 0

        assert self.has_rnn
        assert self.latent_forecasting_loss
        assert not self.output_forecasting_loss

        if self.has_rnn:
            assert self.output_forecasting_loss or self.latent_forecasting_loss

        self.train_residual_AE = params["train_residual_AE"]
        self.dimred_method = params["dimred_method"]

        ##################################################################
        # Encoding model (either CNN/PCA/etc.)
        ##################################################################

        if self.params["dimred_method"] == "ae":
            raise ValueError(
                "[dimred_rnn] Attempted to use CNN autoencoder for dimensionality reduction. In this case use the class crnn. This class only supports the RNN in the latent space with PCA/DiffMaps for autoencoding."
            )

        elif self.params["dimred_method"] == "pca":
            from . import dimred
            self.model_autoencoder = dimred.dimred(self.params)
            self.model_autoencoder.load()
            self.has_autoencoder = 0
            self.has_dimred = 1

        elif self.params["dimred_method"] == "diffmaps":
            from . import dimred
            self.model_autoencoder = dimred.dimred(self.params)
            self.model_autoencoder.load()
            self.has_autoencoder = 0
            self.has_dimred = 1

        else:
            raise ValueError("Invalid autoencoding method.")

        if self.train_residual_AE:
            # Whether the autoencoder is convolutional or not
            self.AE_convolutional = params["AE_convolutional"]
            assert self.AE_convolutional == 1
            assert self.has_rnn == 0
            self.AE_batch_norm = params["AE_batch_norm"]
            self.AE_conv_transpose = params["AE_conv_transpose"]
            self.AE_pool_type = params["AE_pool_type"]
            self.has_dummy_autoencoder = False
            # self.reconstruction_loss = params["reconstruction_loss"]
            self.beta_vae = False
            self.layers_encoder_aug = []
            self.layers_decoder_aug = []
            self.training_loss = "mse"
            self.reconstruction_loss = 1
            self.c1_latent_smoothness_loss = False
            self.params["precision"] = "double"
            self.params["dropout_keep_prob"] = 1.0
            self.load_trained_AE = False

            self.latent_state_dim_pca = self.latent_state_dim
            self.latent_state_dim_auto = self.latent_state_dim
            self.latent_state_dim_cum = 2 * self.latent_state_dim

            self.params["RNN_state_dim"] = self.latent_state_dim_cum

            self.has_dimred = 1

            assert self.params[
                "activation_str_output"] == "tanh", "The autoencoder is learning the residual, scaled to [-1,1]. The requested activation_str_output (={:}) has to be tanh.".format(
                    self.params["activation_str_output"])

        elif self.has_autoencoder:
            # Whether the autoencoder is convolutional or not
            self.AE_convolutional = params["AE_convolutional"]
            assert self.AE_convolutional == 1
            assert self.has_rnn == 1
            self.AE_batch_norm = params["AE_batch_norm"]
            self.AE_conv_transpose = params["AE_conv_transpose"]
            self.AE_pool_type = params["AE_pool_type"]
            self.has_dummy_autoencoder = False
            self.beta_vae = False
            self.layers_encoder_aug = []
            self.layers_decoder_aug = []
            self.training_loss = "mse"
            self.reconstruction_loss = 0
            self.c1_latent_smoothness_loss = False
            self.params["precision"] = "double"
            self.params["dropout_keep_prob"] = 1.0
            self.load_trained_AE = True

            self.latent_state_dim_pca = self.latent_state_dim
            self.latent_state_dim_auto = self.latent_state_dim
            self.latent_state_dim_cum = 2 * self.latent_state_dim

            self.params["RNN_state_dim"] = self.latent_state_dim_cum

            self.has_dimred = 1

            assert self.params[
                "activation_str_output"] == "tanh", "The autoencoder is learning the residual, scaled to [-1,1]. The requested activation_str_output (={:}) has to be tanh.".format(
                    self.params["activation_str_output"])
        else:
            self.has_dummy_autoencoder = False
            self.AE_convolutional = False
            self.beta_vae = False
            self.has_autoencoder = False
            self.layers_encoder_aug = []
            self.layers_decoder_aug = []
            self.training_loss = "mse"
            self.reconstruction_loss = 0
            self.c1_latent_smoothness_loss = False
            self.params["precision"] = "double"
            self.params["dropout_keep_prob"] = 1.0
            self.params["activation_str_general"] = "celu"
            self.load_trained_AE = False

            self.latent_state_dim_pca = self.latent_state_dim
            self.latent_state_dim_auto = 0
            self.latent_state_dim_cum = self.latent_state_dim

            self.params["RNN_state_dim"] = self.latent_state_dim_cum
            self.RNN_convolutional = 0
            self.RNN_trainable_init_hidden_state = 0
            self.multinode = 0
            self.rank_str = ""

            self.has_dimred = 1
            self.train_RNN = 1

        self.model = dimred_rnn_model.dimred_rnn_model(self.params, self)

        if self.has_rnn:
            """ Define the latent scaler """
            self.model.defineLatentStateParams()
            self.model.has_latent_scaler = True
            self.model.printModuleList()

        elif self.has_autoencoder:
            self.model.printModuleList()

        # if (not self.train_residual_AE) and (not self.train_RNN):
        #     """ Just the dimensionality reduction """
        #     self.model_name = self.createModelName()

        # el

        if self.train_residual_AE:
            self.model_name = self.createModelName(
                with_residual_autoencoder=True)

        else:
            self.model_name = self.createModelName(
                with_residual_autoencoder=False)

        print("[dimred_rnn] - model_name:")
        print("[dimred_rnn] {:}".format(self.model_name))
        self.saving_model_path = utils.getModelDir(self) + "/model"

        utils.makeDirectories(self)

        self.model_parameters, self.model_named_params = self.model.getParams()
        """ Initialize model parameters """
        self.model.initializeWeights()

        self.device_count = torch.cuda.device_count()

        self.local_rank = 0
        if self.gpu:
            print("[dimred_rnn] USING CUDA -> SENDING THE MODEL TO THE GPU.")
            torch.cuda.set_device(self.local_rank)
            self.model.sendModelToCuda()
            if self.device_count > 1:
                raise ValueError(
                    "More than one GPU devide detected. Aborting.")
        """ Saving some info file for the model """
        data = {"params": params, "name": self.model_name}
        data_path = utils.getModelDir(self) + "/info"
        utils.saveData(data, data_path, "pickle")
        self.data_info_dict = systems.getSystemDataInfo(self)

        # Dummy parameters:
        self.iterative_loss_validation = 0
        self.iterative_loss_schedule_and_gradient = "none"
        self.RNN_statefull = 1

    def createModelName(self, with_residual_autoencoder=False):
        # print("# createModelName() #")
        keys = self.getKeysInModelName()
        str_gpu = "GPU-" * self.gpu
        str_ = str_gpu + "RNN-" + self.getAutoencoderName()
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        if with_residual_autoencoder:
            raise ValueError("TODO:")
        return str_

    def getAutoencoderName(self):
        return self.model_autoencoder.model_name

    def getKeysInModelName(self):
        keys = {
            'rc_solver': '-SOLVER_',
            'rc_approx_reservoir_size': '-SIZE_',
            'rc_degree': '-DEG_',
            'rc_radius': '-R_',
            'rc_sigma_input': '-S_',
            # 'rc_dynamics_length': '-DYN_',
            'rc_regularization': '-REG_',
            'rc_noise_level_per_mill': '-NS_',
        }
        return keys

    def getKeysInModelName(self, with_autoencoder=False, with_rnn=False):
        keys = {
            'optimizer_str': '-OPT_',
        }
        keys.update({
            'RNN_cell_type': '-C_',
            # 'RNN_activation_str':'-ACT_',
            'RNN_layers_num': '-R_',
            'RNN_layers_size': 'x',
            # 'zoneout_keep_prob':'-ZKP_',
            'sequence_length': '-SL_',
            # 'prediction_horizon':'-PH_',
            # 'num_test_ICS':'-NICS_',
            # 'output_forecasting_loss': '-LFO_',
            # 'latent_forecasting_loss': '-LFL_',
        })
        if not (self.params["precision"] == "double"):
            keys.update({'precision': '-PREC_'})

        if self.params["random_seed_in_name"]:
            keys.update({'random_seed': '-RS_'})

        if self.params["learning_rate_in_name"]:
            keys.update({'learning_rate': '-LR_'})

        return keys

    def printParams(self):
        self.n_trainable_parameters = self.model.countTrainableParams()
        self.n_model_parameters = self.model.countParams()
        # Print parameter information:
        print("[dimred_rnn] Trainable params {:}/{:}".format(
            self.n_trainable_parameters, self.n_model_parameters))
        return 0

    def declareOptimizer(self, lr):
        if self.train_residual_AE:
            # Get the autoencoder params
            self.params_trainable, self.params_named_trainable = self.model.getAutoencoderParams(
            )

            # If it has RNN disable the gradient of the RNN params
            if self.has_rnn:
                rnn_params, rnn_named_params = self.model.getRNNParams()
                for name, param in rnn_named_params:
                    param.requires_grad = False

        elif self.train_RNN:

            self.params_trainable, self.params_named_trainable = self.model.getRNNParams(
            )

            # Disable the gradient of the Autoencoder params
            AE_params, AE_named_params = self.model.getAutoencoderParams()
            for name, param in AE_named_params:
                param.requires_grad = False
        else:
            self.params_trainable = self.model_parameters
            self.params_named_trainable = self.model_named_params

        # Weight decay only when training the autoencoder
        if self.has_rnn and not self.train_residual_AE:
            """ No weight decay in RNN training """
            weight_decay = 0.0

        else:
            weight_decay = self.weight_decay_AE

        print("[dimred_rnn] Learning rate: {:}, Weight decay: {:}".format(
            lr, weight_decay))
        self.printParams()

        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(
                self.params_trainable,
                lr=lr,
                weight_decay=weight_decay,
            )

        elif self.optimizer_str == "sgd":
            momentum = 0.9
            self.optimizer = torch.optim.SGD(
                self.params_trainable,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        elif self.optimizer_str == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.params_trainable,
                lr=lr,
                weight_decay=weight_decay,
            )

        elif self.optimizer_str == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(
                self.params_trainable,
                lr=lr,
                eps=1e-16,
                betas=(0.9, 0.999),
                weight_decouple=True,
                rectify=False,
            )

        else:
            raise ValueError("Optimizer {:} not recognized.".format(
                self.optimizer_str))

    def getInitialRNNHiddenState(self, batch_size):
        if self.has_rnn and self.RNN_trainable_init_hidden_state:
            hidden_state = self.model.getRnnHiddenState(batch_size)
        else:
            hidden_state = self.getZeroRnnHiddenState(batch_size)
        if (self.torch_dtype
                == torch.DoubleTensor) or (self.torch_dtype
                                           == torch.cuda.DoubleTensor):
            if torch.is_tensor(hidden_state):
                hidden_state = hidden_state.double()
        return hidden_state

    def getZeroRnnHiddenState(self, batch_size):
        if self.has_rnn:
            hidden_state = []
            for ln in self.layers_rnn:
                hidden_state.append(
                    self.getZeroRnnHiddenStateLayer(batch_size, ln))
            hidden_state = torch.stack(hidden_state)
            hidden_state = self.getModel().transposeHiddenState(hidden_state)
        else:
            hidden_state = []
        return hidden_state

    def getZeroState(self, batch_size, hidden_units):
        if self.channels == 2:
            return torch.zeros(batch_size, hidden_units, self.Dy, self.Dx)
        elif self.channels == 1:
            return torch.zeros(batch_size, hidden_units, self.Dx)

    def getOutputShape(self):
        if self.channels == 2:
            return [self.input_dim, self.Dy, self.Dx]
        elif self.channels == 1:
            return [self.input_dim, self.Dx]

    def getZeroRnnHiddenStateLayer(self, batch_size, hidden_units):
        if self.RNN_cell_type == "mlp": return torch.zeros(1)
        if self.RNN_convolutional:
            hx = Variable(self.getZeroState(batch_size, hidden_units))
            if "lstm" in self.params["RNN_cell_type"]:
                cx = Variable(self.getZeroState(batch_size, hidden_units))
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(
                    self.params["RNN_cell_type"]))
        else:
            hx = Variable(torch.zeros(batch_size, hidden_units))
            if "lstm" in self.params["RNN_cell_type"]:
                cx = Variable(torch.zeros(batch_size, hidden_units))
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(
                    self.params["RNN_cell_type"]))

    def plotBatchNumber(self, i, n_batches, is_train):
        if self.display_output:
            str_ = "\n" + is_train * "TRAINING: " + (
                not is_train) * "EVALUATION"
            print("{:s} batch {:d}/{:d},  {:f}%".format(
                str_, int(i + 1), int(n_batches), (i + 1) / n_batches * 100.))
            sys.stdout.write("\033[F")

    def sendHiddenStateToGPU(self, h_state):
        if self.has_rnn:
            return h_state.cuda()
        else:
            return h_state

    def detachHiddenState(self, h_state):
        if self.has_rnn:
            return h_state.detach()
        else:
            return h_state

    def getLoss(
        self,
        output,
        target,
        is_latent=False,
    ):
        # Mean squared loss
        loss = output - target
        loss = loss.pow(2.0)
        # Mean over all dimensions
        loss = loss.mean(2)
        # Mean over all batches
        loss = loss.mean(0)
        # Mean over all time-steps
        loss = loss.mean()
        return loss

    def repeatAlongDim(self, var, axis, repeat_times, interleave=False):
        if not interleave:
            repeat_idx = len(var.size()) * [1]
            repeat_idx[axis] = repeat_times
            var = var.repeat(*repeat_idx)
        else:
            var = var.repeat_interleave(repeat_times, dim=axis)
        return var

    # def getDataBatch(self, batch_of_sequences, start, stop, dataset=None):
    #     if self.data_info_dict["structured"]:
    #         data = dataset.getSequencesPart(batch_of_sequences, start, stop)
    #     else:
    #         data = batch_of_sequences[:, start:stop]
    #     return data

    # def applyDimRed(self, data):
    #     """ Use the dimensionality reduction method to project the data """
    #     assert len(np.shape(data))==2+1+self.channels, "[applyDimRed()] Error, len(np.shape(data))={:}, while 2+1+self.channels={:}".format(len(np.shape(data)), 2+1+self.channels)
    #     shape_ = np.shape(data)
    #     data = np.reshape(data, (shape_[0]*shape_[1], -1))
    #     data = self.dimred_model.transform(data)
    #     data = np.reshape(data, (shape_[0], shape_[1], -1))
    #     return data

    # def applyInverseDimRed(self, data):
    #     assert len(np.shape(data))==3
    #     """ Use the dimensionality reduction method to lift the projected data """
    #     shape_ = np.shape(data)
    #     data = np.reshape(data, (shape_[0]*shape_[1], -1))
    #     data = self.dimred_model.inverse_transform(data)
    #     data = np.reshape(data, (shape_[0], shape_[1], -1))
    #     return data

    def trainOnBatch(self, batch_of_sequences, is_train=False, dataset=None):
        # print("# trainOnBatch() #")
        batch_size = len(batch_of_sequences)
        initial_hidden_states = self.getInitialRNNHiddenState(batch_size)

        if self.data_info_dict["structured"]:
            T = dataset.seq_paths[0]["num_timesteps"]
        else:
            T = np.shape(batch_of_sequences)[1]

        # print(dataset.seq_paths[0]["num_timesteps"])
        # print(batch_of_sequences)

        losses_vec = []

        # assert ( T - 1 - self.n_warmup_train) % self.sequence_length == 0, "The time-steps in the sequence (minus the warm-up time-steps) need to be divisible by the sequence_length. T={:}, n_warmup_train={:}, sequence_length={:}, ((T-1-n_warmup_train) % self.sequence_length) == 0, -> {:} % {:} ==0".format(T, self.n_warmup_train, self.sequence_length, T - 1- self.n_warmup_train, self.sequence_length)
        # assert ( T - 1 - self.n_warmup_train - 1) % self.sequence_length == 0, "The time-steps in the sequence (minus the warm-up time-steps) need to be divisible by the sequence_length. T={:}, n_warmup_train={:}, sequence_length={:}, ((T-1-n_warmup_train- 1) % self.sequence_length) == 0, -> {:} % {:} ==0".format(T, self.n_warmup_train, self.sequence_length - 1, T - 1- self.n_warmup_train, self.sequence_length)

        predict_on = self.n_warmup_train
        if self.n_warmup_train > 0:
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()

            input_batch = utils.getDataBatch(self,
                                             batch_of_sequences,
                                             predict_on - self.n_warmup_train,
                                             predict_on,
                                             dataset=dataset)
            target_batch = utils.getDataBatch(self,
                                              batch_of_sequences,
                                              predict_on -
                                              self.n_warmup_train + 1,
                                              predict_on + 1,
                                              dataset=dataset)

            if torch.is_tensor(input_batch):
                input_batch = input_batch.detach().cpu().numpy()
            if torch.is_tensor(target_batch):
                target_batch = target_batch.detach().cpu().numpy()

            input_batch = utils.transform2Tensor(self, input_batch)
            target_batch = utils.transform2Tensor(self, target_batch)
            initial_hidden_states = utils.transform2Tensor(
                self, initial_hidden_states)

            # Adding noise to the input data for regularization
            if self.noise_level > 0.0:
                input_batch += self.noise_level * torch.randn_like(input_batch)

            output_batch, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_batch_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar = self.model.forward(
                input_batch,
                initial_hidden_states,
                is_train=False,
                is_iterative_forecasting=False,
                iterative_forecasting_prob=0,
                iterative_forecasting_gradient=0,
                iterative_propagation_is_latent=False,
                horizon=None,
                input_is_latent=False,
            )

            last_hidden_state = self.detachHiddenState(last_hidden_state)
            initial_hidden_states = last_hidden_state

        num_propagations = int(
            (T - 1 - self.n_warmup_train) / self.sequence_length)
        # print("num_propagations")
        # print(num_propagations)
        assert num_propagations >= 1, "Number of propagations int((T - 1 - self.n_warmup_train) / self.sequence_length) = {:} has to be larger than or equal to one. T={:}, self.n_warmup_train={:}, self.sequence_length={:}".format(
            num_propagations, T, self.n_warmup_train, self.sequence_length)
        # print(ark)

        predict_on += self.sequence_length
        for p in range(num_propagations):
            # print("Propagation {:}/{:}".format(p, num_propagations))
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()
            """ Getting the batch """
            input_batch = utils.getDataBatch(
                self,
                batch_of_sequences,
                predict_on - self.sequence_length,
                predict_on,
                dataset=dataset,
            )
            target_batch = utils.getDataBatch(
                self,
                batch_of_sequences,
                predict_on - self.sequence_length + 1,
                predict_on + 1,
                dataset=dataset,
            )

            input_batch = utils.transform2Tensor(self, input_batch)
            target_batch = utils.transform2Tensor(self, target_batch)
            initial_hidden_states = utils.transform2Tensor(
                self, initial_hidden_states)

            # Adding noise to the input data for regularization
            if self.noise_level > 0.0:
                input_batch += self.noise_level * torch.randn_like(input_batch)

            if not is_train and self.iterative_loss_validation:
                # set iterative forecasting to True in case of validation
                iterative_forecasting_prob = 1.0
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = self.iterative_propagation_during_training_is_latent
                is_iterative_forecasting = True
                # ----------------------------------------
                iterative_forecasting_gradient = False

            elif self.iterative_loss_schedule_and_gradient in ["none"]:
                iterative_forecasting_prob = 0.0
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = False
                is_iterative_forecasting = False
                # ----------------------------------------
                iterative_forecasting_gradient = 0

            elif any(x in self.iterative_loss_schedule_and_gradient
                     for x in ["linear", "inverse_sigmoidal", "exponential"]):
                assert (self.iterative_loss_validation == 1)
                iterative_forecasting_prob = self.getIterativeForecastingProb(
                    self.epochs_iter_global,
                    self.iterative_loss_schedule_and_gradient)
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = self.iterative_propagation_during_training_is_latent
                is_iterative_forecasting = True
                # ----------------------------------------
                iterative_forecasting_gradient = self.iterative_loss_gradient

            else:
                raise ValueError(
                    "self.iterative_loss_schedule_and_gradient={:} not recognized."
                    .format(self.iterative_loss_schedule_and_gradient))
            self.iterative_forecasting_prob = iterative_forecasting_prob

            if self.has_rnn and (
                    self.has_autoencoder or self.has_dimred
            ) and self.latent_forecasting_loss and not self.output_forecasting_loss:
                detach_output = True
                del target_batch
            else:
                detach_output = False

            output_batch, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_batch_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar = self.model.forward(
                input_batch,
                initial_hidden_states,
                is_train=is_train,
                is_iterative_forecasting=is_iterative_forecasting,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                horizon=None,
                input_is_latent=False,
                detach_output=detach_output,
            )
            if detach_output: del input_batch

            # print(input_batch.size())
            # print(target_batch.size())
            # import matplotlib
            # import matplotlib.pyplot as plt
            # plt.plot(input_batch[0, :, 0, 8, 8])
            # plt.plot(target_batch[0, :, 0, 8, 8])
            # plt.plot(output_batch[0, :, 0, 8, 8])
            # plt.show()
            # print(ark)

            if self.output_forecasting_loss:
                output_batch = output_batch[:,
                                            -self.params["prediction_length"]:]
                target_batch = target_batch[:,
                                            -self.params["prediction_length"]:]
                loss_fwd = self.getLoss(
                    output_batch,
                    target_batch,
                )
            else:
                loss_fwd = self.torchZero()

            if self.beta_vae and not self.has_rnn:
                loss_kl = self.getKLLoss(
                    beta_vae_mu,
                    beta_vae_logvar,
                )
            else:
                loss_kl = self.torchZero()

            if not detach_output:
                if self.has_rnn:
                    assert output_batch.size() == target_batch.size(
                    ), "ERROR: Output of network ({:}) does not match with target ({:}).".format(
                        output_batch.size(), target_batch.size())
                else:
                    assert input_batch.size() == input_batch_decoded.size(
                    ), "ERROR: Output of DECODER network ({:}) does not match INPUT ({:}).".format(
                        input_batch_decoded.size(), input_batch.size())

            if self.latent_forecasting_loss:
                outputs = latent_states_pred[:, :-1, :]
                targets = latent_states[:, 1:, :]

                outputs = outputs[:, -self.params["prediction_length"]:]
                targets = targets[:, -self.params["prediction_length"]:]

                # print(is_train)
                # print(np.shape(outputs))
                # print(outputs.min())
                # print(outputs.max())
                # print(np.shape(targets))
                # print(targets.min())
                # print(targets.max())
                # print(ark)

                loss_dyn_fwd = self.getLoss(
                    outputs,
                    targets,
                    is_latent=True,
                )
                # print(loss_dyn_fwd)
            else:
                loss_dyn_fwd = self.torchZero()

            # # print(input_batch.size())
            # import matplotlib
            # import matplotlib.pyplot as plt
            # # from matplotlib.pyplot import plot, draw, show
            # outputs_plot = latent_states_pred[:, :-1, 0].detach().numpy()
            # targets_plot = latent_states[:, 1:, 0].detach().numpy()
            # fig = plt.figure()
            # plt.plot(outputs_plot.T, "r")
            # plt.plot(targets_plot.T, "g")
            # fig.savefig(self.getFigureDir() + "/epoch_{:}_p{:}.png".format(self.epochs_iter, p))
            # plt.close()

            # import matplotlib
            # import matplotlib.pyplot as plt

            if self.reconstruction_loss:
                loss_auto_fwd = self.getLoss(
                    input_batch_decoded,
                    input_batch,
                )
            else:
                loss_auto_fwd = self.torchZero()

            if self.c1_latent_smoothness_loss and not self.has_rnn:
                loss_auto_fwd_c1 = self.getC1Loss(latent_states, )
            else:
                loss_auto_fwd_c1 = self.torchZero()

            # CONSTRUCTING THE LOSS
            loss_batch = 0.0
            num_losses = 0.0

            # ADDING THE FORWARD LOSS (be carefull, if loss_batch=loss_fwd, it is passed by reference!)
            if self.output_forecasting_loss:
                loss_batch += loss_fwd
                num_losses += 1.0
            if self.latent_forecasting_loss:
                loss_batch += loss_dyn_fwd
                num_losses += 1.0
            if self.reconstruction_loss:
                loss_batch += loss_auto_fwd
                num_losses += 1.0
            if self.c1_latent_smoothness_loss and (not self.has_rnn):
                loss_auto_fwd_c1 *= self.c1_latent_smoothness_loss_factor
                loss_batch += loss_auto_fwd_c1
                num_losses += 1.0
            if self.beta_vae and not self.has_rnn:
                beta_vae_weight = self.beta_vae_weight_max * self.getKLLossSchedule(
                    self.epochs_iter_global)
                loss_batch += beta_vae_weight * loss_kl
                num_losses += 1.0
            else:
                beta_vae_weight = 0.0

            loss_batch = loss_batch / num_losses

            # if is_train and p>0:
            if is_train:
                # loss_batch.requires_grad = True
                # loss_batch.backward(retain_graph=True)
                # print("loss_batch.backward()")
                loss_batch.backward()
                # print("self.optimizer.step()")
                self.optimizer.step()
                # if self.optimizer_str == "sgd": self.scheduler.step()

            loss_batch = loss_batch.cpu().detach().numpy()
            loss_fwd = loss_fwd.cpu().detach().numpy()
            loss_dyn_fwd = loss_dyn_fwd.cpu().detach().numpy()
            loss_auto_fwd = loss_auto_fwd.cpu().detach().numpy()
            loss_kl = loss_kl.cpu().detach().numpy()
            loss_auto_fwd_c1 = loss_auto_fwd_c1.cpu().detach().numpy()

            losses_batch = [
                loss_batch, loss_fwd, loss_dyn_fwd, loss_auto_fwd, loss_kl,
                loss_auto_fwd_c1
            ]

            # APPENDING LOSSES
            losses_vec.append(losses_batch)

            if self.RNN_statefull:
                # Propagating the hidden state
                last_hidden_state = self.detachHiddenState(last_hidden_state)
                initial_hidden_states = last_hidden_state
            else:
                initial_hidden_states = self.getInitialRNNHiddenState(
                    batch_size)

            #################################
            ### UPDATING BATCH IDX
            #################################

            predict_on = predict_on + self.sequence_length

        # print("np.mean(np.array(losses_vec), axis=0)")
        losses = np.mean(np.array(losses_vec), axis=0)
        # print(losses)
        # print(ark)
        return losses, iterative_forecasting_prob, beta_vae_weight

    def getKLLossSchedule(self, epoch):
        E = self.max_epochs
        # Use the lambertw function
        # Get the k coefficient, by setting the inflection point to E/2 or E/4
        from scipy.special import lambertw
        inflection_point = E / 4.0
        k = np.real(inflection_point / lambertw(inflection_point))
        weight = 1 - k / (k + np.exp(epoch / k))
        return weight

    def getIterativeForecastingProb(self, epoch, schedule):
        assert (schedule in [
            "linear_with_gradient", "linear_without_gradient",
            "inverse_sigmoidal_with_gradient",
            "inverse_sigmoidal_without_gradient", "exponential_with_gradient",
            "exponential_without_gradient"
        ])
        E = self.max_epochs
        if "linear" in schedule:
            c = 1.0 / E
            prob = c * epoch

        elif "exponential" in schedule:
            k = np.exp(np.log(0.001) / E)
            prob = 1 - np.power(k, epoch)

        elif "inverse_sigmoidal" in schedule:
            # Use the lambertw function
            # Get the k coefficient, by setting the inflection point to E/2 or E/4
            from scipy.special import lambertw
            inflection_point = E / 2.0
            k = np.real(inflection_point / lambertw(inflection_point))
            prob = 1 - k / (k + np.exp(epoch / k))

        return prob

    def torchZero(self):
        return self.torch_dtype([0.0])[0]

    def trainEpoch(self, data_loader, is_train=False, dataset=None):
        # if self.gpu and self.is_master: print(utils.getGpuMemoryMapString(self.smi_handle))
        epoch_losses_vec = []
        for batch_of_sequences in data_loader:
            # K, T, C, Dx, Dy
            losses, iterative_forecasting_prob, beta_vae_weight = self.trainOnBatch(
                batch_of_sequences, is_train=is_train, dataset=dataset)
            epoch_losses_vec.append(losses)
        epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)
        time_ = time.time() - self.start_time
        return epoch_losses, iterative_forecasting_prob, time_, beta_vae_weight

    def train(self):
        print("[dimred_rnn] # trainRNN() #")

        self.loadDimRedModel()

        # if self.load_trained_AE: self.loadAutoencoderModel()

        if self.gpu:
            self.gpu_monitor_process = utils.GPUMonitor(
                self.params["gpu_monitor_every"])

        data_loader_train, sampler_train, dataset_train = utils.getDataLoader(
            self.data_path_train,
            self.data_info_dict,
            self.batch_size,
            shuffle=True,
        )

        data_loader_val, _, dataset_val = utils.getDataLoader(
            self.data_path_val,
            self.data_info_dict,
            self.batch_size,
            shuffle=False,
        )

        # if self.has_autoencoder: self.loadAutoencoderOutputResidualScalingParams()

        if self.train_RNN:
            """ Before starting RNN training, scale the latent space """
            self.loadAutoencoderLatentStateLimits()

        self.sampler_train = sampler_train

        self.declareOptimizer(self.learning_rate)

        # Check if retraining
        if self.retrain == 1:
            print("[dimred_rnn] RESTORING pytorch model")
            self.load()

        elif self.train_RNN == 1:

            print("[dimred_rnn] LOADING dimensionality reduction model...")
            self.loadDimRedModel()

            if self.load_trained_AE:
                print("[dimred_rnn] LOADING autoencoder model...")
                self.loadAutoencoderModel()

            # Saving the initial state
            print("[dimred_rnn] Saving the initial model")
            torch.save(self.getModel().state_dict(), self.saving_model_path)

        elif self.load_trained_AE == 1:
            print("[dimred_rnn] LOADING autoencoder model: \n")
            self.loadDimRedModel()
            print("[dimred_rnn] LOADING autoencoder model...")
            self.loadAutoencoderModel()
            # Saving the initial state
            print("[dimred_rnn] Saving the initial model")
            torch.save(self.getModel().state_dict(), self.saving_model_path)

        self.loss_total_train_vec = []
        self.loss_total_val_vec = []

        self.losses_train_vec = []
        self.losses_val_vec = []

        self.losses_time_train_vec = []
        self.losses_time_val_vec = []

        self.ifp_train_vec = []
        self.ifp_val_vec = []

        self.learning_rate_vec = []

        self.beta_vae_weight_vec = []

        isWallTimeLimit = False

        # Termination criterion:
        # If the training procedure completed the maximum number of epochs

        # Learning rate decrease criterion:
        # If the validation loss does not improve for some epochs (patience)
        # the round is terminated, the learning rate decreased and training
        # proceeds in the next round.

        self.epochs_iter = 0
        self.epochs_iter_global = self.epochs_iter
        self.rounds_iter = 0
        # TRACKING
        # self.tqdm = tqdm(total=self.max_epochs, desc="{:}\t[dimred_rnn]".format(self.rank_str), leave=True)
        self.tqdm = tqdm(total=self.max_epochs)
        # self.tqdm = tqdm(total=self.max_epochs)
        while self.epochs_iter < self.max_epochs and self.rounds_iter < self.max_rounds:
            isWallTimeLimit = self.trainRound(
                data_loader_train,
                data_loader_val,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
            )
            # INCREMENTING THE ROUND NUMBER
            if isWallTimeLimit: break

        # If the training time limit was not reached, save the model...
        if not isWallTimeLimit:
            if self.epochs_iter == self.max_epochs:
                print(
                    "[dimred_rnn] Training finished. Maximum number of epochs reached."
                )
            elif self.rounds_iter == self.max_rounds:
                print(
                    "[dimred_rnn] Training finished. Maximum number of rounds reached."
                )
            else:
                if self.gpu: self.gpu_monitor_process.stop()
                raise ValueError(
                    "[dimred_rnn] Training finished in round {:}, after {:} total epochs. I do not know why!"
                    .format(self.rounds_iter, self.epochs_iter))

            self.save()
            utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                     self.loss_total_val_vec,
                                     self.min_val_total_loss)

            utils.plotAllLosses(self, self.losses_train_vec,
                                self.losses_time_train_vec,
                                self.losses_val_vec, self.losses_time_val_vec,
                                self.min_val_total_loss)
            utils.plotScheduleLoss(self, self.ifp_train_vec, self.ifp_val_vec)
            utils.plotScheduleLearningRate(self, self.learning_rate_vec)
            if self.beta_vae:
                utils.plotScheduleKLLoss(self, self.beta_vae_weight_vec)

        if self.gpu: self.gpu_monitor_process.stop()

    def printLosses(self, label, losses):
        self.losses_labels = [
            "TOTAL", "FWD", "DYN-FWD", "AUTO-REC", "KL", "C1"
        ]
        idx = np.nonzero(losses)[0]
        to_print = "[dimred_rnn] # {:s}-losses: ".format(label)
        for i in range(len(idx)):
            to_print += "{:}={:1.2E} |".format(self.losses_labels[idx[i]],
                                               losses[idx[i]])
        print(to_print)

    def printEpochStats(self, epoch_time_start, epochs_iter, epochs_in_round,
                        losses_train, losses_val):
        epoch_duration = time.time() - epoch_time_start
        time_covered = epoch_duration * epochs_iter
        time_total = epoch_duration * self.max_epochs
        percent = time_covered / time_total * 100
        label = "[dimred_rnn] EP={:} - R={:} - ER={:} - [ TIME= {:}, {:} / {:} - {:.2f} %] - LR={:1.2E}".format(
            epochs_iter, self.rounds_iter, epochs_in_round,
            utils.secondsToTimeStr(epoch_duration),
            utils.secondsToTimeStr(time_covered),
            utils.secondsToTimeStr(time_total), percent,
            self.learning_rate_round)

        size_of_print = len(label)
        print("[dimred_rnn] " + "-" *
              (size_of_print - len("[dimred_rnn] ") + 2))
        print(label)
        self.printLosses("TRAIN", losses_train)
        self.printLosses("VAL  ", losses_val)

    def printLearningRate(self):
        for param_group in self.optimizer.param_groups:
            print("[dimred_rnn] Current learning rate = {:}".format(
                param_group["lr"]))
        return 0

    def getModel(self):
        if (not self.gpu) or (self.device_count <= 1):
            return self.model
        elif self.gpu and self.device_count > 1:
            return self.model.module
        else:
            raise ValueError("{:}Value of self.gpu {:} not recognized.".format(
                self.rank_str, self.gpu))

    def trainRound(self,
                   data_loader_train,
                   data_loader_val,
                   dataset_train=None,
                   dataset_val=None):
        # Check if retraining of a model is requested else random initialization of the weights
        isWallTimeLimit = False

        # Setting the initial learning rate
        if self.rounds_iter == 0:
            if self.retrain and self.retrain_model_data_found:
                self.learning_rate_round = self.learning_rate_round
                self.previous_round_converged = 0
            else:
                self.learning_rate_round = self.learning_rate
                self.previous_round_converged = 0

        elif self.previous_round_converged == 0:
            self.learning_rate_round = self.learning_rate_round
            self.previous_round_converged = 0

        elif self.previous_round_converged == 1:
            self.previous_round_converged = 0
            self.learning_rate_round = self.learning_rate_round / 2

        if self.rounds_iter > 0:
            """ Optimizer has to be re-declared """
            self.declareOptimizer(self.learning_rate_round)
            """ Restore the model """
            print("[dimred_rnn] RESTORING pytorch model")
            self.getModel().load_state_dict(torch.load(self.saving_model_path))

        else:
            """ Saving the initial model """
            print("[dimred_rnn] Saving the initial model")
            torch.save(self.getModel().state_dict(), self.saving_model_path)

        print("[dimred_rnn] ### Round: {:}, Learning rate={:} ###".format(
            self.rounds_iter, self.learning_rate_round))

        losses_train, ifp_train, time_train, beta_vae_weight = self.trainEpoch(
            data_loader_train, is_train=False, dataset=dataset_train)
        if self.iterative_loss_validation: assert (ifp_train == 1.0)

        losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(
            data_loader_val, is_train=False, dataset=dataset_val)
        if self.iterative_loss_validation: assert (ifp_val == 1.0)

        label = "[dimred_rnn] INITIAL (NEW ROUND):  EP{:} - R{:}".format(
            self.epochs_iter, self.rounds_iter)
        print(label)
        self.printLosses("TRAIN", losses_train)
        self.printLosses("VAL  ", losses_val)

        self.min_val_total_loss = losses_val[0]
        self.loss_total_train = losses_train[0]

        RNN_loss_round_train_vec = []
        RNN_loss_round_val_vec = []

        RNN_loss_round_train_vec.append(losses_train[0])
        RNN_loss_round_val_vec.append(losses_val[0])

        self.loss_total_train_vec.append(losses_train[0])
        self.loss_total_val_vec.append(losses_val[0])

        self.losses_train_vec.append(losses_train)
        self.losses_time_train_vec.append(time_train)
        self.losses_val_vec.append(losses_val)
        self.losses_time_val_vec.append(time_val)

        for epochs_iter in range(self.epochs_iter, self.max_epochs + 1):
            epoch_time_start = time.time()
            epochs_in_round = epochs_iter - self.epochs_iter
            self.epochs_iter_global = epochs_iter

            losses_train, ifp_train, time_train, beta_vae_weight = self.trainEpoch(
                data_loader_train, is_train=True, dataset=dataset_train)
            losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(
                data_loader_val, is_train=False, dataset=dataset_val)
            RNN_loss_round_train_vec.append(losses_train[0])
            RNN_loss_round_val_vec.append(losses_val[0])
            self.loss_total_train_vec.append(losses_train[0])
            self.loss_total_val_vec.append(losses_val[0])

            self.losses_train_vec.append(losses_train)
            self.losses_time_train_vec.append(time_train)
            self.losses_val_vec.append(losses_val)
            self.losses_time_val_vec.append(time_val)

            self.ifp_val_vec.append(ifp_val)
            self.ifp_train_vec.append(ifp_train)
            self.beta_vae_weight_vec.append(beta_vae_weight)

            self.learning_rate_vec.append(self.learning_rate_round)

            self.printEpochStats(epoch_time_start, epochs_iter,
                                 epochs_in_round, losses_train, losses_val)

            if losses_val[0] < self.min_val_total_loss:
                print("[dimred_rnn] Saving model !")
                self.min_val_total_loss = losses_val[0]
                self.loss_total_train = losses_train[0]
                torch.save(self.getModel().state_dict(),
                           self.saving_model_path)

            if epochs_in_round > self.overfitting_patience:
                if all(self.min_val_total_loss <
                       RNN_loss_round_val_vec[-self.overfitting_patience:]):
                    self.previous_round_converged = True
                    break

            # # LEARNING RATE SCHEDULER (PLATEU ON VALIDATION LOSS)
            # if self.optimizer_str == "adam": self.scheduler.step(losses_val[0])
            self.tqdm.update(1)
            isWallTimeLimit = self.isWallTimeLimit()
            if isWallTimeLimit:
                break

        self.rounds_iter += 1
        self.epochs_iter = epochs_iter
        return isWallTimeLimit

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            print(
                "[dimred_rnn] ### Maximum train time reached: saving model... ###"
            )
            self.tqdm.close()
            self.save()
            utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                     self.loss_total_val_vec,
                                     self.min_val_total_loss)
            utils.plotAllLosses(self, self.losses_train_vec,
                                self.losses_time_train_vec,
                                self.losses_val_vec, self.losses_time_val_vec,
                                self.min_val_total_loss)
            utils.plotScheduleLoss(self, self.ifp_train_vec, self.ifp_val_vec)
            utils.plotScheduleLearningRate(self, self.learning_rate_vec)
            return True
        else:
            return False

    def delete(self):
        pass

    def loadDimRedModel(self):
        model_name_dimred = self.getAutoencoderName()
        print("[dimred_rnn] Loading dimensionality reduction from model: {:}".
              format(model_name_dimred))
        data_path = self.saving_path + self.model_dir + model_name_dimred + "/data"
        print("[dimred_rnn] Datafile: {:}".format(data_path))
        try:
            data = utils.loadData(data_path, "pickle")
        except Exception as inst:
            raise ValueError(
                "[Error] Dimensionality reduction results {:s} not found.".
                format(data_path))
        self.dimred_model = data["dimred_model"]
        del data
        return 0

    # def loadAutoencoderModel(self, in_cpu=False):
    #     model_name_autoencoder = self.createModelName(with_autoencoder=True, with_rnn=False)
    #     print("[crnn] Loading autoencoder with name:")
    #     print(model_name_autoencoder)
    #     AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model"
    #     # self.getModel().load_state_dict(torch.load(AE_path), strict=False)
    #     try:
    #         if not in_cpu and self.gpu:
    #             print("[crnn] # LOADING autoencoder model in GPU.")
    #             self.getModel().load_state_dict(torch.load(AE_path), strict=False)
    #         else:
    #             print("[crnn] # LOADING autoencoder model in CPU...")
    #             self.getModel().load_state_dict(torch.load(
    #                 AE_path, map_location=torch.device('cpu')), strict=False)
    #     except Exception as inst:
    #         print(
    #             "[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the autoencoder ? If you run on a cluster, is the GPU detected ? Did you use the srun command ?"
    #             .format(AE_path))
    #         raise ValueError(inst)
    #     AE_data_path = self.saving_path + self.model_dir + model_name_autoencoder + "/data"
    #     # data = utils.loadData(AE_data_path, "pickle")
    #     # del data
    #     return 0

    def save(self):
        print("[dimred_rnn] Recording time...")
        self.total_training_time = time.time() - self.start_time
        if hasattr(self, 'loss_total_train_vec'):
            if len(self.loss_total_train_vec) != 0:
                self.training_time = self.total_training_time / len(
                    self.loss_total_train_vec)
            else:
                self.training_time = self.total_training_time
        else:
            self.training_time = self.total_training_time

        print("[dimred_rnn] Total training time per epoch is {:}".format(
            utils.secondsToTimeStr(self.training_time)))
        print("[dimred_rnn] Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("[dimred_rnn] Script used {:} MB".format(self.memory))

        data = {
            "params": self.params,
            "model_name": self.model_name,
            "losses_labels": self.losses_labels,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "training_time": self.training_time,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_model_parameters": self.n_model_parameters,
            "loss_total_train_vec": self.loss_total_train_vec,
            "loss_total_val_vec": self.loss_total_val_vec,
            "min_val_total_loss": self.min_val_total_loss,
            "loss_total_train": self.loss_total_train,
            "losses_train_vec": self.losses_train_vec,
            "losses_time_train_vec": self.losses_time_train_vec,
            "losses_val_vec": self.losses_val_vec,
            "losses_time_val_vec": self.losses_time_val_vec,
            "ifp_val_vec": self.ifp_val_vec,
            "ifp_train_vec": self.ifp_train_vec,
            "learning_rate_vec": self.learning_rate_vec,
            "learning_rate": self.learning_rate,
            "learning_rate_round": self.learning_rate_round,
            "beta_vae_weight_vec": self.beta_vae_weight_vec,
        }
        fields_to_write = [
            "memory",
            "total_training_time",
            "n_model_parameters",
            "n_trainable_parameters",
            "min_val_total_loss",
        ]
        if self.write_to_log == 1:
            logfile_train = utils.getLogFileDir(self) + "/train.txt"
            print("[dimred_rnn] Writing to log-file in path {:}".format(
                logfile_train))
            utils.writeToLogFile(self, logfile_train, data, fields_to_write)

        data_path = utils.getModelDir(self) + "/data"
        utils.saveData(data, data_path, "pickle")

    def load(self, in_cpu=False):
        """ First load the dimensionality reduction model """
        self.loadDimRedModel()
        if self.load_trained_AE: self.loadAutoencoderModel()
        """ Then load the RNN model """
        try:
            if not in_cpu and self.gpu:
                print("[dimred_rnn] # LOADING model in GPU.")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path))

            else:
                print("[dimred_rnn] # LOADING model in CPU...")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path,
                               map_location=torch.device('cpu')))

        except Exception as inst:
            print(
                "[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the model?"
                .format(self.saving_model_path))
            raise ValueError(inst)

        print("[dimred_rnn] # Model loaded successfully!")

        data_path = utils.getModelDir(self) + "/data"

        try:
            data = utils.loadData(data_path, "pickle")
            self.loss_total_train_vec = data["loss_total_train_vec"]
            self.loss_total_val_vec = data["loss_total_val_vec"]
            self.min_val_total_loss = data["min_val_total_loss"]
            self.losses_time_train_vec = data["losses_time_train_vec"]
            self.losses_time_val_vec = data["losses_time_val_vec"]
            self.losses_val_vec = data["losses_val_vec"]
            self.losses_train_vec = data["losses_train_vec"]
            self.losses_labels = data["losses_labels"]
            self.ifp_train_vec = data["ifp_train_vec"]
            self.ifp_val_vec = data["ifp_val_vec"]
            self.learning_rate_vec = data["learning_rate_vec"]
            self.learning_rate_round = data["learning_rate_round"]
            self.beta_vae_weight_vec = data["beta_vae_weight_vec"]
            del data

            self.retrain_model_data_found = True

        except Exception as inst:
            print(
                "[Error (soft)] Model {:s} found. The data from training (result, losses, etc.), however, is missing."
                .format(self.saving_model_path))
            self.retrain_model_data_found = False
        return 0

    def computeLatentStateInfo(self, latent_states_all):
        #########################################################
        # In case of plain CNN (no MLP between encoder-decoder):
        # shape either  (n_ics, T, latent_state, 1, 1)
        # shape or      (n_ics, T, 1, 1, latent_state)
        #########################################################
        # In case of CNN-MLP (encoder-MLP-latent_space-decoder):
        # shape either  (n_ics, T, latent_state)
        # Case (n_ics, T, latent_state)
        assert len(np.shape(
            latent_states_all)) == 3, "np.shape(latent_states_all)={:}".format(
                np.shape(latent_states_all))
        latent_states_all = np.reshape(latent_states_all,
                                       (-1, self.latent_state_dim_cum))
        min_ = np.min(latent_states_all, axis=0)
        max_ = np.max(latent_states_all, axis=0)
        mean_ = np.mean(latent_states_all, axis=0)
        std_ = np.std(latent_states_all, axis=0)
        latent_state_info = {}
        latent_state_info["min"] = min_
        latent_state_info["max"] = max_
        latent_state_info["mean"] = mean_
        latent_state_info["std"] = std_
        return latent_state_info

    def loadAutoencoderOutputResidualScalingParams(self):
        print("[dimred_rnn] loadAutoencoderOutputResidualScalingParams()")
        model_name_autoencoder = self.createDimRedName()
        AE_results_testing_path = self.saving_path + self.results_dir + model_name_autoencoder + "/results_dimred_testing_val"

        try:
            data = utils.loadData(AE_results_testing_path, "pickle")

        except Exception as inst:
            print(
                "[Error] AE testing results file:\n{:}\nNOT FOUND. Result file from AE testing needed to load the bounds of the latent state."
                .format(AE_results_testing_path))
            raise ValueError(inst)

        if "abs_dimred_scaled_error_max" in data.keys():
            print(
                "[dimred_rnn] abs_dimred_scaled_error_max found in AE testing file."
            )
            abs_dimred_scaled_error_max = data["abs_dimred_scaled_error_max"]
        else:
            raise ValueError(
                "[dimred_rnn] abs_dimred_scaled_error_max not found in AE testing file."
            )

        self.model.setOutputResidualScalerParams(
            data=abs_dimred_scaled_error_max)

        del data
        return 0

    def loadAutoencoderLatentStateLimits(self):
        print("[dimred_rnn] loadAutoencoderLatentStateLimits()")

        if self.has_autoencoder:
            model_name_autoencoder = self.createModelName(
                with_autoencoder=True, with_rnn=False)
            AE_results_testing_path = self.saving_path + self.results_dir + model_name_autoencoder + "/results_dimred_testing_val"
        else:
            """ Loading dimensionality reduction results """
            model_name_autoencoder = self.getAutoencoderName()
            AE_results_testing_path = self.saving_path + self.results_dir + model_name_autoencoder + "/results_dimred_testing_val"

        try:
            data = utils.loadData(AE_results_testing_path, "pickle")

        except Exception as inst:
            print(
                "[Error] AE testing results file:\n{:}\nNOT FOUND. Result file from AE testing needed to load the bounds of the latent state."
                .format(AE_results_testing_path))
            raise ValueError(inst)

        if "latent_state_info" in data.keys():
            print("[dimred_rnn] latent bounds found in AE testing file.")
            latent_state_info = data["latent_state_info"]
        else:
            print(
                "[dimred_rnn] latent bounds not found in AE testing file. Computing them..."
            )
            # for key in data: print(key)
            # Loading the bounds of the latent state
            latent_states_all = data["latent_states_all"]
            latent_state_info = self.computeLatentStateInfo(latent_states_all)
        self.model.setLatentStateBounds(min_=latent_state_info["min"],
                                        max_=latent_state_info["max"],
                                        mean_=latent_state_info["mean"],
                                        std_=latent_state_info["std"])
        del data
        return 0

    def getTestingModes(self):
        modes = []
        if self.has_rnn:
            # if self.params[
            #     "iterative_state_forecasting"]:
            #     modes.append("iterative_state_forecasting")
            if self.params["iterative_latent_forecasting"]:
                modes.append("iterative_latent_forecasting")
            if self.params["teacher_forcing_forecasting"]:
                modes.append("teacher_forcing_forecasting")

        elif self.has_autoencoder:
            modes.append("dimred_testing")
        return modes

    def test(self):
        if self.gpu and self.params["gpu_monitor_every"] > 0:
            self.gpu_monitor_process = utils.GPUMonitor(
                self.params["gpu_monitor_every"])
        self.test_()
        if self.gpu and self.params["gpu_monitor_every"] > 0:
            self.gpu_monitor_process.stop()

    def test_(self):
        if self.load() == 0:
            test_on = []
            self.n_warmup = self.params["n_warmup"]
            assert self.n_warmup > 0
            print("[dimred_rnn] Warming-up steps: {:d}".format(self.n_warmup))

            testing_modes = self.getTestingModes()
            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            for set_ in test_on:
                common_testing.testModesOnSet(self,
                                              set_=set_,
                                              testing_modes=testing_modes)
        return 0

    def testMaster(self):
        if self.load() == 0:
            # MODEL LOADED IN EVALUATION MODE
            with torch.no_grad():
                if self.params["n_warmup"] is None:
                    self.n_warmup = self.sequence_length
                else:
                    self.n_warmup = int(self.params["n_warmup"])

                print(
                    "[dimred_rnn] WARMING UP STEPS (for statefull RNNs): {:d}".
                    format(self.n_warmup))

                test_on = []
                if self.params["test_on_test"]: test_on.append("test")
                if self.params["test_on_val"]: test_on.append("val")
                if self.params["test_on_train"]: test_on.append("train")

                for set_ in test_on:
                    self.testOnSet(set_)
        return 0

    def forward(self,
                input_sequence,
                init_hidden_state,
                input_is_latent=False,
                iterative_propagation_is_latent=False):
        input_sequence = utils.transform2Tensor(self, input_sequence)
        init_hidden_state = utils.transform2Tensor(self, init_hidden_state)
        outputs, next_hidden_state, latent_states, latent_states_pred, _, _, time_latent_prop_t, _, _ = self.model.forward(
            input_sequence,
            init_hidden_state,
            is_train=False,
            is_iterative_forecasting=False,
            iterative_forecasting_prob=0,
            iterative_forecasting_gradient=0,
            horizon=None,
            input_is_latent=input_is_latent,
            iterative_propagation_is_latent=iterative_propagation_is_latent)
        outputs = outputs.detach().cpu().numpy()
        latent_states_pred = latent_states_pred.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        return outputs, next_hidden_state, latent_states, latent_states_pred, time_latent_prop_t

    def forecast(self, input_sequence, hidden_state, horizon):
        input_sequence = utils.transform2Tensor(self, input_sequence)
        hidden_state = utils.transform2Tensor(self, hidden_state)
        outputs, next_hidden_state, latent_states, latent_states_pred, _, _, time_latent_prop, _, _ = self.model.forward(
            input_sequence,
            hidden_state,
            is_train=False,
            is_iterative_forecasting=True,
            iterative_forecasting_prob=1.0,
            horizon=horizon,
            iterative_propagation_is_latent=True,
            input_is_latent=True,
        )
        outputs = outputs.detach().cpu().numpy()
        latent_states_pred = latent_states_pred.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        return outputs, next_hidden_state, latent_states, latent_states_pred, time_latent_prop

    def predictSequence(self,
                        input_sequence,
                        testing_mode=None,
                        dt=1,
                        prediction_horizon=None):
        print("[dimred_rnn] # predictSequence() #")
        print("[dimred_rnn] {:}:".format(np.shape(input_sequence)))
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.n_warmup != prediction_horizon:
            raise ValueError(
                "[dimred_rnn] Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})"
                .format(N, self.n_warmup, prediction_horizon))

        # PREPARING THE HIDDEN STATES
        initial_hidden_states = self.getInitialRNNHiddenState(1)

        if self.has_rnn:
            assert self.n_warmup >= 1, "Warm up steps cannot be < 1 in RNNs. Increase the iterative prediction length."
        elif self.has_predictor:
            assert self.n_warmup == 1, "Warm up steps cannot be != 1 in Predictor."

        warmup_data_input = input_sequence[:self.n_warmup - 1]
        warmup_data_input = warmup_data_input[np.newaxis]
        warmup_data_target = input_sequence[1:self.n_warmup]
        warmup_data_target = warmup_data_target[np.newaxis]

        if testing_mode in self.getTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup +
                                    prediction_horizon]
        else:
            raise ValueError(
                "[dimred_rnn] Testing mode {:} not recognized.".format(
                    testing_mode))

        warmup_data_input = utils.transform2Tensor(self, warmup_data_input)
        initial_hidden_states = utils.transform2Tensor(self,
                                                       initial_hidden_states)

        if self.n_warmup > 1:
            warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _, _, _, _, _ = self.model.forward(
                warmup_data_input, initial_hidden_states, is_train=False)

            warmup_data_output = warmup_data_output.cpu().detach().numpy()
            warmup_data_output = np.reshape(warmup_data_output,
                                            np.shape(warmup_data_target))
        else:
            # In case of predictor with n_warmup=1 (no warmup)
            # assert(self.has_predictor)
            last_hidden_state = initial_hidden_states

        if ("iterative_latent" in testing_mode):
            iterative_propagation_is_latent = 1
            # GETTING THE LAST LATENT STATE (K, T, LD)
            # In iterative latent forecasting, the input is the latent state
            input_latent = latent_states_pred[:, -1, :]
            input_latent.unsqueeze_(0)
            input_t = input_latent
        elif "teacher_forcing" in testing_mode:
            iterative_propagation_is_latent = 0
            input_t = input_sequence[self.n_warmup - 1:-1]
            input_t = input_t.cpu().detach().numpy()
            input_t = input_t[np.newaxis]

        else:
            raise ValueError(
                "[dimred_rnn] I do not know how to initialize the state for {:}."
                .format(testing_mode))

        input_t = utils.transform2Tensor(self, input_t)

        if self.gpu:
            input_t = input_t.cuda()
            last_hidden_state = self.sendHiddenStateToGPU(last_hidden_state)

        time_start = time.time()
        if "teacher_forcing" in testing_mode:
            input_t = utils.transform2Tensor(self, input_t)
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop, _, _ = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=False,
                horizon=prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=False)
        elif "iterative_latent" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            input_t = utils.transform2Tensor(self, input_t)
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop, _, _ = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=True,
                iterative_forecasting_prob=1.0,
                horizon=prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=iterative_propagation_is_latent,
            )
        else:
            raise ValueError(
                "[dimred_rnn] Testing mode {:} not recognized.".format(
                    testing_mode))
        time_end = time.time()
        time_total = time_end - time_start

        # Correcting the time-measurement in case of evolution of the original system (in this case, we do not need to internally propagate the latent space of the RNN)
        time_total = time_latent_prop

        time_total_per_iter = time_total / prediction_horizon

        target = target.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        latent_states = latent_states.cpu().detach().numpy()
        if self.has_rnn: RNN_outputs = RNN_outputs.cpu().detach().numpy()

        prediction = prediction[0]
        prediction = np.reshape(prediction, np.shape(target))
        # print(np.shape(prediction))
        # print(ark)

        if self.has_rnn: RNN_outputs = RNN_outputs[0]
        latent_states = latent_states[0]

        target = np.array(target)
        prediction = np.array(prediction)
        latent_states = np.array(latent_states)
        if self.has_rnn: RNN_outputs = np.array(RNN_outputs)

        print("[dimred_rnn] Shapes of prediction/target/latent_states:")
        print("[dimred_rnn] {:}".format(np.shape(prediction)))
        print("[dimred_rnn] {:}".format(np.shape(target)))
        print("[dimred_rnn] {:}".format(np.shape(latent_states)))

        # print("Min/Max")
        # print("Target:")
        # print(np.max(target[:,0]))
        # print(np.min(target[:,0]))
        # print("Prediction:")
        # print(np.max(prediction[:,0]))
        # print(np.min(prediction[:,0]))

        if self.n_warmup > 1:
            warmup_data_target = warmup_data_target.cpu().detach().numpy()
            # warmup_data_output = warmup_data_output.cpu().detach().numpy()
            warmup_latent_states = warmup_latent_states.cpu().detach().numpy()

            target_augment = np.concatenate((warmup_data_target[0], target),
                                            axis=0)
            prediction_augment = np.concatenate(
                (warmup_data_output[0], prediction), axis=0)
            latent_states_augmented = np.concatenate(
                (warmup_latent_states[0], latent_states), axis=0)
        else:
            # assert(self.has_predictor)
            target_augment = target
            prediction_augment = prediction
            latent_states_augmented = latent_states

        print(
            "[dimred_rnn] Shapes of prediction/target/latent_states (augmented):"
        )
        print("[dimred_rnn] {:}".format(np.shape(prediction_augment)))
        print("[dimred_rnn] {:}".format(np.shape(target_augment)))
        print("[dimred_rnn] {:}".format(np.shape(latent_states_augmented)))

        return prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter

    def plot(self):
        if self.write_to_log:
            for testing_mode in self.getTestingModes():
                common_plot.writeLogfiles(self, testing_mode=testing_mode)
        else:
            print("[dimred] # write_to_log=0. #")

        if self.params["plotting"]:
            for testing_mode in self.getTestingModes():
                common_plot.plot(self, testing_mode=testing_mode)
        else:
            print("[dimred] # plotting=0. No plotting. #")
