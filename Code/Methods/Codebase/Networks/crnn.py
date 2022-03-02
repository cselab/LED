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
from . import crnn_model
import h5py
""" Printing """
from functools import partial

print = partial(print, flush=True)
import warnings
""" Common libraries """
from . import common_testing
from . import common_plot
""" Horovod """
try:
    import horovod.torch as hvd
    # print("[crnn] Horovod : ".format(hvd.__version__))
    print("[crnn] Imported Horovod.")
except ImportError:
    hvd = None


class crnn():
    def __init__(self, params):
        super(crnn, self).__init__()
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

        # Whether to run on multiple nodes (multi-GPU parallelism)
        self.multinode = params["multinode"]

        self.reference_train_time = utils.getReferenceTrainingTime(
            params["reference_train_time"], params["buffer_train_time"])

        # Checking whether the GPU is available and setting the default tensor datatype
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            if self.params["precision"] == "double":
                self.torch_dtype = torch.cuda.DoubleTensor
            elif self.params["precision"] == "single":
                self.torch_dtype = torch.cuda.FloatTensor
            else:
                raise ValueError("Invalid precision {:}.".format(
                    self.params["precision"]))

            if self.params["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True
        else:
            if self.params["precision"] == "double":
                self.torch_dtype = torch.DoubleTensor
            elif self.params["precision"] == "single":
                self.torch_dtype = torch.FloatTensor
            else:
                raise ValueError("Invalid precision {:}.".format(
                    self.params["precision"]))

        self.batch_size = params['batch_size']

        if self.multinode:

            if hvd is None:
                raise ValueError(
                    "[crnn] Horovod is not installed/configured properly. Import failed. Multi-node run is not possible without Horovod. Please install Horovod before running in multi-node mode."
                )

            # Initialize Horovod library (before setting any random seed)
            hvd.init()
            self.rank = hvd.rank()
            self.local_rank = hvd.local_rank()
            self.is_master = not self.rank
            self.rank_str = "[Rank {:}] ".format(self.rank)
            self.world_size = hvd.size()
            self.print(
                "# Horovod initialized in [global rank: {:} / {:}], [local rank: {:}]."
                .format(self.rank, self.world_size, self.local_rank))

            if self.batch_size % self.world_size == 0:
                batch_size_total = self.batch_size
                self.batch_size = int(batch_size_total // self.world_size)
                if self.is_master:
                    self.print(
                        "# Batch size of local ranks: {:} / {:}.".format(
                            self.batch_size, batch_size_total))
            else:
                raise ValueError(
                    "[deepledZ] Horovod: In multi-gpu training, the total batch size {:} needs to be divisible by the number of ranks {:}."
                    .format(self.batch_size, self.world_size))

        else:
            self.rank = 0
            self.local_rank = 0
            self.is_master = True
            self.rank_str = ""
            self.world_size = 1

        ##################################################################
        # RANDOM SEEDING
        ##################################################################
        self.random_seed = params["random_seed"]

        # Setting the random seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu: torch.cuda.manual_seed(self.random_seed)

        self.iterative_propagation_during_training_is_latent = params[
            "iterative_propagation_during_training_is_latent"]

        self.iterative_loss_schedule_and_gradient = params[
            "iterative_loss_schedule_and_gradient"]
        self.iterative_loss_validation = params["iterative_loss_validation"]
        if self.iterative_loss_schedule_and_gradient not in [
                "none",
                "exponential_with_gradient",
                "linear_with_gradient",
                "inverse_sigmoidal_with_gradient",
                "exponential_without_gradient",
                "linear_without_gradient",
                "inverse_sigmoidal_without_gradient",
        ]:
            raise ValueError(
                "Iterative loss schedule {:} not recognized.".format(
                    self.iterative_loss_schedule_and_gradient))
        else:
            if "without_gradient" in self.iterative_loss_schedule_and_gradient or "none" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = 0
            elif "with_gradient" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = 1
            else:
                raise ValueError(
                    "self.iterative_loss_schedule_and_gradient={:} not recognized."
                    .format(self.iterative_loss_schedule_and_gradient))

        # Optimizer to use
        self.optimizer_str = params["optimizer_str"]

        # Training loss (mse or crossentropy)
        self.training_loss = params["training_loss"]
        assert (self.training_loss in ["mse", "crossentropy"])

        # Whether the autoencoder is convolutional or not
        self.AE_convolutional = params["AE_convolutional"]
        self.AE_batch_norm = params["AE_batch_norm"]
        self.AE_conv_transpose = params["AE_conv_transpose"]
        self.AE_pool_type = params["AE_pool_type"]

        # Beta variational autoencoder
        self.beta_vae = params["beta_vae"]
        self.beta_vae_weight_max = params["beta_vae_weight_max"]

        self.c1_latent_smoothness_loss = params["c1_latent_smoothness_loss"]
        self.c1_latent_smoothness_loss_factor = params[
            "c1_latent_smoothness_loss_factor"]

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

        ##################################################################
        # SETTING FOR CONVOLUTIONAL RNN
        ##################################################################
        # Convolutional RNN
        self.RNN_convolutional = params['RNN_convolutional']
        # The kernel of convolutional RNN
        self.RNN_kernel_size = params['RNN_kernel_size']
        # The kernel of convolutional RNN
        self.RNN_layers_size = params['RNN_layers_size']
        # If the initial state of the RNN is trainable
        self.RNN_trainable_init_hidden_state = params[
            'RNN_trainable_init_hidden_state']
        # If the RNN is statefull, iteratively propagating the hidden state during training
        self.RNN_statefull = params['RNN_statefull']

        self.input_dim = params['input_dim']

        self.channels = params['channels']
        self.Dz, self.Dy, self.Dx = utils.getChannels(self.channels, params)

        # Dropout probabilities for regularizing the AE
        self.dropout_keep_prob = params["dropout_keep_prob"]
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

        if self.c1_latent_smoothness_loss:
            assert self.sequence_length > 1, "c1_latent_smoothness_loss cannot be used with sequence_length={:}<1.".format(
                self.sequence_length)

        ##################################################################
        # SCALER
        ##################################################################
        self.scaler = params["scaler"]

        ##################################################################
        # TRAINING PARAMETERS
        ##################################################################
        # Whether to retrain or not
        self.retrain = params['retrain']

        self.overfitting_patience = params['overfitting_patience']
        self.max_epochs = params['max_epochs']
        self.max_rounds = params['max_rounds']
        self.learning_rate = params['learning_rate']
        self.lr_reduction_factor = params['lr_reduction_factor']
        self.weight_decay = params['weight_decay']

        if self.multinode:
            # assert self.max_rounds==1, "In multinode set the parameters max_rounds=1 and with_scheduler=1. Current setting of max_rounds={:} not supported.".format(self.max_rounds)
            self.print(
                "In multinode setting, max_rounds=1 and max_scheduler_rounds has to be set. Found argument max_rounds={:}. Setting max_scheduler_rounds=max_rounds."
                .format(self.max_rounds))
            self.max_scheduler_rounds = self.max_rounds
            self.max_rounds = 1
            self.with_scheduler = True
            assert self.optimizer_str not in [
                "adabelief"
            ], "Multinode setting does not support Adabelief."
        else:
            self.with_scheduler = False

        self.train_AE_only = params["train_AE_only"]
        self.train_RNN_only = params["train_RNN_only"]
        self.load_trained_AE = params["load_trained_AE"]

        self.output_forecasting_loss = params["output_forecasting_loss"]
        self.latent_forecasting_loss = params["latent_forecasting_loss"]
        self.reconstruction_loss = params["reconstruction_loss"]

        if (params["latent_state_dim"] == 0
                or params["latent_state_dim"] is None):
            self.has_autoencoder = 0
        else:
            self.has_autoencoder = 1

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
            if self.RNN_cell_type not in [
                    'lstm', 'gru', 'lstm_2', 'lstm_3', 'mlp'
            ]:
                raise ValueError('Error: Invalid RNN cell type.')
        else:
            self.has_rnn = 0

        if (self.latent_forecasting_loss == 1 or self.reconstruction_loss
                == 1) and (self.has_autoencoder == 0):
            raise ValueError(
                "latent_forecasting_loss and reconstruction_loss are not meaningfull without latent state (Autoencoder mode)."
            )

        # Adding the autoencoder latent dimension if this is not None
        if params["latent_state_dim"] is not None and (
                params["latent_state_dim"] > 0):
            if not self.AE_convolutional:
                """ Parsing the Autoencoder layers """
                self.layers_encoder = [self.params["AE_layers_size"]
                                       ] * self.params["AE_layers_num"]
                self.layers_decoder = self.layers_encoder[::-1]

                self.latent_state_dim = self.params["latent_state_dim"]
                self.params["RNN_state_dim"] = self.latent_state_dim
                self.params["decoder_input_dim"] = self.params["RNN_state_dim"]

                if self.is_master:
                    self.print("[crnn] - Encoder layers: {:}".format(
                        self.layers_encoder))
                    self.print("[crnn] - Decoder layers: {:}".format(
                        self.layers_decoder))

            self.iterative_propagation_during_training_is_latent = params[
                "iterative_propagation_during_training_is_latent"]
            self.has_dummy_autoencoder = False
        else:
            self.layers_encoder = []
            self.params["AE_layers_size"] = 0
            self.params["AE_layers_num"] = 0

            if self.RNN_convolutional:
                self.params["RNN_state_dim"] = params["input_dim"]
                self.has_dummy_autoencoder = False
                assert self.RNN_kernel_size > 0
            else:
                if self.channels == 1:
                    total_dim = params["input_dim"] * self.Dx
                elif self.channels == 2:
                    total_dim = params["input_dim"] * self.Dx * self.Dy
                else:
                    raise ValueError("Not implemented.")
                self.params["RNN_state_dim"] = total_dim
                self.has_dummy_autoencoder = True

            self.iterative_propagation_during_training_is_latent = 0

        self.model = crnn_model.crnn_model(params, self)
        self.latent_state_dim = self.params["latent_state_dim"]
        self.model.printModuleList()

        self.model_name = self.createModelName()
        if self.is_master: self.print("[crnn] - model_name:")
        if self.is_master: self.print("[crnn] {:}".format(self.model_name))
        self.saving_model_path = utils.getModelDir(self) + "/model"
        """ Make directories if master """
        if self.is_master: utils.makeDirectories(self)
        """ Print parameters before parallelization """
        if self.is_master: self.printParams()

        self.model_parameters, self.model_named_params = self.model.getParams()
        """ Initialize model parameters """
        if self.is_master: self.model.initializeWeights()
        """ No broadcast yet. Broadcast after initialization """
        # if self.multinode: self.broadcastParameters()

        self.device_count = torch.cuda.device_count()

        if self.gpu:
            self.print("[crnn] USING CUDA -> SENDING THE MODEL TO THE GPU.")
            torch.cuda.set_device(self.local_rank)
            self.model.sendModelToCuda()

            if self.device_count > 1 and (not self.multinode):
                raise ValueError(
                    "More than one GPU devide detected. Aborting.")

        # utils.printGPUInfo(self.device_count)
        """ Saving some info file for the model """
        data = {
            # "model": self,
            "params": params,
            "selfParams": self.params,
            "name": self.model_name
        }
        data_path = utils.getModelDir(self) + "/info"
        if self.is_master: utils.saveData(data, data_path, "pickle")
        self.data_info_dict = systems.getSystemDataInfo(self)

        # if self.is_master: self.getPropagationTime()

    def print(self, str_):
        if self.multinode:
            print("{:}{:}".format(self.rank_str, str_))
        else:
            print(str_)

    def getPropagationTime(self):
        if self.AE_convolutional:
            if self.channels == 2:
                times_ = 10
                time_prop = 0.0
                for i in range(times_):
                    input_ = torch.randn([
                        self.batch_size,
                        1,
                        self.input_dim,
                        self.Dx,
                        self.Dy,
                    ], )
                    time_start = time.time()
                    input_ = utils.transform2Tensor(self, input_)
                    input_ = self.model.forwardEncoder(input_)
                    input_ = self.model.forwardDecoder(input_)
                    time_prop += time.time() - time_start
                time_prop = time_prop / self.batch_size / times_
                self.print(
                    "[Autoencoder] Propagation time = {:}".format(time_prop))
        return 0

    def getKeysInModelName(self, with_autoencoder=True, with_rnn=True):
        keys = {
            'scaler': '-SC_',
            'optimizer_str': '-OPT_',
        }

        if not (self.params["precision"] == "double"):
            keys.update({'precision': '-PREC_'})

        if self.has_autoencoder and with_autoencoder:
            if self.params["learning_rate_in_name"]:
                if self.has_rnn and self.params["learning_rate_AE"] is not None:
                    keys.update({'learning_rate_AE': '-LR_'})
                    keys.update({'noise_level_AE': '-NL_'})
                else:
                    keys.update({'learning_rate': '-LR_'})
                    keys.update({'noise_level': '-NL_'})

        keys.update({
            'weight_decay': '-L2_',
        })

        if self.has_autoencoder and with_autoencoder:
            if self.has_rnn and self.load_trained_AE:
                if self.params["random_seed_in_AE_name"]:
                    keys.update({'random_seed_in_AE_name': '-RS_'})
            elif self.params["random_seed_in_name"]:
                keys.update({'random_seed': '-RS_'})

            # keys.update( { 'input_dim':'-DIM_', })
            if self.AE_convolutional:
                # keys.update( { 'AE_convolutional':'-CONV-AUTO_', })
                keys.update({
                    'conv_layers_channels': '-CNL_',
                    'conv_layers_kernel_sizes': '-KRN_',
                    'AE_batch_norm': '-BN_',
                })
                if self.params["AE_batch_norm_affine"]:
                    keys.update({'AE_batch_norm_affine': '-AF_'})
                keys.update({
                    'AE_conv_transpose': '-TR_',
                    'AE_interp_subsampling_input': '-SI_',
                    'AE_pool_type': '-PL_',
                })

                if self.params["activation_str_output"] != "tanhplus":
                    keys.update({'activation_str_output': '-ACO_'})

                if self.beta_vae:
                    keys.update({
                        'beta_vae_weight_max': '-BVAE_',
                    })
                if self.c1_latent_smoothness_loss:
                    keys.update({
                        'c1_latent_smoothness_loss_factor': '-C1LOSS_',
                    })
            else:
                keys.update({
                    'AE_layers_num': '-AUTO_',
                    'AE_layers_size': 'x',
                })
            keys.update({
                'activation_str_general': '-ACT_',
                'dropout_keep_prob': '-DKP_',
                'latent_state_dim': '-LD_',
                # 'output_forecasting_loss':'-ForLoss_',
                # 'latent_forecasting_loss':'-DynLoss_',
                # 'reconstruction_loss':'-RecLoss_',
            })

        if self.has_rnn and with_rnn:  # RNN MODE
            if self.load_trained_AE:
                keys.update({
                    'load_trained_AE': '-PRETRAIN-AE_',
                })
            if self.RNN_convolutional:
                keys.update({
                    'RNN_convolutional': '-RNN-CONV_',
                    'RNN_activation_str': '-ACT_',
                })
            if self.training_loss in ["crossentropy"]:
                keys.update({
                    'training_loss': '-LOSS_',
                })
            if self.params["RNN_trainable_init_hidden_state"]:
                keys.update({
                    'RNN_trainable_init_hidden_state': '-TIHS_',
                })

            if self.params["random_seed_in_name"]:
                keys.update({'random_seed': '-RS_'})

            if self.params["learning_rate_in_name"]:
                keys.update({'learning_rate': '-LR_'})

            keys.update({
                'RNN_cell_type': '-C_',
                # 'RNN_activation_str':'-ACT_',
                'RNN_layers_num': '-R_',
                'RNN_layers_size': 'x',
                # 'zoneout_keep_prob':'-ZKP_',
                'sequence_length': '-SL_',
                # 'prediction_horizon':'-PH_',
                # 'num_test_ICS':'-NICS_',
                'output_forecasting_loss': '-LFO_',
                'latent_forecasting_loss': '-LFL_',
            })
            if self.params["iterative_loss_schedule_and_gradient"] not in [
                    "none"
            ]:
                keys.update({
                    'iterative_loss_schedule_and_gradient': '-ITS_',
                })
            if self.params["iterative_loss_validation"]:
                keys.update({
                    'iterative_loss_validation': '-ITV_',
                })

        return keys

    def createModelName(self):
        # print("# createModelName() #")
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.gpu + "ARNN"
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        return str_

    def getAutoencoderName(self):
        keys = self.getKeysInModelName(with_rnn=False)
        str_ = "GPU-" * self.gpu + "ARNN"
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        return str_

    def printParams(self):
        self.n_trainable_parameters = self.model.countTrainableParams()
        self.n_model_parameters = self.model.countParams()
        # Print parameter information:
        self.print("[crnn] Trainable params {:}/{:}".format(
            self.n_trainable_parameters, self.n_model_parameters))
        return 0

    def broadcastParameters(self):
        if self.is_master:
            self.print("Master rank broadcasting the model parameters.")
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        return 0

    def broadcastOptimizerState(self):
        if self.is_master:
            self.print("Master rank broadcasting the optimizer state.")
        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.none if (self.params["hvd_compression"]
                                               == 0) else hvd.Compression.fp16
        # compression = hvd.Compression.none
        operation = hvd.Average if (self.params["hvd_adasum"]
                                    == 0) else hvd.Adasum
        # operation = hvd.Average
        # self.params_trainable
        # self.model.named_parameters()
        named_parameters = self.model.named_parameters()
        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=named_parameters,
            compression=compression,
            op=operation,
            backward_passes_per_step=5,
        )
        return 0

    def declareScheduler(self):
        """ Only define scheduler in case of multinode """
        if self.with_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=self.lr_reduction_factor,
                patience=self.overfitting_patience,
                verbose=True)
        else:
            self.scheduler = None

    def declareOptimizer(self, lr):
        # print("[crnn] Learning rate: {}".format(lr))
        if self.train_AE_only:
            # Get the autoencoder params
            self.params_trainable, self.params_named_trainable = self.model.getAutoencoderParams(
            )

            # If it has RNN disable the gradient of the RNN params
            if self.has_rnn:
                rnn_params, rnn_named_params = self.model.getRNNParams()
                for name, param in rnn_named_params:
                    param.requires_grad = False

        elif self.train_RNN_only:

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
        if self.has_rnn and not self.train_AE_only:
            weight_decay = 0.0
            if self.weight_decay > 0:
                if self.is_master:
                    self.print("[crnn] No weight decay in RNN training.")
        else:
            weight_decay = self.weight_decay

        if self.is_master:
            self.print("[crnn] Learning rate: {:}, Weight decay: {:}".format(
                lr, self.weight_decay))
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
            self.print("{:s} batch {:d}/{:d},  {:f}%".format(
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

    def multinodeMetric(self, tensor, name):
        if self.multinode:
            # tensor = self.torch_dtype(tensor)
            tensor = hvd.allreduce(tensor, name=name)
            # temp = tensor.item()
        return tensor

    def getKLLoss(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def getC1Loss(self, latent_states):
        c1_loss = torch.pow(latent_states[:, 1:] - latent_states[:, :-1], 2.0)
        c1_loss = torch.mean(c1_loss)
        return c1_loss

    def getLoss(
        self,
        output,
        target,
        is_latent=False,
    ):

        # self.noise_level = 0.01
        # target += self.noise_level * torch.randn_like(target)

        if "mse" in self.training_loss:

            if self.beta_vae and not self.has_rnn:
                # Sum reduction
                loss = output - target
                loss = loss.pow(2.0)
                # Mean over all dimensions
                loss = loss.sum()

            # elif self.params["precision"] == "single":
            #     # print(output.size())
            #     # print(target.size())
            #     # Mean squared loss
            #     loss = output - target
            #     loss = loss.pow(2.0)
            #     # Mean over all dimensions
            #     loss = loss.mean(2)
            #     # Sum over all batches
            #     loss = loss.sum(0)
            #     # Sum over all time-steps
            #     loss = loss.sum()

            else:
                # print(output.size())
                # print(target.size())
                # Mean squared loss
                loss = output - target
                loss = loss.pow(2.0)
                # Mean over all dimensions
                loss = loss.mean(2)
                # Mean over all batches
                loss = loss.mean(0)
                # Mean over all time-steps
                loss = loss.mean()

        elif "crossentropy" in self.training_loss:
            # print(target.min())
            # print(target.max())
            # print(target)
            # temp  = target.numpy()
            # print(temp)
            # temp= temp.flatten()
            # temp = set(temp)
            # print(temp)
            cross_entropy = torch.nn.BCELoss(reduction='none')
            # Cross-entropy loss
            # epsilon = 1e-45
            # loss = - target * torch.log(output + epsilon) + (1-target) * torch.log(1-output + epsilon)
            loss = cross_entropy(output, target)
            loss = loss.mean()
            # print(cross_entropy)
            # print(loss.size())
            # print(ark)

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

    def trainOnBatch(self, batch_of_sequences, is_train=False, dataset=None):
        batch_size = len(batch_of_sequences)
        initial_hidden_states = self.getInitialRNNHiddenState(batch_size)

        if self.data_info_dict["structured"]:
            T = dataset.seq_paths[0]["num_timesteps"]
        else:
            T = np.shape(batch_of_sequences)[1]

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

            # Transform to pytorch and forward the network
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
        # print(np.shape(batch_of_sequences))
        # print("predict_on")
        # print(predict_on)
        for p in range(num_propagations):
            # print("Propagation {:}/{:}".format(p, num_propagations))
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()
            """ Getting the batch """

            input_batch = utils.getDataBatch(self,
                                             batch_of_sequences,
                                             predict_on - self.sequence_length,
                                             predict_on,
                                             dataset=dataset)
            target_batch = utils.getDataBatch(self,
                                              batch_of_sequences,
                                              predict_on -
                                              self.sequence_length + 1,
                                              predict_on + 1,
                                              dataset=dataset)

            # print("np.shape(input_batch)")
            # print(np.shape(input_batch))
            # print("np.shape(target_batch)")
            # print(np.shape(target_batch))

            # Transform to pytorch and forward the network
            input_batch = utils.transform2Tensor(self, input_batch)
            target_batch = utils.transform2Tensor(self, target_batch)
            initial_hidden_states = utils.transform2Tensor(
                self, initial_hidden_states)
            # if self.gpu: initial_hidden_states = self.sendHiddenStateToGPU(initial_hidden_states)

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
            """ In case 
            """

            if self.has_rnn and self.has_autoencoder and self.latent_forecasting_loss and not self.output_forecasting_loss:
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

            # if torch.is_tensor(input_batch): print(input_batch.size())
            # if torch.is_tensor(target_batch): print(target_batch.size())
            # if torch.is_tensor(input_batch_decoded): print(input_batch_decoded.size())
            # if torch.is_tensor(output_batch): print(output_batch.size())
            # print(ark)

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

                assert outputs.size() == targets.size(
                ), "ERROR: Latent output of network ({:}) does not match with target ({:}).".format(
                    outputs.size(), targets.size())

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
                # self.print("loss_batch.backward()")
                loss_batch.backward()
                # self.print("self.optimizer.step()")
                self.optimizer.step()
                # if self.optimizer_str == "sgd": self.scheduler.step()

            if self.multinode:
                loss_batch = self.multinodeMetric(loss_batch, "loss_batch")
                if self.output_forecasting_loss:
                    loss_fwd = self.multinodeMetric(loss_fwd, "loss_fwd")
                if self.latent_forecasting_loss:
                    loss_dyn_fwd = self.multinodeMetric(
                        loss_dyn_fwd, "loss_dyn_fwd")
                if self.reconstruction_loss:
                    loss_auto_fwd = self.multinodeMetric(
                        loss_auto_fwd, "loss_auto_fwd")
                if self.beta_vae:
                    loss_kl = self.multinodeMetric(loss_kl, "loss_kl")
                if self.c1_latent_smoothness_loss:
                    loss_auto_fwd_c1 = self.multinodeMetric(
                        loss_auto_fwd_c1, "loss_auto_fwd_c1")

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

        # self.print("np.mean(np.array(losses_vec), axis=0)")
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
        # if self.gpu and self.is_master: self.print(utils.getGpuMemoryMapString(self.smi_handle))
        epoch_losses_vec = []
        # print("# trainEpoch() #")
        for batch_of_sequences in data_loader:
            # K, T, C, Dx, Dy
            losses, iterative_forecasting_prob, beta_vae_weight = self.trainOnBatch(
                batch_of_sequences, is_train=is_train, dataset=dataset)
            epoch_losses_vec.append(losses)
        epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)
        time_ = time.time() - self.start_time
        """ Updating the learning rate scheduler """
        if (self.scheduler is not None) and (is_train == False):
            self.scheduler.step(epoch_losses[0])
        return epoch_losses, iterative_forecasting_prob, time_, beta_vae_weight

    def train(self):
        self.print("[crnn] # train() #")
        if self.is_master:
            if self.gpu and self.params["gpu_monitor_every"]:
                self.gpu_monitor_process = utils.GPUMonitor(
                    self.params["gpu_monitor_every"], self.multinode,
                    self.rank_str)
        """ The master rank has initalized the model parameters. Broadcast here before training """
        if self.multinode: self.broadcastParameters()

        data_loader_train, sampler_train, dataset_train = utils.getDataLoader(
            self.data_path_train,
            self.data_info_dict,
            self.batch_size,
            shuffle=True,
            print_=self.is_master,
            rank_str=self.rank_str,
            gpu=self.gpu,
            multinode=self.multinode,
            rank=self.rank,
            world_size=self.world_size,
        )

        data_loader_val, _, dataset_val = utils.getDataLoader(
            self.data_path_val,
            self.data_info_dict,
            self.batch_size,
            shuffle=False,
            print_=self.is_master,
            rank_str=self.rank_str,
            gpu=self.gpu,
            multinode=self.multinode,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Before starting training, scale the latent space
        if self.model.has_latent_scaler:
            self.loadAutoencoderLatentStateLimits()

        self.sampler_train = sampler_train

        self.declareOptimizer(self.learning_rate)
        # In case of multinode, distribute the optimizer
        if self.multinode: self.broadcastOptimizerState()
        self.declareScheduler()

        # Check if retraining
        if self.retrain == 1:

            if self.is_master:
                self.print("[crnn] RESTORING pytorch model")
                self.load()

            if self.multinode: self.broadcastParameters()

        elif self.train_RNN_only == 1:

            if self.is_master:
                self.print("[crnn] LOADING autoencoder model: \n")
                self.loadAutoencoderModel()
                # Saving the initial state
                self.print("[crnn] Saving the initial model")
                torch.save(self.getModel().state_dict(),
                           self.saving_model_path)

            if self.multinode: self.broadcastParameters()

        elif self.load_trained_AE == 1:
            if self.is_master:
                self.print("[crnn] LOADING autoencoder model: \n")
                self.loadAutoencoderModel()
                # Saving the initial state
                self.print("[crnn] Saving the initial model")
                torch.save(self.getModel().state_dict(),
                           self.saving_model_path)

            if self.multinode: self.broadcastParameters()

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
        if self.is_master: self.tqdm = tqdm(total=self.max_epochs)
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
                self.print(
                    "[crnn] Training finished. Maximum number of epochs reached."
                )
            elif (not self.with_scheduler
                  ) and self.rounds_iter == self.max_rounds:
                self.print(
                    "[crnn] Training finished. Maximum number of rounds reached."
                )
            elif (self.with_scheduler) and (self.scheduler_updates
                                            == self.max_scheduler_rounds):
                self.print(
                    "[crnn] Training finished. Maximum number of scheduler LR updates (rounds) reached."
                )
            else:
                if self.is_master:
                    if self.gpu:
                        self.gpu_monitor_process.stop()
                # print(self.rounds_iter)
                # print(self.epochs_iter)
                raise ValueError(
                    "[crnn] Training finished in round {:}, after {:} total epochs. I do not know why!"
                    .format(self.rounds_iter, self.epochs_iter))

            if self.is_master:
                self.saveModel()
                if self.params["plotting"]:
                    utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                             self.loss_total_val_vec,
                                             self.min_val_total_loss)

                    utils.plotAllLosses(self, self.losses_train_vec,
                                        self.losses_time_train_vec,
                                        self.losses_val_vec,
                                        self.losses_time_val_vec,
                                        self.min_val_total_loss)
                    utils.plotScheduleLoss(self, self.ifp_train_vec,
                                           self.ifp_val_vec)
                    utils.plotScheduleLearningRate(self,
                                                   self.learning_rate_vec)
                    if self.beta_vae:
                        utils.plotScheduleKLLoss(self,
                                                 self.beta_vae_weight_vec)

        if self.is_master:
            if self.gpu and self.params["gpu_monitor_every"]:
                self.gpu_monitor_process.stop()

    def printLosses(self, label, losses):
        self.losses_labels = [
            "TOTAL", "FWD", "DYN-FWD", "AUTO-REC", "KL", "C1"
        ]
        idx = np.nonzero(losses)[0]
        to_print = "[crnn] # {:s}-losses: ".format(label)
        for i in range(len(idx)):
            to_print += "{:}={:1.2E} |".format(self.losses_labels[idx[i]],
                                               losses[idx[i]])
        self.print(to_print)

    def printEpochStats(self, epoch_time_start, epochs_iter, epochs_in_round,
                        losses_train, losses_val):
        epoch_duration = time.time() - epoch_time_start
        time_covered = epoch_duration * epochs_iter
        time_total = epoch_duration * self.max_epochs
        percent = time_covered / time_total * 100
        label = "[crnn] EP={:} - R={:} - SR={:} - ER={:} - [ TIME= {:}, {:} / {:} - {:.2f} %] - LR={:1.2E}".format(
            epochs_iter, self.rounds_iter, self.scheduler_updates,
            epochs_in_round, utils.secondsToTimeStr(epoch_duration),
            utils.secondsToTimeStr(time_covered),
            utils.secondsToTimeStr(time_total), percent,
            self.learning_rate_round)

        size_of_print = len(label)
        self.print("[crnn] " + "-" * (size_of_print - len("[crnn] ") + 2))
        self.print(label)
        self.printLosses("TRAIN", losses_train)
        self.printLosses("VAL  ", losses_val)

    def printLearningRate(self):
        for param_group in self.optimizer.param_groups:
            self.print("[crnn] Current learning rate = {:}".format(
                param_group["lr"]))
        return 0

    def getModel(self):
        return self.model

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
            self.learning_rate_round = self.learning_rate_round * self.lr_reduction_factor

        if self.rounds_iter > 0:
            # """ Optimizer has to be re-declared """
            # self.declareOptimizer(self.learning_rate_round)
            # if self.multinode:  self.broadcastOptimizerState()

            if self.is_master:
                """ Restore the model """
                self.print("[crnn] RESTORING pytorch model")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path))

            if self.multinode: self.broadcastParameters()

            del self.optimizer
            """ Optimizer has to be re-declared """
            self.declareOptimizer(self.learning_rate_round)
            if self.multinode:
                self.broadcastOptimizerState()

        else:
            if self.is_master:
                self.print("[crnn] Saving the initial model")
                torch.save(self.getModel().state_dict(),
                           self.saving_model_path)

        if self.is_master:
            self.print("[crnn] ### Round: {:}, Learning rate={:} ###".format(
                self.rounds_iter, self.learning_rate_round))

        # Setting the epoch in the multinode case (used in the sampler seed) to ensure consistent sampling among modes
        if self.multinode and self.sampler_train is not None:
            self.sampler_train.set_epoch(0)

        losses_train, ifp_train, time_train, beta_vae_weight = self.trainEpoch(
            data_loader_train, is_train=False, dataset=dataset_train)
        if self.iterative_loss_validation: assert (ifp_train == 1.0)

        losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(
            data_loader_val, is_train=False, dataset=dataset_val)
        if self.iterative_loss_validation: assert (ifp_val == 1.0)

        if self.is_master:
            label = "[crnn] INITIAL (NEW ROUND):  EP{:} - R{:}".format(
                self.epochs_iter, self.rounds_iter)
            self.print(label)
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

        self.scheduler_updates = 0
        for epochs_iter in range(self.epochs_iter, self.max_epochs + 1):
            epoch_time_start = time.time()
            epochs_in_round = epochs_iter - self.epochs_iter
            self.epochs_iter_global = epochs_iter

            # In multinode case, use a single random shuffling of the training data (per epoch) that is divided amongst all k workers. This is achieved by setting the epoch in the sampler (random seed)
            if self.multinode and self.sampler_train is not None:
                self.sampler_train.set_epoch(epochs_iter + 1)
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

            if self.scheduler is not None:
                # self.print("self.optimizer.param_groups = {:}".format(self.optimizer.param_groups))
                # self.print("self.optimizer.param_groups[0]['lr'] = {:}".format(self.optimizer.param_groups[0]['lr']))
                self.learning_rate_round_prev = self.learning_rate_round
                self.learning_rate_round = self.optimizer.param_groups[0]['lr']
                if self.learning_rate_round < self.learning_rate_round_prev:
                    self.scheduler_updates += 1

            self.learning_rate_vec.append(self.learning_rate_round)

            if self.is_master:
                self.printEpochStats(epoch_time_start, epochs_iter,
                                     epochs_in_round, losses_train, losses_val)

            if self.is_master:
                if losses_val[0] < self.min_val_total_loss:
                    self.print("[crnn] Saving model !")
                    self.min_val_total_loss = losses_val[0]
                    self.loss_total_train = losses_train[0]
                    if self.is_master:
                        torch.save(self.getModel().state_dict(),
                                   self.saving_model_path)

            if self.scheduler is None:
                if epochs_in_round > self.overfitting_patience:
                    if all(self.min_val_total_loss < np.array(
                            RNN_loss_round_val_vec[-self.overfitting_patience:]
                    )):
                        self.previous_round_converged = True
                        break
            elif self.scheduler_updates == self.max_scheduler_rounds:
                break

            # # LEARNING RATE SCHEDULER (PLATEU ON VALIDATION LOSS)
            # if self.optimizer_str == "adam": self.scheduler.step(losses_val[0])
            if self.is_master: self.tqdm.update(1)
            isWallTimeLimit = self.isWallTimeLimit()
            if isWallTimeLimit:
                break

        self.rounds_iter += 1
        self.epochs_iter = epochs_iter
        return isWallTimeLimit

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            if self.is_master:
                self.print(
                    "[crnn] ### Maximum train time reached: saving model... ###"
                )
            if self.is_master:
                self.tqdm.close()
                self.saveModel()
                if self.params["plotting"]:
                    utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                             self.loss_total_val_vec,
                                             self.min_val_total_loss)
                    utils.plotAllLosses(self, self.losses_train_vec,
                                        self.losses_time_train_vec,
                                        self.losses_val_vec,
                                        self.losses_time_val_vec,
                                        self.min_val_total_loss)
                    utils.plotScheduleLoss(self, self.ifp_train_vec,
                                           self.ifp_val_vec)
                    utils.plotScheduleLearningRate(self,
                                                   self.learning_rate_vec)
            return True
        else:
            return False

    def delete(self):
        pass

    def saveModel(self):
        if self.is_master:
            self.print("[crnn] Recording time...")
            self.total_training_time = time.time() - self.start_time
            if hasattr(self, 'loss_total_train_vec'):
                if len(self.loss_total_train_vec) != 0:
                    self.training_time = self.total_training_time / len(
                        self.loss_total_train_vec)
                else:
                    self.training_time = self.total_training_time
            else:
                self.training_time = self.total_training_time

            self.print("[crnn] Total training time per epoch is {:}".format(
                utils.secondsToTimeStr(self.training_time)))
            self.print("[crnn] Total training time is {:}".format(
                utils.secondsToTimeStr(self.total_training_time)))

            self.memory = utils.getMemory()
            self.print("[crnn] Script used {:} MB".format(self.memory))

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
                self.print("[crnn] Writing to log-file in path {:}".format(
                    logfile_train))
                utils.writeToLogFile(self, logfile_train, data,
                                     fields_to_write)

            data_path = utils.getModelDir(self) + "/data"
            utils.saveData(data, data_path, "pickle")

    def load(self, in_cpu=False):
        try:
            if not in_cpu and self.gpu:
                self.print("[crnn] # LOADING model in GPU.")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path))

            else:
                self.print("[crnn] # LOADING model in CPU...")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path,
                               map_location=torch.device('cpu')))

        except Exception as inst:
            self.print(
                "[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the model?"
                .format(self.saving_model_path))
            raise ValueError(inst)

        self.print("[crnn] # Model loaded successfully!")

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
            self.print(
                "[Error (soft)] Model {:s} found. The data from training (result, losses, etc.), however, is missing."
                .format(self.saving_model_path))
            self.retrain_model_data_found = False
        return 0

    def loadAutoencoderModel(self, in_cpu=False):
        model_name_autoencoder = self.getAutoencoderName()
        print("[crnn] Loading autoencoder with name:")
        print(model_name_autoencoder)
        AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model"
        # self.getModel().load_state_dict(torch.load(AE_path), strict=False)
        try:
            if not in_cpu and self.gpu:
                print("[crnn] # LOADING autoencoder model in GPU.")
                self.getModel().load_state_dict(torch.load(AE_path),
                                                strict=False)
            else:
                print("[crnn] # LOADING autoencoder model in CPU...")
                self.getModel().load_state_dict(torch.load(
                    AE_path, map_location=torch.device('cpu')),
                                                strict=False)
        except Exception as inst:
            self.print(
                "[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the autoencoder ? If you run on a cluster, is the GPU detected ? Did you use the srun command ?"
                .format(AE_path))
            raise ValueError(inst)
        AE_data_path = self.saving_path + self.model_dir + model_name_autoencoder + "/data"
        # data = utils.loadData(AE_data_path, "pickle")
        # del data
        return 0

    def computeLatentStateInfo(self, latent_states_all):
        #########################################################
        # In case of plain CNN (no MLP between encoder-decoder):
        # shape either  (n_ics, T, latent_state, 1, 1)
        # shape or      (n_ics, T, 1, 1, latent_state)
        #########################################################
        # In case of CNN-MLP (encoder-MLP-latent_space-decoder):
        # Shape (n_ics, T, latent_state)
        assert len(np.shape(
            latent_states_all)) == 3, "np.shape(latent_states_all)={:}".format(
                np.shape(latent_states_all))
        latent_states_all = np.reshape(latent_states_all,
                                       (-1, self.params["latent_state_dim"]))
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

    def loadAutoencoderLatentStateLimits(self):
        print("[crnn] loadAutoencoderLatentStateLimits()")
        model_name_autoencoder = self.getAutoencoderName()
        AE_results_testing_path = self.saving_path + self.results_dir + model_name_autoencoder + "/results_autoencoder_testing_val"
        try:
            data = utils.loadData(AE_results_testing_path, "pickle")
        except Exception as inst:
            self.print(
                "[Error] AE testing results file:\n{:}\nNOT FOUND. Result file from AE testing needed to load the bounds of the latent state."
                .format(AE_results_testing_path))
            raise ValueError(inst)

        if "latent_state_info" in data.keys():
            print("[crnn] latent bounds found in AE testing file.")
            latent_state_info = data["latent_state_info"]
        else:
            print(
                "[crnn] latent bounds not found in AE testing file. Computing them..."
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

    def getRNNTestingModes(self):
        modes = []
        if self.params["iterative_state_forecasting"]:
            modes.append("iterative_state_forecasting")
        if self.params["iterative_latent_forecasting"]:
            modes.append("iterative_latent_forecasting")
        if self.params["teacher_forcing_forecasting"]:
            modes.append("teacher_forcing_forecasting")
        return modes

    def test(self):
        if self.is_master:
            if self.gpu and self.params["gpu_monitor_every"]:
                self.gpu_monitor_process = utils.GPUMonitor(
                    self.params["gpu_monitor_every"], self.multinode,
                    self.rank_str)
            self.testMaster()
            if self.gpu and self.params["gpu_monitor_every"]:
                self.gpu_monitor_process.stop()
        else:
            return 0

    def testMaster(self):
        if self.load() == 0:
            # MODEL LOADED IN EVALUATION MODE
            with torch.no_grad():
                test_on = []

                if self.has_rnn:
                    self.n_warmup = self.params["n_warmup"]
                    assert self.n_warmup > 0
                    print("[crnn] Warming-up steps: {:d}".format(
                        self.n_warmup))
                    testing_modes = self.getRNNTestingModes()
                    print("[crnn] WARMING UP STEPS (for statefull RNNs): {:d}".
                          format(self.n_warmup))
                else:
                    testing_modes = self.getAutoencoderTestingModes()

                test_on = []
                if self.params["test_on_test"]: test_on.append("test")
                if self.params["test_on_val"]: test_on.append("val")
                if self.params["test_on_train"]: test_on.append("train")
                for set_ in test_on:
                    common_testing.testModesOnSet(self,
                                                  set_=set_,
                                                  testing_modes=testing_modes)
        return 0

    def forward(self,
                input_sequence,
                init_hidden_state,
                input_is_latent=False,
                iterative_propagation_is_latent=False):
        input_sequence = utils.transform2Tensor(self, input_sequence)
        init_hidden_state = utils.transform2Tensor(self, init_hidden_state)
        outputs, next_hidden_state, latent_states, latent_states_pred, _, _, time_latent_prop, _, _ = self.model.forward(
            input_sequence,
            init_hidden_state,
            is_train=False,
            is_iterative_forecasting=False,
            iterative_forecasting_prob=0,
            iterative_forecasting_gradient=0,
            horizon=None,
            input_is_latent=input_is_latent,
            iterative_propagation_is_latent=iterative_propagation_is_latent,
        )
        outputs = outputs.detach().cpu().numpy()
        latent_states_pred = latent_states_pred.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        return outputs, next_hidden_state, latent_states, latent_states_pred, time_latent_prop

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

    def encodeDecode(self, input_sequence):
        input_sequence = utils.transform2Tensor(self, input_sequence)
        initial_hidden_states = self.getInitialRNNHiddenState(
            len(input_sequence))

        _, _, latent_states, _, _, input_decoded, _, _, _ = self.model.forward(
            input_sequence, initial_hidden_states, is_train=False)
        input_decoded = input_decoded.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        return input_decoded, latent_states

    def getAutoencoderTestingModes(self):
        return [
            "autoencoder_testing",
        ]

    def getTestingModes(self):
        modes = self.getAutoencoderTestingModes() + self.getRNNTestingModes()
        return modes

    def predictSequence(self,
                        input_sequence,
                        testing_mode=None,
                        dt=1,
                        prediction_horizon=None):
        self.print("[crnn] # predictSequence() #")
        self.print("[crnn] {:}:".format(np.shape(input_sequence)))
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.n_warmup != prediction_horizon:
            raise ValueError(
                "[crnn] Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})"
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

        if testing_mode in self.getRNNTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup +
                                    prediction_horizon]
        else:
            raise ValueError(
                "[crnn] Testing mode {:} not recognized.".format(testing_mode))

        warmup_data_input = utils.transform2Tensor(self, warmup_data_input)
        initial_hidden_states = utils.transform2Tensor(self,
                                                       initial_hidden_states)
        # if self.gpu: initial_hidden_states = self.sendHiddenStateToGPU(initial_hidden_states)

        # print(initial_hidden_states)
        # print(warmup_data_input[-1])

        if self.n_warmup > 1:
            warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _, _, _, _, _ = self.model.forward(
                warmup_data_input, initial_hidden_states, is_train=False)
        else:
            # In case of predictor with n_warmup=1 (no warmup)
            # assert(self.has_predictor)
            last_hidden_state = initial_hidden_states

        # print(latent_states_pred)

        prediction = []

        if ("iterative_latent" in testing_mode):
            iterative_propagation_is_latent = 1
            # GETTING THE LAST LATENT STATE (K, T, LD)
            # In iterative latent forecasting, the input is the latent state
            input_latent = latent_states_pred[:, -1, :]
            input_latent.unsqueeze_(0)
            input_t = input_latent
        elif ("iterative_state" in testing_mode):
            iterative_propagation_is_latent = 0
            # LATTENT PROPAGATION
            input_t = input_sequence[self.n_warmup - 1]
            input_t = input_t[np.newaxis, np.newaxis, :]
        elif "teacher_forcing" in testing_mode:
            iterative_propagation_is_latent = 0
            input_t = input_sequence[self.n_warmup - 1:-1]
            input_t = input_t.cpu().detach().numpy()
            input_t = input_t[np.newaxis]
        else:
            raise ValueError(
                "[crnn] I do not know how to initialize the state for {:}.".
                format(testing_mode))

        input_t = utils.transform2Tensor(self, input_t)
        last_hidden_state = utils.transform2Tensor(self, last_hidden_state)

        time_start = time.time()
        if "teacher_forcing" in testing_mode:
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
        elif "iterative_state" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
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
                "[crnn] Testing mode {:} not recognized.".format(testing_mode))
        time_end = time.time()
        time_total = time_end - time_start

        # Correcting the time-measurement in case of evolution of the original system (in this case, we do not need to internally propagate the latent space of the RNN)
        time_total = time_latent_prop

        time_total_per_iter = time_total / prediction_horizon

        prediction = prediction[0]
        if self.has_rnn: RNN_outputs = RNN_outputs[0]
        latent_states = latent_states[0]

        target = target.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        latent_states = latent_states.cpu().detach().numpy()
        if self.has_rnn: RNN_outputs = RNN_outputs.cpu().detach().numpy()

        target = np.array(target)
        prediction = np.array(prediction)
        latent_states = np.array(latent_states)
        if self.has_rnn: RNN_outputs = np.array(RNN_outputs)

        self.print("[crnn] Shapes of prediction/target/latent_states:")
        self.print("[crnn] {:}".format(np.shape(prediction)))
        self.print("[crnn] {:}".format(np.shape(target)))
        self.print("[crnn] {:}".format(np.shape(latent_states)))

        # print("Min/Max")
        # print("Target:")
        # print(np.max(target[:,0]))
        # print(np.min(target[:,0]))
        # print("Prediction:")
        # print(np.max(prediction[:,0]))
        # print(np.min(prediction[:,0]))

        if self.n_warmup > 1:
            warmup_data_target = warmup_data_target.cpu().detach().numpy()
            warmup_data_output = warmup_data_output.cpu().detach().numpy()
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

        return prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter

    def plotTraining(self):
        self.print("[crnn] # plotTraining() #")
        if self.is_master:
            if self.load() == 0:
                if self.params["plotting"]:
                    utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                             self.loss_total_val_vec,
                                             self.min_val_total_loss)
                    utils.plotAllLosses(self, self.losses_train_vec,
                                        self.losses_time_train_vec,
                                        self.losses_val_vec,
                                        self.losses_time_val_vec,
                                        self.min_val_total_loss)
        else:
            return 0

    def plot(self):
        if self.is_master:
            if self.has_rnn:
                testing_modes = self.getRNNTestingModes()

            elif self.has_autoencoder:
                testing_modes = self.getAutoencoderTestingModes()

            if self.write_to_log:
                for testing_mode in testing_modes:
                    common_plot.writeLogfiles(self, testing_mode=testing_mode)
            else:
                print("[crnn] # write_to_log=0. #")

            if self.params["plotting"]:
                for testing_mode in testing_modes:
                    common_plot.plot(self, testing_mode=testing_mode)
            else:
                print("[crnn] # plotting=0. No plotting. #")
        else:
            return 0

    def encode(self, inputs):
        # print("encode")
        # print(np.shape(inputs))
        with torch.no_grad():
            inputs = utils.transform2Tensor(self, inputs)
            latent = self.model.forwardEncoder(inputs)
            latent = latent.detach().cpu().numpy()
        # print(np.shape(latent))
        return latent

    def decode(self, latent):
        # print("decode")
        # print(np.shape(latent))
        with torch.no_grad():
            latent = utils.transform2Tensor(self, latent)
            inputs_decoded = self.model.forwardDecoder(latent)
            inputs_decoded = inputs_decoded.detach().cpu().numpy()
        return inputs_decoded









    def debug(self):
        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        self.model_name = "GPU-" + self.model_name
        print(self.model_name)
        # print(self.getMultiscaleTestingModes())

        for set_name in plot_on:

            for testing_mode in self.getRNNTestingModes():

                # Loading the results
                data_path = utils.getResultsDir(
                    self) + "/results_{:}_{:}".format(
                        testing_mode, set_name)
                results = utils.loadData(data_path, self.save_format)

                predictions_all = results["predictions_all"]
                targets_all = results["targets_all"]

                # print(np.shape(predictions_all))
                # print(np.shape(targets_all))

                error_dict = utils.getErrorLabelsDict(self)

                num_ics = len(targets_all)
                T = len(targets_all[0])
                tqdm_bar = tqdm(total=num_ics * T)

                # print(ark)
                for ic_num in range(num_ics):
                    predictions_all_ic = predictions_all[ic_num]
                    targets_all_ic = targets_all[ic_num]

                    error_dict_ic = utils.getErrorLabelsDict(self)

                    T = len(targets_all_ic)
                    for t in range(T):
                        target_save_path = targets_all_ic[t]
                        prediction_save_path = predictions_all_ic[t]


                        target_save_path = np.array(target_save_path)
                        target_save = utils.getDataHDF5Fields(target_save_path[0], [target_save_path[1]])


                        prediction_save_path = np.array(prediction_save_path)
                        prediction_save = utils.getDataHDF5Fields(prediction_save_path[0], [prediction_save_path[1]])

                        prediction_save = prediction_save[0]
                        target_save = target_save[0]

                        # print(np.shape(prediction_save))
                        # print(np.shape(target_save))
                        errors = utils.computeErrors(
                            target_save,
                            prediction_save,
                            self.data_info_dict,
                            single_sample=True)

                        # Updating the error
                        for error in errors:
                            error_dict_ic[error].append(errors[error])

                        tqdm_bar.update(1)

                    # Updating the error
                    for error in error_dict.keys():
                        error_dict[error].append(error_dict_ic[error])


                # Computing the average over time
                error_dict_avg = {}
                for key in error_dict:
                    error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
                utils.printErrors(error_dict_avg)

                results = {
                    **results,
                    **error_dict,
                    **error_dict_avg,
                }
                utils.saveData(results, data_path, self.save_format)

                for key in results: print(key)
