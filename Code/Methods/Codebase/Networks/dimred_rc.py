#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
""" Libraries """
from tqdm import tqdm
import numpy as np
import time
import os
""" Auxiliary libraries for regression """
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from sklearn.linear_model import Ridge
""" Utilities """
from .. import Utils as utils
from .. import Systems as systems
""" Networks """
from . import crnn_model
from . import dimred
""" Printing """
from functools import partial

print = partial(print, flush=True)
import warnings
""" Common libraries """
from . import common_testing
from . import common_plot

# """ Torch """
# import torch
# import sys
# from torch.autograd import Variable


class dimred_rc():
    def __init__(self, params):
        super(dimred_rc, self).__init__()
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

        ##################################################################
        # RANDOM SEEDING
        ##################################################################
        self.random_seed = params["random_seed"]

        # Setting the random seed
        np.random.seed(self.random_seed)

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

        self.input_dim = params['input_dim']

        self.channels = params['channels']
        self.Dz, self.Dy, self.Dx = utils.getChannels(self.channels, params)

        ##################################################################
        # SCALER
        ##################################################################
        self.scaler = params["scaler"]

        self.latent_state_dim = params["latent_state_dim"]

        ##################################################################
        # RC parameters
        ##################################################################
        self.rc_solver = params["rc_solver"]
        self.rc_approx_reservoir_size = params["rc_approx_reservoir_size"]
        self.rc_degree = params["rc_degree"]
        self.rc_radius = params["rc_radius"]
        self.rc_sigma_input = params["rc_sigma_input"]
        self.rc_dynamics_length = params["rc_dynamics_length"]
        self.rc_regularization = params["rc_regularization"]
        self.rc_noise_level_per_mill = params["rc_noise_level_per_mill"]

        self.gpu = 0
        ##################################################################
        # Encoding model (either CNN/PCA/etc.)
        ##################################################################

        if self.params["dimred_method"] == "ae":
            from . import crnn
            self.has_autoencoder = 1
            """ Setting dummy parameters """
            params_autoencoder = self.params.copy()
            params_autoencoder["learning_rate"] = self.params[
                "learning_rate_AE"]
            params_autoencoder["random_seed"] = self.params[
                "random_seed_in_AE_name"]
            """ Make sure no autoencoder training """
            params_autoencoder["reconstruction_loss"] = 0
            params_autoencoder["output_forecasting_loss"] = 0
            params_autoencoder["latent_forecasting_loss"] = 0
            assert params_autoencoder["latent_state_dim"] > 0
            self.model_autoencoder = crnn.crnn(params_autoencoder)
            self.model_autoencoder.model.printModuleList()
            # self.loadAutoencoderModel()
            self.gpu = self.model_autoencoder.gpu

        elif self.params["dimred_method"] == "pca":
            from . import dimred
            self.has_pca = 1
            self.model_autoencoder = dimred.dimred(self.params)
            self.model_autoencoder.load()

        elif self.params["dimred_method"] == "diffmaps":
            from . import dimred
            self.has_pca = 1
            self.model_autoencoder = dimred.dimred(self.params)
            self.model_autoencoder.load()
        else:
            raise ValueError("Invalid autoencoding method.")
        """ Model name """
        self.model_name = self.createModelName()

        print("[dimred_rc] - model_name:")
        print("[dimred_rc] {:}".format(self.model_name))
        self.saving_model_path = utils.getModelDir(self) + "/model"

        utils.makeDirectories(self)
        """ Saving some info file for the model """
        data = {"params": params, "name": self.model_name}
        data_path = utils.getModelDir(self) + "/info"
        utils.saveData(data, data_path, self.params["save_format"])
        self.data_info_dict = systems.getSystemDataInfo(self)

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

    def createModelName(self, with_residual_autoencoder=False):
        # print("# createModelName() #")
        keys = self.getKeysInModelName()
        # str_gpu = "GPU-" * self.gpu
        str_gpu = ""
        str_ = "RC-" + self.getAutoencoderName()
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        if with_residual_autoencoder:
            raise ValueError("TODO:")
        return str_

    def getAutoencoderName(self):
        return self.model_autoencoder.model_name

    def getSparseWeights(self, sizex, sizey, radius, sparsity):
        print("[dimred_rc] Weight initialization.")
        W = sparse.random(sizex,
                          sizey,
                          density=sparsity,
                          random_state=self.random_seed)
        print("[dimred_rc] Eigenvalue decomposition.")
        eigenvalues, eigvectors = splinalg.eigs(W)
        eigenvalues = np.abs(eigenvalues)
        W = (W / np.max(eigenvalues)) * radius
        return W

    def augmentHidden(self, h):
        h_aug = h.copy()
        # h_aug = pow(h_aug, 2.0)
        # h_aug = np.concatenate((h,h_aug), axis=0)
        h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def getAugmentedStateSize(self):
        return self.rc_reservoir_size

    def train(self):
        print("[dimred_rc] # train() #")
        """ RC trained on raw data """
        data_path_train = self.data_path_train + "_raw"
        if os.path.isdir(data_path_train):
            print(
                "[dimred_rc] Raw data directory {:} found. Using the raw (unbatched) data to train the Reservoir Computer."
                .format(data_path_train))
        else:
            print(
                "[dimred_rc] Raw data directory {:} not found. Using the batched data."
                .format(data_path_train))
            data_path_train = self.data_path_train

        batch_size = 1
        data_loader_train, sampler_train, data_set_train = utils.getDataLoader(
            data_path_train,
            self.data_info_dict,
            batch_size,
            shuffle=False,
        )

        print("[dimred_rc] Forward encoder and get latent states.")
        time_start = time.time()
        latent_states = []

        for sequence in data_loader_train:
            sequence = utils.getDataBatch(self,
                                          sequence,
                                          0,
                                          -1,
                                          dataset=data_set_train)
            """ Encode """
            latent_state = self.model_autoencoder.encode(sequence)
            # latent_state = latent_state.detach().cpu().numpy()
            latent_states.append(latent_state)

        latent_states = np.array(latent_states)
        latent_states = np.reshape(latent_states,
                                   (-1, *np.shape(latent_states)[2:]))
        latent_states = np.array(latent_states)

        n_ics, T, D = np.shape(latent_states)

        total_time = time.time() - time_start
        print("[dimred_rc] Total encoder propagation time is {:}".format(
            utils.secondsToTimeStr(total_time)))

        train_input_sequences = latent_states
        print("[dimred_rc] training data size = {:}".format(
            np.shape(train_input_sequences)))

        num_seq, N, input_dim = np.shape(train_input_sequences)

        print("[dimred_rc] Initializing the reservoir weights...")
        nodes_per_input = int(
            np.ceil(self.rc_approx_reservoir_size / input_dim))
        self.rc_reservoir_size = int(input_dim * nodes_per_input)
        self.rc_sparsity = self.rc_degree / self.rc_reservoir_size
        print("[dimred_rc] Network sparsity {:}".format(self.rc_sparsity))
        print("[dimred_rc] Computing sparse hidden to hidden weight matrix...")
        W_h = self.getSparseWeights(self.rc_reservoir_size,
                                    self.rc_reservoir_size, self.rc_radius,
                                    self.rc_sparsity)

        # Initializing the input weights
        print("[dimred_rc] Initializing the input weights...")
        W_in = np.zeros((self.rc_reservoir_size, input_dim))
        q = int(self.rc_reservoir_size / input_dim)
        for i in range(0, input_dim):
            W_in[i * q:(i + 1) * q,
                 i] = self.rc_sigma_input * (-1 + 2 * np.random.rand(q))

        # Training length per sequence
        tl = N - self.rc_dynamics_length
        if self.rc_dynamics_length >= N:
            raise ValueError(
                "[dimred_rc] The rc_dynamics_length={:} cannot be larger than the total sequence length={:}."
                .format(self.rc_dynamics_length, N))

        h_initial = []
        for seq_num in range(num_seq):
            train_input_sequence = train_input_sequences[seq_num]
            print("[dimred_rc] Training: Dynamics prerun...")
            h = np.zeros((self.rc_reservoir_size, 1))
            for t in range(self.rc_dynamics_length):
                if self.display_output == True:
                    if (t % 100 == 0) and (seq_num % 10) == 0:
                        print(
                            "[dimred_rc] Training: Sequence {:}/{:}, {:2.3f}% - Dynamics prerun: T {:}/{:}, {:2.3f}%"
                            .format(seq_num, num_seq, seq_num / num_seq * 100,
                                    t, self.rc_dynamics_length,
                                    t / self.rc_dynamics_length * 100))
                i = np.reshape(train_input_sequence[t], (-1, 1))
                h = np.tanh(W_h @ h + W_in @ i)
                # H_dyn[t] = self.augmentHidden(h)
            h_initial.append(h)

        print("[dimred_rc] h_initial {:}".format(np.shape(h_initial)))

        if self.rc_solver == "pinv":
            NORMEVERY = 10
            HTH = np.zeros(
                (self.getAugmentedStateSize(), self.getAugmentedStateSize()))
            YTH = np.zeros((input_dim, self.getAugmentedStateSize()))
        H = []
        Y = []

        print("[dimred_rc] Training: Teacher forcing...")

        for seq_num in range(num_seq):
            # Get the sequence and the initial hidden state
            train_input_sequence = train_input_sequences[seq_num]
            h = h_initial[seq_num]
            for t in range(tl - 1):
                if self.display_output == True:
                    if (t % 100 == 0) and (seq_num % 10) == 0:
                        print(
                            "[dimred_rc] Training - Teacher forcing: seq {:}/{:}, T {:}/{:}, {:2.3f}%"
                            .format(seq_num, num_seq, t, tl,
                                    (seq_num * tl + t) / (tl * num_seq) * 100))
                i = np.reshape(
                    train_input_sequence[t + self.rc_dynamics_length], (-1, 1))
                h = np.tanh(W_h @ h + W_in @ i)
                # Augment the hidden state
                h_aug = self.augmentHidden(h)
                H.append(h_aug[:, 0])
                target = np.reshape(
                    train_input_sequence[t + self.rc_dynamics_length + 1],
                    (-1, 1))
                Y.append(target[:, 0])
                if self.rc_solver == "pinv" and (t % NORMEVERY == 0):
                    # Batched approach used in the pinv case
                    H = np.array(H)
                    Y = np.array(Y)
                    HTH += H.T @ H
                    YTH += Y.T @ H
                    H = []
                    Y = []

        if self.rc_solver == "pinv" and (len(H) != 0):
            # ADDING THE REMAINING BATCH
            H = np.array(H)
            Y = np.array(Y)
            HTH += H.T @ H
            YTH += Y.T @ H
            print("[dimred_rc] Teacher forcing ended.")
            print("[dimred_rc] H {:}".format(np.shape(H)))
            print("[dimred_rc] Y {:}".format(np.shape(Y)))
            print("[dimred_rc] HTH {:}".format(np.shape(HTH)))
            print("[dimred_rc] YTH {:}".format(np.shape(YTH)))
        else:
            print("[dimred_rc] Teacher forcing ended.")
            print("[dimred_rc] H {:}".format(np.shape(H)))
            print("[dimred_rc] Y {:}".format(np.shape(Y)))

        print("[dimred_rc] Solver used to find W_out: {:}.".format(
            self.rc_solver))

        if self.rc_solver == "pinv":
            """
            Learns mapping H -> Y with Penrose Pseudo-Inverse
            """
            I = np.identity(np.shape(HTH)[1])
            pinv_ = scipypinv2(HTH + self.rc_regularization * I)
            W_out = YTH @ pinv_

        elif self.rc_solver in [
                "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"
        ]:
            """
            Learns mapping H -> Y with Ridge Regression
            """
            ridge = Ridge(alpha=self.rc_regularization,
                          fit_intercept=False,
                          normalize=False,
                          copy_X=True,
                          solver=self.rc_solver)
            # print(np.shape(H))
            # print(np.shape(Y))
            # print("##")
            ridge.fit(H, Y)
            W_out = ridge.coef_

        else:
            raise ValueError("Undefined rc_solver={:}.".format(self.rc_solver))

        print("[dimred_rc] Finalizing weights.")
        self.W_in = W_in
        self.W_h = W_h
        self.W_out = W_out

        print("[dimred_rc] Computing number of parameters.")
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.W_in) + np.size(
            self.W_h) + np.size(self.W_out)
        print("[dimred_rc] Number of trainable parameters: {}".format(
            self.n_trainable_parameters))
        print("[dimred_rc] Total number of parameters: {}".format(
            self.n_model_parameters))
        print("[dimred_rc] Saving model.")
        self.save()

    def save(self):

        print("[dimred_rc] Recording time...")
        self.total_training_time = time.time() - self.start_time
        print("[dimred_rc] Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("[dimred_rc] Script used {:} MB".format(self.memory))

        data = {
            "rc_reservoir_size": self.rc_reservoir_size,
            "rc_sparsity": self.rc_sparsity,
            "W_out": self.W_out,
            "W_in": self.W_in,
            "W_h": self.W_h,
            "params": self.params,
            "model_name": self.model_name,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_model_parameters": self.n_model_parameters,
        }
        fields_to_write = [
            "memory",
            "total_training_time",
            "n_model_parameters",
            "n_trainable_parameters",
        ]
        if self.write_to_log == 1:
            logfile_train = utils.getLogFileDir(self) + "/train.txt"
            print("[dimred_rc] Writing to log-file in path {:}".format(
                logfile_train))
            utils.writeToLogFile(self, logfile_train, data, fields_to_write)
        data_path = utils.getModelDir(self) + "/data"
        utils.saveData(data, data_path, self.params["save_format"])

    def load(self):

        data_path = utils.getModelDir(self) + "/data"

        try:
            data = utils.loadData(data_path, self.params["save_format"])
            self.rc_reservoir_size = data["rc_reservoir_size"]
            self.rc_sparsity = data["rc_sparsity"]
            self.W_out = data["W_out"]
            self.W_in = data["W_in"]
            self.W_h = data["W_h"]
            self.n_trainable_parameters = data["n_trainable_parameters"]
            self.n_model_parameters = data["n_model_parameters"]
            del data

        except Exception as inst:
            print(
                "[Error (soft)] Model {:s} found. The data from training (result, losses, etc.), however, is missing."
                .format(self.saving_model_path))

        print("[dimred_rc] # Model loaded successfully!")
        return 0

    def loadAutoencoderModel(self, in_cpu=False):
        model_name_autoencoder = self.getAutoencoderName()
        print("[dimred_rc] Loading autoencoder with name:")
        print(model_name_autoencoder)
        AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model"
        self.model_autoencoder.model = utils.loadModel(
            AE_path, self.model_autoencoder.model, in_cpu=False, strict=False)
        return 0

    def getRCTestingModes(self):
        modes = []
        if self.params["iterative_latent_forecasting"]:
            modes.append("iterative_latent_forecasting")
        if self.params["teacher_forcing_forecasting"]:
            modes.append("teacher_forcing_forecasting")
        return modes

    def getTestingModes(self):
        return self.getRCTestingModes()

    def test(self):
        if self.load() == 0:
            testing_modes = self.getTestingModes()
            test_on = []
            self.n_warmup = self.params["n_warmup"]
            assert self.n_warmup > 0
            print("[dimred_rc] Warming-up steps: {:d}".format(self.n_warmup))

            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            for set_ in test_on:
                common_testing.testModesOnSet(self,
                                              set_=set_,
                                              testing_modes=testing_modes)
        return 0

    def sendHiddenStateToGPU(self, temp):
        return temp

    def warmup(self, input_seq, h):
        return self.forward(input_seq, h, input_is_latent=False)

    def forward(self,
                input_seq,
                h,
                input_is_latent=False,
                iterative_propagation_is_latent=None):
        shape_ = np.shape(input_seq)
        assert shape_[0] == 1
        input_seq = input_seq[0]
        h = h[0]

        time_latent_prop_t = 0.0

        latent_states = []
        latent_states_pred = []
        output = []
        for t in range(len(input_seq)):
            input_t = input_seq[t]

            if not input_is_latent:
                input_t = input_t[np.newaxis, np.newaxis]
                input_t = self.model_autoencoder.encode(input_t)
                input_t = input_t[0, 0]

            latent_states.append(input_t)

            time_start = time.time()

            # h_prev = h
            i = np.reshape(input_t, (-1, 1))
            h = np.tanh(self.W_h @ h + self.W_in @ i)
            out = self.W_out @ self.augmentHidden(h)
            out = out[:, 0]

            time_latent_prop_t += time.time() - time_start

            latent_states_pred.append(out)
            dec_inp = out[np.newaxis, np.newaxis]
            output_t = self.model_autoencoder.decode(dec_inp)[0, 0]
            output.append(output_t)
        """ Adjusting back the output """
        output = np.array(output)
        output = output[np.newaxis]

        h = h[np.newaxis]
        # h_prev = h_prev[np.newaxis]

        latent_states = np.array(latent_states)
        latent_states = latent_states[np.newaxis]

        latent_states_pred = np.array(latent_states_pred)
        latent_states_pred = latent_states_pred[np.newaxis]
        return output, h, latent_states, latent_states_pred, time_latent_prop_t

    def forecast(self, input_t, h, horizon):
        # print("# forecast() #")

        assert np.shape(input_t)[0] == 1
        assert np.shape(h)[0] == 1

        input_t = input_t[0]
        h = h[0]

        time_latent_prop_t = 0.0

        latent_states = []
        latent_states_pred = []
        output = []
        for t in range(horizon):

            time_start = time.time()

            assert np.shape(input_t)[0] == 1
            latent_states.append(input_t[0])

            i = np.reshape(input_t, (-1, 1))
            h = np.tanh(self.W_h @ h + self.W_in @ i)
            out = self.W_out @ self.augmentHidden(h)
            out = out[:, 0]

            latent_states_pred.append(out)

            time_latent_prop_t += time.time() - time_start

            input_t = out[np.newaxis]

            dec_inp = out[np.newaxis, np.newaxis]
            output_t = self.model_autoencoder.decode(dec_inp)[0, 0]

            output.append(output_t)
        """ Adjusting back the output """
        output = np.array(output)
        output = output[np.newaxis]

        h = h[np.newaxis]

        latent_states = np.array(latent_states)
        latent_states = latent_states[np.newaxis]

        latent_states_pred = np.array(latent_states_pred)
        latent_states_pred = latent_states_pred[np.newaxis]

        return output, h, latent_states, latent_states_pred, time_latent_prop_t

    def getInitialRNNHiddenState(self, k):
        h = np.zeros((k, self.rc_reservoir_size, 1))
        return h

    def predictSequence(self,
                        input_sequence,
                        testing_mode=None,
                        dt=1,
                        prediction_horizon=None):
        print("[dimred_rc] # predictSequence() #")
        print("[dimred_rc] {:}:".format(np.shape(input_sequence)))
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.n_warmup != prediction_horizon:
            raise ValueError(
                "[dimred_rc] Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})"
                .format(N, self.n_warmup, prediction_horizon))

        warmup_data_input = input_sequence[:self.n_warmup - 1]
        warmup_data_input = warmup_data_input[np.newaxis]

        data_input = input_sequence[self.n_warmup - 1:]
        data_input = data_input[np.newaxis, :]

        warmup_data_target = input_sequence[1:self.n_warmup]
        warmup_data_target = warmup_data_target[np.newaxis]

        if testing_mode in self.getRCTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup +
                                    prediction_horizon]
            target = target.detach().cpu().numpy()
        else:
            raise ValueError(
                "[dimred_rc] Testing mode {:} not recognized.".format(
                    testing_mode))

        warmup_data_output = []
        warmup_latent_states = []
        h = np.zeros((self.rc_reservoir_size, 1))
        for t in range(self.n_warmup - 1):
            if self.display_output == True and t % 20 == 0:
                print(
                    "[dimred_rc] Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(
                        t, self.n_warmup, t / self.n_warmup * 100))
            input_t = warmup_data_input[0, t].unsqueeze(0).unsqueeze(0)
            input_t = self.model_autoencoder.encode(input_t)
            input_t = input_t[0, 0]
            i = np.reshape(input_t, (-1, 1))
            h = np.tanh(self.W_h @ h + self.W_in @ i)
            out = self.W_out @ self.augmentHidden(h)
            out = out[:, 0]

            warmup_latent_states.append(out)
            dec_inp = out[np.newaxis, np.newaxis]
            output_t = self.model_autoencoder.decode(dec_inp)
            output_t = output_t[0, 0]
            warmup_data_output.append(output_t)

        warmup_data_output = np.array(warmup_data_output)
        warmup_data_output = warmup_data_output[np.newaxis]

        warmup_latent_states = np.array(warmup_latent_states)

        time_latent_prop = 0.0
        prediction = []
        latent_states = []
        for t in range(prediction_horizon):
            if self.display_output == True and t % 20 == 0:
                print("[dimred_rc] Prediction: T {:}/{:}, {:2.3f}%".format(
                    t, prediction_horizon, t / prediction_horizon * 100))
            """ Different testing modes """
            if "iterative_latent" in testing_mode:
                i = out
                i = np.reshape(i, (-1, 1))
            elif "teacher_forcing" in testing_mode:
                input_t = data_input[0, t]
                i = i[..., np.newaxis]
            else:
                raise ValueError(
                    "[dimred_rc] Testing mode {:} not recognized.".format(
                        testing_mode))

            time_start = time.time()
            h = np.tanh(self.W_h @ h + self.W_in @ i)
            out = self.W_out @ self.augmentHidden(h)
            time_latent_prop += time.time() - time_start

            latent_states.append(out[:, 0])

            dec_inp = out[:, 0][np.newaxis, np.newaxis]
            output_t = self.model_autoencoder.decode(dec_inp)
            output_t = output_t[0, 0]
            prediction.append(output_t)

        latent_states = np.array(latent_states)
        prediction = np.array(prediction)

        time_total_per_iter = time_latent_prop / prediction_horizon
        print("[dimred_rc] Shapes of prediction/target/latent_states:")
        print("[dimred_rc] {:}".format(np.shape(prediction)))
        print("[dimred_rc] {:}".format(np.shape(target)))
        print("[dimred_rc] {:}".format(np.shape(latent_states)))

        # print("Min/Max")
        # print("Target:")
        # print(np.max(target[:,0]))
        # print(np.min(target[:,0]))
        # print("Prediction:")
        # print(np.max(prediction[:,0]))
        # print(np.min(prediction[:,0]))

        if self.n_warmup > 1:
            # warmup_data_target = warmup_data_target.cpu().detach().numpy()

            target_augment = np.concatenate((warmup_data_target[0], target),
                                            axis=0)
            prediction_augment = np.concatenate(
                (warmup_data_output[0], prediction), axis=0)
            latent_states_augmented = np.concatenate(
                (warmup_latent_states, latent_states), axis=0)
        else:
            # assert(self.has_predictor)
            target_augment = target
            prediction_augment = prediction
            latent_states_augmented = latent_states

        # print("[dimred_rc] Shapes of warmup prediction/target/latent_states:")
        # print("[dimred_rc] {:}".format(np.shape(warmup_data_output)))
        # print("[dimred_rc] {:}".format(np.shape(warmup_data_target)))
        # print("[dimred_rc] {:}".format(np.shape(warmup_latent_states)))

        return prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter

    def plot(self):
        if self.write_to_log:
            for testing_mode in self.getTestingModes():
                common_plot.writeLogfiles(self, testing_mode=testing_mode)
        else:
            print("[dimred_rc] # write_to_log=0. #")

        if self.params["plotting"]:
            for testing_mode in self.getTestingModes():
                common_plot.plot(self, testing_mode=testing_mode)
        else:
            print("[dimred_rc] # plotting=0. No plotting. #")


    def debug(self):
        print("[dimred_rc] # debug() #")
        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        self.model_name = self.model_name[:len("RC-")] + "GPU-" + self.model_name[len("RC-"):]
        print("self.model_name")
        print(self.model_name)
        # print(self.getMultiscaleTestingModes())



        for set_name in plot_on:

            for testing_mode in self.getTestingModes():

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
