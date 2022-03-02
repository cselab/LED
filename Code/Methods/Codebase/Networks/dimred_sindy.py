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
""" Utilities """
from .. import Utils as utils
from .. import Systems as systems
""" Printing """
from functools import partial

print = partial(print, flush=True)
import warnings
""" Common libraries """
from . import common_testing
from . import common_plot
""" PySINDy """
import pysindy as ps


class dimred_sindy():
    def __init__(self, params):
        super(dimred_sindy, self).__init__()
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
        # SINDy parameters
        ##################################################################
        self.sindy_integrator_type = params["sindy_integrator_type"]
        self.sindy_degree = params["sindy_degree"]
        self.sindy_threshold = params["sindy_threshold"]
        self.sindy_library = params["sindy_library"]
        self.sindy_interp_factor = params["sindy_interp_factor"]
        self.sindy_smoother_polyorder = params["sindy_smoother_polyorder"]
        self.sindy_smoother_window_size = params["sindy_smoother_window_size"]

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

    def getKeysInModelName(self, with_autoencoder=True, with_sindy=True):
        keys = {
            'sindy_integrator_type': '-TYPE_',
            'sindy_degree': '-PO_',
            'sindy_threshold': '-THRES_',
            'sindy_library': '-LIB_',
            'sindy_interp_factor': '-INT_',
            # 'sindy_smoother_polyorder': '-PORD_',
            # 'sindy_smoother_window_size': '-WS_',
        }

        if self.sindy_smoother_window_size > 0:
            keys.update({
                'sindy_smoother_polyorder': '-PORD_',
                'sindy_smoother_window_size': '-WS_',
            })

        if self.params["truncate_data_batches"]:
            keys.update({
                'truncate_data_batches': '-D_',
            })
        return keys

    def createModelName(self, with_residual_autoencoder=False):
        # print("# createModelName() #")
        keys = self.getKeysInModelName()
        # str_gpu = "GPU-" * self.gpu
        str_gpu = ""
        str_ = "SINDy-" + self.getAutoencoderName()
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        if with_residual_autoencoder:
            raise ValueError("TODO:")
        return str_

    def getAutoencoderName(self):
        return self.model_autoencoder.model_name

    def train(self):
        print("[dimred_sindy] # train() #")
        """ Sindy trained on raw data """
        data_path_train = self.data_path_train + "_raw"
        if os.path.isdir(data_path_train):
            print(
                "[dimred_sindy] Raw data directory {:} found. Using the raw (unbatched) data to train SINDy."
                .format(data_path_train))
        else:
            print(
                "[dimred_sindy] Raw data directory {:} not found. Using the batched data."
                .format(data_path_train))
            data_path_train = self.data_path_train

        batch_size = 1
        data_loader_train, sampler_train, data_set_train = utils.getDataLoader(
            data_path_train, self.data_info_dict, batch_size, shuffle=False)

        print("[dimred_sindy] Forward encoder and get latent states.")
        time_start = time.time()
        latent_states = []

        for sequence in data_loader_train:
            sequence = utils.getDataBatch(self,
                                          sequence,
                                          0,
                                          -1,
                                          dataset=data_set_train)
            latent_state = self.model_autoencoder.encode(sequence)
            latent_states.append(latent_state)

        latent_states = np.array(latent_states)
        latent_states = np.reshape(latent_states,
                                   (-1, *np.shape(latent_states)[2:]))
        latent_states = np.array(latent_states)
        n_ics, T, D = np.shape(latent_states)
        total_time = time.time() - time_start
        print("[dimred_sindy] Total encoder propagation time is {:}".format(
            utils.secondsToTimeStr(total_time)))

        training_data = latent_states
        print("[dimred_sindy] training data size = {:}".format(
            np.shape(training_data)))

        sindy_smoother_window_size = self.sindy_smoother_window_size
        sindy_smoother_polyorder = self.sindy_smoother_polyorder
        """ Smoothing the training data, in case sindy_smoother_window_size >0 """
        if sindy_smoother_window_size > 0:
            training_data = utils.smoothLatentSpace(
                self, training_data, sindy_smoother_window_size,
                sindy_smoother_polyorder)
            print("[dimred_sindy] smoothed training data size = {:}".format(
                np.shape(training_data)))

        interp_factor = self.sindy_interp_factor
        dt = self.data_info_dict["dt"]
        training_data, time_fine, dt_fine = utils.interpolateLatentSpace(
            self, training_data, interp_factor, dt)
        print("[dimred_sindy] latent state interpolation:")
        print("[dimred_sindy] dt={:}, dt_fine={:}".format(dt, dt_fine))

        if len(np.shape(training_data)) == 2:
            multiple_trajectories = False
        elif len(np.shape(training_data)) == 3:
            multiple_trajectories = True
        else:
            raise ValueError("Not implemented.")

        if self.sindy_integrator_type == "continuous":
            print("[dimred_sindy] (real) dt = {:}".format(dt))

            if self.sindy_library == "poly":
                feature_library = ps.PolynomialLibrary(
                    degree=self.sindy_degree)
            elif self.sindy_library == "fourier":
                feature_library = ps.FourierLibrary(
                    n_frequencies=self.sindy_degree)
            else:
                raise ValueError(
                    "Unknown feature library {:}.".format(feature_library))

            self.model_sindy = ps.SINDy(
                optimizer=ps.STLSQ(threshold=self.sindy_threshold),
                feature_library=feature_library,
                feature_names=list(["z_" + "{:d}".format(i)
                                    for i in range(D)]))

            dt_sindy = time_fine[1] - time_fine[0]
            print(
                "[dimred_sindy] (dimred_sindy) dt = {:}".format(time_fine[1] -
                                                                time_fine[0]))
            # self.model_sindy.fit(training_data.copy(), t=dt_sindy, multiple_trajectories=multiple_trajectories)
            print("[dimred_sindy] fitting...")
            self.model_sindy.fit(training_data,
                                 t=dt_sindy,
                                 multiple_trajectories=multiple_trajectories)
        elif self.sindy_integrator_type == "discrete":
            self.model_sindy = ps.SINDy(discrete_time=True)
            self.model_sindy.fit(training_data,
                                 multiple_trajectories=multiple_trajectories)

        else:
            raise ValueError("Invalid integrator type.")

        self.model_sindy.print()

        error_train = utils.plotTrainingExamplesAndGetError(
            self,
            time_fine,
            training_data,
            dt_sindy,
            plot=self.params["plotting"])
        self.error_train = error_train

        self.sindy_params = {
            "dt": dt,
            "dt_fine": dt_fine,
            "error_train": error_train,
        }

        self.save()

    def save(self):

        print("[dimred_sindy] Recording time...")
        self.total_training_time = time.time() - self.start_time
        print("[dimred_sindy] Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("[dimred_sindy] Script used {:} MB".format(self.memory))

        data = {
            "model_sindy": self.model_sindy,
            "sindy_params": self.sindy_params,
            "params": self.params,
            "model_name": self.model_name,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "error_train": self.error_train,
        }
        fields_to_write = [
            "memory",
            "total_training_time",
            "error_train",
        ]
        if self.write_to_log == 1:
            logfile_train = utils.getLogFileDir(self) + "/train.txt"
            print("[dimred_sindy] Writing to log-file in path {:}".format(
                logfile_train))
            utils.writeToLogFile(self, logfile_train, data, fields_to_write)

            logfile_model = utils.getLogFileDir(self) + "/logfile_model.txt"
            print("[dimred_sindy] Saving model as log-file.")
            with utils.Capturing() as temp_output:
                self.model_sindy.print()
            utils.writeLinesToLogfile(logfile_model, temp_output)

        data_path = utils.getModelDir(self) + "/data"
        utils.saveData(data, data_path, self.params["save_format"])

    def load(self, in_cpu=False):

        data_path = utils.getModelDir(self) + "/data"

        try:
            data = utils.loadData(data_path, self.params["save_format"])
            self.model_sindy = data["model_sindy"]
            self.sindy_params = data["sindy_params"]
            del data

        except Exception as inst:
            raise ValueError(
                "[dimred_sindy] Error, model {:s} not found.".format(
                    self.saving_model_path))

        print("[dimred_sindy] # Model loaded successfully!")

        return 0

    def loadAutoencoderModel(self, in_cpu=False):
        model_name_autoencoder = self.getAutoencoderName()
        print("[dimred_rc] Loading autoencoder with name:")
        print(model_name_autoencoder)
        AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model"
        self.model_autoencoder.model = utils.loadModel(
            AE_path, self.model_autoencoder.model, in_cpu=False, strict=False)
        return 0

    def getInitialRNNHiddenState(self, k):
        return [None]

    def getTestingModes(self):
        modes = []
        if self.params["iterative_latent_forecasting"]:
            modes.append("iterative_latent_forecasting")
        if self.params["teacher_forcing_forecasting"]:
            modes.append("teacher_forcing_forecasting")
        return modes

    def test(self):
        if self.load() == 0:
            test_on = []
            self.n_warmup = self.params["n_warmup"]
            assert self.n_warmup > 0
            print("[dimred_sindy] Warming-up steps: {:d}".format(
                self.n_warmup))

            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            for set_ in test_on:
                common_testing.testModesOnSet(
                    self, set_=set_, testing_modes=self.getTestingModes())
        return 0

    def warmup(self, input_seq, h):
        print("# warmup() #")
        input_seq = input_seq[:, -1:]
        time_latent_prop_t = 0.0
        return input_seq, h, [None], [None], time_latent_prop_t

    def forward(self,
                input_seq,
                h,
                input_is_latent=False,
                iterative_propagation_is_latent=None):
        shape_ = np.shape(input_seq)
        assert shape_[0] == 1
        input_seq = input_seq[0]

        time_latent_prop_t = 0.0
        for t in range(len(input_seq)):
            input_t = input_seq[t]

            if not input_is_latent:
                input_t = input_t[np.newaxis, np.newaxis]
                input_t = self.model_autoencoder.encode(input_t)
                latent_state = input_t[0, 0]
            else:
                latent_state = input_t

            time_start = time.time()
            if self.sindy_integrator_type == "continuous":
                time_domain = np.linspace(
                    0, 1, num=self.sindy_interp_factor + 1,
                    endpoint=True) * self.sindy_params["dt"]
            elif self.sindy_integrator_type == "discrete":
                time_domain = 2
            else:
                raise ValueError("Invalid integrator type.")

            # print(time_domain)
            latent_state_pred = self.model_sindy.simulate(
                latent_state, time_domain)
            # Get only the last timestep (dt)
            latent_state_pred = latent_state_pred[-1]
            latent_states_pred_ = latent_state_pred
            # print(latent_states_pred_)

            time_latent_prop_t += time.time() - time_start

        latent_state_pred = latent_state_pred[np.newaxis]
        latent_state_pred = latent_state_pred[np.newaxis]

        outputs = self.model_autoencoder.decode(latent_state_pred)  #[0,0]
        return outputs, h, latent_state_pred, latent_state_pred, time_latent_prop_t

    def sendHiddenStateToGPU(self, temp):
        return temp

    def forecast(self, input_t, hidden_state, horizon=None):
        # print("# forecast() #")

        time_start = time.time()

        # LATENT/ORIGINAL DYNAMICS PROPAGATION
        if self.sindy_integrator_type == "continuous":
            time_domain = np.linspace(
                0,
                horizon,
                num=(horizon * self.sindy_interp_factor) + 1,
                endpoint=True) * self.sindy_params["dt"]
        elif self.sindy_integrator_type == "discrete":
            time_domain = horizon
        else:
            raise ValueError("Invalid integrator type.")

        # print("[dimred_sindy] time_domain = {:}".format(time_domain))
        assert (np.shape(input_t)[0] == 1)
        input_t = input_t[0]
        assert (np.shape(input_t)[0] == 1)
        input_t = input_t[0]
        input_t = np.array(input_t)
        # print("np.shape(input_t)={:}".format(np.shape(input_t)))

        latent_states = self.model_sindy.simulate(input_t, time_domain)
        latent_states = latent_states[1:]
        latent_states = latent_states[::self.sindy_interp_factor]

        time_end = time.time()
        time_latent_prop = (time_end - time_start)

        latent_states = latent_states[np.newaxis]
        prediction = self.model_autoencoder.decode(latent_states)

        last_hidden_state = []
        latent_states_pred = latent_states
        return prediction, last_hidden_state, latent_states, latent_states_pred, time_latent_prop

    def predictSequence(self,
                        input_sequence,
                        testing_mode=None,
                        dt=1,
                        prediction_horizon=None):
        print("[dimred_sindy] # predictSequence() #")
        print("[dimred_sindy] {:}:".format(np.shape(input_sequence)))
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.n_warmup != prediction_horizon:
            raise ValueError(
                "[dimred_sindy] Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})"
                .format(N, self.n_warmup, prediction_horizon))

        warmup_data_input = input_sequence[:self.n_warmup - 1]
        warmup_data_input = warmup_data_input[np.newaxis]
        warmup_data_target = input_sequence[1:self.n_warmup]
        warmup_data_target = warmup_data_target[np.newaxis]

        if testing_mode in self.getTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup +
                                    prediction_horizon]
            target = target.detach().cpu().numpy()
        else:
            raise ValueError(
                "[dimred_sindy] Testing mode {:} not recognized.".format(
                    testing_mode))

        if self.n_warmup > 1:
            warmup_latent_states = self.model_autoencoder.encode(
                warmup_data_input)
            warmup_data_output = self.model_autoencoder.decode(
                warmup_latent_states)
            latent_states_pred = warmup_latent_states
        else:
            pass

        prediction = []

        if ("iterative_latent" in testing_mode):
            input_latent = latent_states_pred[:, -1, :]
            input_t = input_latent
        elif "teacher_forcing" in testing_mode:
            input_t = input_sequence[self.n_warmup - 1:-1]
            input_t = input_t[np.newaxis]
        else:
            raise ValueError(
                "[dimred_sindy] I do not know how to initialize the state for {:}."
                .format(testing_mode))

        time_latent_prop = 0.0
        time_start = time.time()
        if "teacher_forcing" in testing_mode:
            latent_states_input = self.model_autoencoder.encode(input_t)
            latent_states = []
            assert (np.shape(latent_states_input)[0] == 1)
            latent_states_input = latent_states_input[0]
            for latent_state_input in latent_states_input:
                input_t = latent_state_input

                if self.sindy_integrator_type == "continuous":
                    time_domain = np.linspace(
                        0, 1, num=self.sindy_interp_factor + 1,
                        endpoint=True) * self.sindy_params["dt"]
                elif self.sindy_integrator_type == "discrete":
                    time_domain = 2
                else:
                    raise ValueError("Invalid integrator type.")
                latent_state_pred = self.model_sindy.simulate(
                    input_t, time_domain)
                # Get only the last timestep (dt)
                latent_state_pred = latent_state_pred[-1]
                latent_state_pred = latent_state_pred[np.newaxis]
                latent_states.append(latent_state_pred)
            latent_states = np.array(latent_states)
            latent_states = np.swapaxes(latent_states, 0, 1)
            prediction = self.model_autoencoder.decode(latent_states)

        elif "iterative_latent" in testing_mode:
            time0 = time.time()

            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            if self.sindy_integrator_type == "continuous":
                time_domain = np.linspace(
                    0,
                    prediction_horizon,
                    num=(prediction_horizon * self.sindy_interp_factor) + 1,
                    endpoint=True) * self.sindy_params["dt"]
            elif self.sindy_integrator_type == "discrete":
                time_domain = prediction_horizon
            else:
                raise ValueError("Invalid integrator type.")

            assert (np.shape(input_t)[0] == 1)
            input_t = input_t[0]

            latent_states = self.model_sindy.simulate(input_t, time_domain)
            latent_states = latent_states[1:]
            latent_states = latent_states[::self.sindy_interp_factor]

            latent_states = latent_states[np.newaxis]
            prediction = self.model_autoencoder.decode(latent_states)
            time1 = time.time()
            time_latent_prop += (time1 - time0)
        else:
            raise ValueError(
                "[dimred_sindy] Testing mode {:} not recognized.".format(
                    testing_mode))
        time_end = time.time()
        time_total = time_end - time_start

        # Correcting the time-measurement in case of evolution of the original system (in this case, we do not need to internally propagate the latent space of the RNN)
        time_total = time_latent_prop

        time_total_per_iter = time_total / prediction_horizon

        prediction = prediction[0]
        latent_states = latent_states[0]

        target = np.array(target)
        prediction = np.array(prediction)
        latent_states = np.array(latent_states)

        print("[dimred_sindy] Shapes of prediction/target/latent_states:")
        print("[dimred_sindy] {:}".format(np.shape(prediction)))
        print("[dimred_sindy] {:}".format(np.shape(target)))
        print("[dimred_sindy] {:}".format(np.shape(latent_states)))

        # print("Min/Max")
        # print("Target:")
        # print(np.max(target[:,0]))
        # print(np.min(target[:,0]))
        # print("Prediction:")
        # print(np.max(prediction[:,0]))
        # print(np.min(prediction[:,0]))

        if self.n_warmup > 1:

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

    def plot(self):
        if self.write_to_log:
            for testing_mode in self.getTestingModes():
                common_plot.writeLogfiles(self, testing_mode=testing_mode)
        else:
            print("[dimred_sindy] # write_to_log=0. #")

        if self.params["plotting"]:
            for testing_mode in self.getTestingModes():
                common_plot.plot(self, testing_mode=testing_mode)
        else:
            print("[dimred_sindy] # plotting=0. No plotting. #")





    def debug(self):
        print("[dimred_sindy] # debug() #")
        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        self.model_name = self.model_name[:len("SINDy-")] + "GPU-" + self.model_name[len("SINDy-"):]
        print("self.model_name")
        print(self.model_name)

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
