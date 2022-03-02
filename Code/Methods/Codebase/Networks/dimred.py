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
""" Torch """
import torch
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


class dimred():
    def __init__(self, params):
        super(dimred, self).__init__()
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

        ##############################################################
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

        ##################################################################
        # DimRed parameters (PCA/DiffMaps)
        ##################################################################
        self.latent_state_dim = params["latent_state_dim"]
        self.dimred_method = params["dimred_method"]

        if self.dimred_method == "pca":
            """ PCA has no hyper-parameters """
            pass

        elif self.dimred_method == "diffmaps":
            """ hyperparameter of diffusion maps that multiplies the median of the data """
            self.diffmaps_weight = self.params["diffmaps_weight"]
            """ number of neighbors considered during lifting """
            self.diffmaps_num_neighbors = self.params["diffmaps_num_neighbors"]

        else:
            raise ValueError(
                "Invalid dimensionality reduction method {:}.".format(
                    self.dimred_method))
        """ Model name """
        self.model_name = self.createModelName()

        print("[dimred] - model_name:")
        print("[dimred] {:}".format(self.model_name))
        self.saving_model_path = utils.getModelDir(self) + "/model"

        utils.makeDirectories(self)
        """ Saving some info file for the model """
        data = {"params": params, "name": self.model_name}
        data_path = utils.getModelDir(self) + "/info"
        utils.saveData(data, data_path, self.params["save_format"])
        self.data_info_dict = systems.getSystemDataInfo(self)

    def getKeysInModelName(self):
        keys = {
            'scaler': '-scaler_',
            'dimred_method': '-METHOD_',
        }

        if self.dimred_method == "pca":
            keys.update({
                'latent_state_dim': '-LD_',
            })

        elif self.dimred_method == "diffmaps":
            keys.update({
                'diffmaps_weight': '-DMW_',
                'diffmaps_num_neighbors': '-DMN_',
                'latent_state_dim': '-LD_',
            })

        else:
            raise ValueError("Not implemented.")

        return keys

    def createModelName(self):
        keys = self.getKeysInModelName()
        str_ = "DimRed"
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        return str_

    def applyDimRed(self, data):
        """ Use the dimensionality reduction method to project the data """
        assert len(
            np.shape(data)
        ) == 2 + 1 + self.channels, "[applyDimRed()] Error, len(np.shape(data))={:}, while 2+1+self.channels={:}".format(
            len(np.shape(data)), 2 + 1 + self.channels)
        shape_ = np.shape(data)
        data = np.reshape(data, (shape_[0] * shape_[1], -1))
        data = self.dimred_model.transform(data)
        data = np.reshape(data, (shape_[0], shape_[1], -1))
        return data

    def applyInverseDimRed(self, data):
        assert len(np.shape(data)) == 3
        """ Use the dimensionality reduction method to lift the projected data """
        shape_ = np.shape(data)
        data = np.reshape(data, (shape_[0] * shape_[1], -1))
        data = self.dimred_model.inverse_transform(data)
        data = np.reshape(data,
                          (shape_[0], shape_[1], *utils.getInputShape(self)))
        return data

    def encodeDecode(self, data):
        latent_state = self.applyDimRed(data)
        output = self.applyInverseDimRed(latent_state)
        return output, latent_state

    def encode(self, data):
        latent_state = self.applyDimRed(data)
        return latent_state

    def decode(self, latent_state):
        data = self.applyInverseDimRed(latent_state)
        return data

    def train(self):
        print("[dimred] # train() # Performing dimensionality reduction...")

        data = []
        batch_size = 1
        """ Dimensionality reduction performed on raw data """
        data_path_train = self.data_path_train + "_raw"
        if os.path.isdir(data_path_train):
            print(
                "[dimred] Raw data directory {:} found. Using the raw (unbatched) data to perform dimensionality reduction."
                .format(data_path_train))
        else:
            print(
                "[dimred] Raw data directory {:} not found. Using the batched data."
                .format(data_path_train))
            data_path_train = self.data_path_train

        data_loader_train, sampler_train, data_set_train = utils.getDataLoader(
            data_path_train,
            self.data_info_dict,
            batch_size,
            shuffle=False,
        )
        for sequence in data_loader_train:
            sequence = utils.getDataBatch(self,
                                          sequence,
                                          0,
                                          -1,
                                          dataset=data_set_train)
            if torch.is_tensor(sequence):
                sequence = sequence.detach().cpu().numpy()
            data.append(sequence)

        data = np.array(data)
        data = np.reshape(data, (-1, *np.shape(data)[2:]))
        data = np.reshape(data, (np.shape(data)[0] * np.shape(data)[1], -1))

        shape_ = np.shape(data)
        data = np.unique(data, axis=0)
        shape = np.shape(data)
        print("[dimred] Douplicates removed from shape {:} -> {:}.".format(
            shape_, shape))

        print(
            "[dimred] Performing dimensionality reduction on data with size {:}."
            .format(np.shape(data)))
        """ Perform PCA in data """
        if self.dimred_method == "pca":
            from sklearn.decomposition import PCA
            self.dimred_model = PCA(n_components=self.latent_state_dim)
            self.dimred_model.fit(data)

            data_red = self.dimred_model.transform(data)

        elif self.dimred_method == "diffmaps":
            from .diffmaps import DiffusionMap

            self.dimred_model = DiffusionMap(data, self.latent_state_dim,
                                             self.diffmaps_weight)

        else:
            raise ValueError(
                "Incalid dimensionality reduction method {:}.".format(
                    self.dimred_method))

        self.saveDimRed()

    def delete(self):
        pass

    def saveDimRed(self):
        model_name_dimred = self.createModelName()
        print(
            "[dimred] Saving dimensionality reduction results with name: {:}".
            format(model_name_dimred))

        print("[dimred] Recording time...")
        self.total_training_time = time.time() - self.start_time

        print("[dimred] Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("[dimred] Script used {:} MB".format(self.memory))

        data = {
            "params": self.params,
            "model_name": self.model_name,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "dimred_model": self.dimred_model,
        }
        fields_to_write = [
            "memory",
            "total_training_time",
        ]
        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + model_name_dimred + "/train.txt"
            print("[dimred] Writing to log-file in path {:}".format(
                logfile_train))
            utils.writeToLogFile(self, logfile_train, data, fields_to_write)

        data_folder = self.saving_path + self.model_dir + model_name_dimred
        os.makedirs(data_folder, exist_ok=True)
        data_path = data_folder + "/data"
        utils.saveData(data, data_path, self.params["save_format"])

    def load(self):
        model_name_dimred = self.createModelName()
        print(
            "[dimred] Loading dimensionality reduction from model: {:}".format(
                model_name_dimred))
        data_path = self.saving_path + self.model_dir + model_name_dimred + "/data"
        print("[dimred] Datafile: {:}".format(data_path))
        try:
            data = utils.loadData(data_path, self.params["save_format"])
        except Exception as inst:
            raise ValueError(
                "[Error] Dimensionality reduction results {:s} not found.".
                format(data_path))
        self.dimred_model = data["dimred_model"]
        del data
        return 0

    def test(self):
        if self.load() == 0:
            testing_modes = self.getTestingModes()
            test_on = []
            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            for set_ in test_on:
                common_testing.testModesOnSet(self,
                                              set_=set_,
                                              testing_modes=testing_modes)
        return 0

    def getTestingModes(self):
        return ["dimred_testing"]

    def plot(self):
        if self.write_to_log:
            common_plot.writeLogfiles(self, testing_mode="dimred_testing")
        else:
            print("[dimred] # write_to_log=0. #")

        if self.params["plotting"]:
            common_plot.plot(self, testing_mode="dimred_testing")
        else:
            print("[dimred] # plotting=0. No plotting. #")



