#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np

import torch
import sys
from torch.autograd import Variable

import time
import warnings
import h5py

from .. import Utils as utils
from .. import Systems as systems
from . import utils_multiscale_plotting as utils_multiscale_plotting

from . import utils_multiscale_structured as mutils_structured
from . import utils_multiscale_unstructured as mutils_unstructured

from tqdm import tqdm

class multiscaleTestingClass:
    def __init__(self, model, params_dict):
        print("[utils_multiscale] # multiscaleTestingClass #")
        super(multiscaleTestingClass, self).__init__()
        self.model = model
        self.params = params_dict

        self.multiscale_micro_steps_list = params_dict[
            "multiscale_micro_steps_list"]
        self.multiscale_macro_steps_list = params_dict[
            "multiscale_macro_steps_list"]
        self.model_class = self.model.__class__.__name__
        assert self.model_class in [
            "crnn", "dimred_rc", "dimred_rnn", "dimred_sindy"
        ]
        """ Adding parameters to the data_info_dict """
        self.microdynamics_info_dict = systems.getMicrodynamicsInfo(self.model)

    def getMultiscaleParams(self, testing_mode, prediction_horizon):
        temp = testing_mode.split("_")
        multiscale_macro_steps = int(float(temp[-1]))
        multiscale_micro_steps = int(float(temp[-3]))
        macro_steps_per_round = []
        micro_steps_per_round = []
        steps = 0
        while (steps < prediction_horizon):
            steps_to_go = prediction_horizon - steps
            if steps_to_go >= multiscale_macro_steps:
                macro_steps_per_round.append(multiscale_macro_steps)
                steps += multiscale_macro_steps
            elif steps_to_go != 0:
                macro_steps_per_round.append(steps_to_go)
                steps += steps_to_go
            else:
                raise ValueError("This was not supposed to happen.")
            steps_to_go = prediction_horizon - steps
            if steps_to_go >= multiscale_micro_steps:
                micro_steps_per_round.append(multiscale_micro_steps)
                steps += multiscale_micro_steps
            elif steps_to_go != 0:
                micro_steps_per_round.append(steps_to_go)
                steps += steps_to_go

        print(
            "[utils_multiscale] macro_steps_per_round: \n[utils_multiscale] {:}"
            .format(macro_steps_per_round))
        print(
            "[utils_multiscale] micro_steps_per_round: \n[utils_multiscale] {:}"
            .format(micro_steps_per_round))
        multiscale_rounds = np.max(
            [len(micro_steps_per_round),
             len(macro_steps_per_round)])
        print("[utils_multiscale] multiscale_rounds: \n[utils_multiscale] {:}".
              format(multiscale_rounds))
        return multiscale_rounds, macro_steps_per_round, micro_steps_per_round, multiscale_micro_steps, multiscale_macro_steps

    def getMultiscaleTestingModes(self):
        modes = []
        for micro_steps in self.multiscale_micro_steps_list:
            for macro_steps in self.multiscale_macro_steps_list:
                mode = "multiscale_forecasting_micro_{:}_macro_{:}".format(
                    int(micro_steps), int(macro_steps))
                modes.append(mode)
        return modes

    def getFieldsToCompare(self):
        error_labels = utils.getErrorLabelsDict(self.model)
        error_labels = error_labels.keys()
        fields_to_compare = [key for key in error_labels]

        # fields_to_compare = [key + "_avg" for key in error_labels]
        # fields_to_compare += error_labels

        # print(error_labels_avg)
        # print(ark)
        # fields_to_compare = [
        # "time_total_per_iter",
        # # "rmnse_avg_over_ics",
        # "rmnse_avg",
        # # "num_accurate_pred_050_avg",
        # # "error_freq",
        # ]
        fields_to_compare.append("time_total_per_iter")
        return fields_to_compare

    def test(self):
        if self.model.gpu:
            self.gpu_monitor_process = utils.GPUMonitor(
                self.params["gpu_monitor_every"], self.model.multinode,
                self.model.rank_str)

        self.test_()

        if self.model.gpu:
            self.gpu_monitor_process.stop()

    def test_(self):
        """ Load model """
        if self.model.load() == 0:
            print("[utils_multiscale] # Model loaded successfully!")

            with torch.no_grad():
                if self.params["n_warmup"] is None:
                    self.model.n_warmup = 0
                else:
                    self.model.n_warmup = int(self.params["n_warmup"])

                print("[utils_multiscale] Warming up steps: {:d}".format(
                    self.model.n_warmup))

                test_on = []
                if self.params["test_on_test"]: test_on.append("test")
                if self.params["test_on_val"]: test_on.append("val")
                if self.params["test_on_train"]: test_on.append("train")

                # print(test_on)
                for set_ in test_on:
                    self.testOnSet(set_)
        return 0

    def testOnSet(self, set_="train"):
        print("[utils_multiscale] #####     Testing on set: {:}     ######".
              format(set_))
        """ Macro scale dt """
        dt = self.model.data_info_dict["dt"]
        print("[utils_multiscale] Macro scale dt = {:}".format(dt))

        if set_ == "test":
            data_path = self.model.data_path_test
        elif set_ == "val":
            data_path = self.model.data_path_val
        elif set_ == "train":
            data_path = self.model.data_path_train
        else:
            raise ValueError("Invalid set {:}.".format(set_))

        data_loader_test, _, data_set = utils.getDataLoader(
            data_path,
            self.model.data_info_dict,
            batch_size=1,
            shuffle=False,
        )

        self.testingRoutine(data_loader_test, dt, set_, data_set)

        return 0

    def testingRoutine(
        self,
        data_loader,
        dt,
        set_,
        data_set,
    ):

        for testing_mode in self.getMultiscaleTestingModes():
            self.testOnMode(data_loader, dt, set_, testing_mode, data_set)

        return 0

    def testOnMode(self, data_loader, dt, set_, testing_mode, data_set):
        assert (testing_mode in self.getMultiscaleTestingModes())
        assert (set_ in ["train", "test", "val"])
        print("[utils_multiscale] ---- Testing on Mode {:} ----".format(
            testing_mode))

        if self.model.num_test_ICS > 0:

            if self.model.data_info_dict["structured"]:

                # Testing on structured data
                results = mutils_structured.predictIndexesOnStructured(
                    self, data_set, dt, set_, testing_mode)

            else:
                results = mutils_unstructured.predictIndexes(
                    self, data_loader, dt, set_, testing_mode)

            data_path = utils.getResultsDir(
                self.model) + "/results_{:}_{:}".format(testing_mode, set_)
            utils.saveData(results, data_path, self.model.save_format)
        else:
            print(
                "[utils_multiscale] Model has RNN but no initial conditions set to test num_test_ICS={:}."
                .format(self.model.num_test_ICS))

        return 0

    def writeLogfiles(self):
        print("[utils_multiscale] # writeLogfiles() #")

        write_logs_on = []
        if self.model.params["test_on_test"]: write_logs_on.append("test")
        if self.model.params["test_on_val"]: write_logs_on.append("val")
        if self.model.params["test_on_train"]: write_logs_on.append("train")

        for set_name in write_logs_on:

            # Postprocessing of RNN testing results
            for testing_mode in self.getMultiscaleTestingModes():

                # Loading the results
                data_path = utils.getResultsDir(
                    self.model) + "/results_{:}_{:}".format(
                        testing_mode, set_name)
                results = utils.loadData(data_path, self.model.save_format)

                if self.model.write_to_log:
                    logfile = utils.getLogFileDir(
                        self.model) + "/results_{:}_{:}.txt".format(
                            testing_mode, set_name)
                    utils.writeToLogFile(self.model, logfile, results,
                                         results["fields_2_save_2_logfile"])

    def plot(self):
        self.writeLogfiles()
        if self.model.params["plotting"]:
            self.plot_()
        else:
            print("[utils_multiscale] # plotting=0. No plotting. #")

    def plot_(self):
        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        for set_name in plot_on:

            fields_to_compare = self.getFieldsToCompare()
            fields_to_compare = systems.addFieldsToCompare(
                self.model, fields_to_compare)

            dicts_to_compare = {}
            latent_states_dict = {}

            write_logs_on = []

            for testing_mode in self.getMultiscaleTestingModes():

                # Loading the results
                data_path = utils.getResultsDir(
                    self.model) + "/results_{:}_{:}".format(
                        testing_mode, set_name)
                results = utils.loadData(data_path, self.model.save_format)

                # # Plotting the state distributions
                # if self.model.params["plot_state_distributions"]:
                #     utils.plotStateDistributions(self.model, results, set_name,
                #                                  testing_mode)

                # Plotting the state distributions specific to a system
                if self.model.params["plot_system"]:
                    systems.plotSystem(self.model, results, set_name,
                                       testing_mode)

                if self.model.params["plot_errors_in_time"]:
                    utils.plotErrorsInTime(self.model, results, set_name,
                                           testing_mode)

                ic_indexes = results["ic_indexes"]
                dt = results["dt"]
                n_warmup = results["n_warmup"]

                predictions_augmented_all = results[
                    "predictions_augmented_all"]
                targets_augmented_all = results["targets_augmented_all"]

                predictions_all = results["predictions_all"]
                targets_all = results["targets_all"]
                latent_states_all = results["latent_states_all"]

                latent_states_dict[testing_mode] = latent_states_all

                results_dict = {}
                for field in fields_to_compare:
                    results_dict[field] = results[field]
                dicts_to_compare[testing_mode] = results_dict

                if self.model.params["plot_testing_ics_examples"]:

                    max_index = np.min(
                        [3, np.shape(results["targets_all"])[0]])

                    for idx in range(max_index):
                        print("[utils_multiscale] Plotting IC {:}/{:}.".format(
                            idx, max_index))

                        # Plotting the latent dynamics for these examples
                        if self.model.params["plot_latent_dynamics"]:
                            utils.plotLatentDynamics(self.model, set_name,
                                                     latent_states_all[idx],
                                                     idx, testing_mode)

                        results_idx = {
                            "Reference": targets_all[idx],
                            "prediction": predictions_all[idx],
                            "latent_states": latent_states_all[idx],
                            "fields_2_save_2_logfile": [],
                        }

                        self.model.parent = self
                        utils.createIterativePredictionPlots(self.model, \
                            targets_all[idx], \
                            predictions_all[idx], \
                            dt, idx, set_name, \
                            testing_mode=testing_mode, \
                            latent_states=latent_states_all[idx], \
                            warm_up=n_warmup, \
                            target_augment=targets_augmented_all[idx], \
                            prediction_augment=predictions_augmented_all[idx], \
                            )

            if self.model.params["plot_multiscale_results_comparison"]:
                utils_multiscale_plotting.plotMultiscaleResultsComparison(
                    self.model,
                    dicts_to_compare,
                    set_name,
                    fields_to_compare,
                    results["dt"],
                )
                # utils.plotLatentDynamicsComparison(self, latent_states_dict, set_name)




    def debug(self):
        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        self.model.model_name = "GPU-" + self.model.model_name
        print(self.model.model_name)
        # print(self.getMultiscaleTestingModes())

        for set_name in plot_on:

            for testing_mode in self.getMultiscaleTestingModes():

                # Loading the results
                data_path = utils.getResultsDir(
                    self.model) + "/results_{:}_{:}".format(
                        testing_mode, set_name)
                results = utils.loadData(data_path, self.model.save_format)

                predictions_all = results["predictions_all"]
                targets_all = results["targets_all"]

                # print(np.shape(predictions_all))
                # print(np.shape(targets_all))

                error_dict = utils.getErrorLabelsDict(self.model)

                num_ics = len(targets_all)
                T = len(targets_all[0])
                tqdm_bar = tqdm(total=num_ics * T)

                # print(ark)
                for ic_num in range(num_ics):
                    predictions_all_ic = predictions_all[ic_num]
                    targets_all_ic = targets_all[ic_num]

                    error_dict_ic = utils.getErrorLabelsDict(self.model)

                    T = len(targets_all_ic)
                    for t in range(T):
                        target_save_path = targets_all_ic[t]
                        prediction_save_path = predictions_all_ic[t]


                        target_save_path = np.array(target_save_path)
                        target_save_path = correctStructuredDataPaths(self.model, target_save_path)
                        target_save = utils.getDataHDF5Fields(target_save_path[0], [target_save_path[1]])


                        prediction_save_path = np.array(prediction_save_path)
                        prediction_save_path = correctStructuredDataPaths(self.model, prediction_save_path)
                        prediction_save = utils.getDataHDF5Fields(prediction_save_path[0], [prediction_save_path[1]])

                        prediction_save = prediction_save[0]
                        target_save = target_save[0]

                        # print(np.shape(prediction_save))
                        # print(np.shape(target_save))
                        errors = utils.computeErrors(
                            target_save,
                            prediction_save,
                            self.model.data_info_dict,
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
                utils.saveData(results, data_path, self.model.save_format)

                for key in results: print(key)
