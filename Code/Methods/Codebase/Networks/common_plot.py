#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

from .. import Utils as utils
from .. import Systems as systems

import numpy as np


def writeLogfiles(model, testing_mode=None):
    if model.write_to_log:
        print("[writeLogfiles()] # writeLogfiles() #")
        write_logs_on = []
        if model.params["test_on_test"]: write_logs_on.append("test")
        if model.params["test_on_val"]: write_logs_on.append("val")
        if model.params["test_on_train"]: write_logs_on.append("train")

        for set_name in write_logs_on:
            data_path = utils.getResultsDir(model) + "/results_{:}_{:}".format(
                testing_mode, set_name)
            results = utils.loadData(data_path, model.save_format)
            # Writing to a log-file
            logfile = utils.getLogFileDir(
                model) + "/results_{:}_{:}.txt".format(testing_mode, set_name)
            utils.writeToLogFile(model, logfile, results,
                                 results["fields_2_save_2_logfile"])
    return 0


def plot(model, testing_mode):
    print("[common_plot] # plot() #")
    plot_on = []
    if model.params["test_on_test"]: plot_on.append("test")
    if model.params["test_on_val"]: plot_on.append("val")
    if model.params["test_on_train"]: plot_on.append("train")

    for set_name in plot_on:

        if "autoencoder" in testing_mode or "dimred" in testing_mode:
            data_path = utils.getResultsDir(model) + "/results_{:}_{:}".format(
                testing_mode, set_name)
            results = utils.loadData(data_path, model.save_format)

            # # Computing the spectrum
            # if model.data_info_dict["compute_errors_in_time"] and model.params["plot_errors_in_time"]:
            #     utils.plotErrorsInTime(model, results, set_name, testing_mode)

            # # Plot the distribution on the state
            # if model.params["plot_state_distributions"]:
            #     utils.plotStateDistributions(model, results, set_name, testing_mode)

            # Plot distributions specific to a system
            if model.params["plot_system"]:
                systems.plotSystem(model, results, set_name, testing_mode)

            # Plotting examples of testing initial conditions
            if model.params["plot_testing_ics_examples"]:
                ic_plot = np.min([1, len(results["inputs_all"])])
                for ic in range(ic_plot):
                    print("[common_plot] IC {:}".format(ic))

                    # Plotting the latent dynamics for these examples
                    if model.params["plot_latent_dynamics"]:
                        utils.plotLatentDynamics(
                            model, set_name, results["latent_states_all"][ic],
                            ic, testing_mode)

                    utils.plotTestingContours(
                        model,
                        results["inputs_all"][ic],
                        results["outputs_all"][ic],
                        results["dt"],
                        ic,
                        set_name,
                        latent_states=results["latent_states_all"][ic],
                        testing_mode=testing_mode,
                    )
        elif "iterative_state_forecasting" in testing_mode or "iterative_latent_forecasting" in testing_mode or "teacher_forcing_forecasting" in testing_mode:

            # Loading the results
            data_path = utils.getResultsDir(model) + "/results_{:}_{:}".format(
                testing_mode, set_name)
            results = utils.loadData(data_path, model.save_format)

            # Plotting the error in time
            if model.data_info_dict["compute_errors_in_time"] and model.params[
                    "plot_errors_in_time"]:
                utils.plotErrorsInTime(model, results, set_name, testing_mode)

            # # Plotting the state distributions
            # if model.params["plot_state_distributions"]:
            #     utils.plotStateDistributions(model, results, set_name, testing_mode)

            # Plotting the state distributions specific to a system
            if model.params["plot_system"]:
                systems.plotSystem(model, results, set_name, testing_mode)

            # Computing the spectrum
            if model.params["compute_spectrum"]:
                utils.plotSpectrum(model, results, set_name, testing_mode)

            ic_indexes = results["ic_indexes"]
            dt = results["dt"]
            n_warmup = results["n_warmup"]

            predictions_augmented_all = results["predictions_augmented_all"]
            targets_augmented_all = results["targets_augmented_all"]

            predictions_all = results["predictions_all"]
            targets_all = results["targets_all"]
            latent_states_all = results["latent_states_all"]

            if model.params["plot_testing_ics_examples"]:
                # max_index = np.min([5, np.shape(results["targets_all"])[0]])
                max_index = np.min([1, np.shape(results["targets_all"])[0]])
                for idx in range(max_index):
                    print("[common_plot] IC {:}".format(idx))

                    results_idx = {
                        "Reference": targets_all[idx],
                        "prediction": predictions_all[idx],
                        "latent_states": latent_states_all[idx],
                        "fields_2_save_2_logfile": [],
                    }

                    # Plotting the latent dynamics for these examples
                    if model.params["plot_latent_dynamics"]:
                        utils.plotLatentDynamics(
                            model,
                            set_name,
                            results["latent_states_augmented_all"][idx],
                            idx,
                            testing_mode,
                            warm_up=n_warmup)

                    utils.createIterativePredictionPlots(model, \
                        targets_all[idx], \
                        predictions_all[idx], \
                        dt, idx, set_name, \
                        testing_mode=testing_mode, \
                        latent_states=latent_states_all[idx], \
                        warm_up=n_warmup, \
                        target_augment=targets_augmented_all[idx], \
                        prediction_augment=predictions_augmented_all[idx], \
                        )

                    # utils.plotTestingContours(model, \
                    #     targets_all[idx], \
                    #     predictions_all[idx], \
                    #     dt, \
                    #     idx, \
                    #     set_name, \
                    #     latent_states=latent_states_all[idx], \
                    #     testing_mode=testing_mode, \
                    # )
