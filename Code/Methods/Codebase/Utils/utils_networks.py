#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import os
import numpy as np
from . import utils_processing

import torch


def loadModel(model_path, model, in_cpu=False, strict=False):
    gpu = torch.cuda.is_available()
    """ Loading model in GPU if available """
    try:
        if not in_cpu and gpu:
            print("[utils_networks] # Loading model in GPU...")
            model.load_state_dict(torch.load(model_path), strict=strict)
        else:
            print("[utils_networks] # Loading model in CPU...")
            model.load_state_dict(torch.load(model_path,
                                             map_location=torch.device('cpu')),
                                  strict=strict)
    except Exception as inst:
        print(
            "[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the autoencoder ? If you run on a cluster, is the GPU detected ? Did you use the srun command ? If you run on a cluster, is the GPU detected ? Did you use the srun command ?"
            .format(model_path))
        raise ValueError(inst)
    return model


def toPrecision(model, data):
    if isinstance(data, list):
        return [toPrecision(model, element) for element in data]
    if model.params["precision"] == "single":
        if torch.is_tensor(data):
            return data.float()
        else:
            return data.astype(np.float32)
    elif model.params["precision"] == "double":
        if torch.is_tensor(data):
            return data.double()
        else:
            return data.astype(np.float64)
    else:
        raise ValueError("Data {:} not recognized.".format(data))


def toTensor(model, data):
    if isinstance(data, list):
        return [toTensor(model, element) for element in data]
    if not model.gpu:
        return model.torch_dtype(data)

    else:
        if torch.is_tensor(data):
            data = data.cuda()
            return model.torch_dtype(data)

        else:
            return model.torch_dtype(data)


def transform2Tensor(model, data):
    data = toPrecision(model, data)
    data = toTensor(model, data)
    return data


def getInputShape(model):
    if model.channels == 1:
        return [model.input_dim, model.Dx]
    elif model.channels == 2:
        return [model.input_dim, model.Dx, model.Dy]
    else:
        raise ValueError("Not implemented.")


def computeLatentStateInfo(model, latent_states_all):
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
                                   (-1, model.latent_state_dim))
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


def addResultsIterative(
    model,
    predictions_all,
    targets_all,
    latent_states_all,
    predictions_augmented_all,
    targets_augmented_all,
    latent_states_augmented_all,
    time_total_per_iter,
    testing_mode,
    ic_indexes,
    dt,
    error_dict,
    error_dict_avg,
    latent_states_all_data=None,
):

    additional_results_dict, additional_errors_dict = utils_processing.computeAdditionalResults(
        model, predictions_all, targets_all, dt)
    error_dict_avg = {**error_dict_avg, **additional_errors_dict}

    state_statistics = utils_processing.computeStateDistributionStatistics(
        model, targets_all, predictions_all)

    fields_2_save_2_logfile = [
        "time_total_per_iter",
    ]
    fields_2_save_2_logfile += list(error_dict_avg.keys())

    results = {
        "fields_2_save_2_logfile": fields_2_save_2_logfile,
        "predictions_all": predictions_all,
        "targets_all": targets_all,
        "latent_states_all": latent_states_all,
        "predictions_augmented_all": predictions_augmented_all,
        "targets_augmented_all": targets_augmented_all,
        "latent_states_augmented_all": latent_states_augmented_all,
        "n_warmup": model.n_warmup,
        "testing_mode": testing_mode,
        "dt": dt,
        "time_total_per_iter": time_total_per_iter,
        "ic_indexes": ic_indexes,
    }
    results = {
        **results,
        **additional_results_dict,
        **error_dict,
        **error_dict_avg,
    }
    return results


def addResultsAutoencoder(
    model,
    outputs_all,
    inputs_all,
    latent_states_all,
    dt,
    error_dict_avg,
    latent_states_all_data=None,
):

    # Computing additional errors based on all predictions (e.g. frequency spectra)
    additional_results_dict, additional_errors_dict = utils_processing.computeAdditionalResults(
        model, outputs_all, inputs_all, dt)

    if latent_states_all_data == None:
        latent_states_all_data = latent_states_all
    latent_state_info = computeLatentStateInfo(model, latent_states_all_data)

    results = {
        "dt": dt,
        "latent_states_all": latent_states_all,
        "outputs_all": outputs_all,
        "inputs_all": inputs_all,
        "latent_state_info": latent_state_info,
        "fields_2_save_2_logfile": [],
    }
    results["fields_2_save_2_logfile"] += list(error_dict_avg.keys())
    results = {
        **results,
        **error_dict_avg,
        **additional_results_dict,
        **additional_errors_dict
    }

    state_statistics = utils_processing.computeStateDistributionStatistics(
        model, inputs_all, outputs_all)

    results = {**results, **state_statistics}
    return results


def getErrorDictAvg(error_dict):
    # Computing the average over time
    error_dict_avg = {}
    for key in error_dict:
        error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
    utils_processing.printErrors(error_dict_avg)
    return error_dict_avg


def makeDirectories(model):
    os.makedirs(getModelDir(model), exist_ok=True)
    os.makedirs(getFigureDir(model), exist_ok=True)
    os.makedirs(getResultsDir(model), exist_ok=True)
    os.makedirs(getLogFileDir(model), exist_ok=True)
    return 0


def getModelDir(model):
    model_dir = model.saving_path + model.model_dir + model.model_name
    return model_dir


def getFigureDir(model, unformatted=False):
    fig_dir = model.saving_path + model.fig_dir + model.model_name
    return fig_dir


def getResultsDir(model, unformatted=False):
    results_dir = model.saving_path + model.results_dir + model.model_name
    return results_dir


def getLogFileDir(model, unformatted=False):
    logfile_dir = model.saving_path + model.logfile_dir + model.model_name
    return logfile_dir


def getChannels(channels, params):
    if channels == 0:
        Dz, Dy, Dx = 0, 0, 1
    elif channels == 1:
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = 0, 0, params["Dx"]
    elif channels == 2:
        assert (params["Dy"] > 0)
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = 0, params["Dy"], params["Dx"]
    elif channels == 3:
        assert (params["Dz"] > 0)
        assert (params["Dy"] > 0)
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = params["Dz"], params["Dy"], params["Dx"]
    return Dz, Dy, Dx


def processList(var):
    # A function that processes a list [1,2,3] and returns a string 1-2-3
    if isinstance(var, list):
        var_ = ""
        len_ = len(var)
        for i in range(len_):
            element = var[i]
            var_ += str(element)
            if i < len(var) - 1: var_ += "-"
    else:
        var_ = var
    return var_


# def fileHasEnding(filename, ending):
#     # First split from .ending
#     filename = filename.split(".")[0]
#     endingTrue = filename[-len(ending):]
#     temp = endingTrue == ending
#     return temp

# def clearDirectory(self, folder, filepatternending):
#     import os, shutil
#     for filename in os.listdir(folder):
#         if self.fileHasEnding(filename, filepatternending):
#             file_path = os.path.join(folder, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))
#     return 0
