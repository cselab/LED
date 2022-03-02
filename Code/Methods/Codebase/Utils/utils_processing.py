#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
import hickle as hkl
import io
import os
import time
import torch
import warnings

######################################################
#""" Utilities """
######################################################
from . import utils_time
from . import utils_data
from . import utils_statistics
from . import utils_metrics


def writeToLogFile(model, logfile, data, fields_to_write):
    with io.open(logfile, 'a+') as f:
        f.write("model_name:" + str(model.model_name))
        for field in fields_to_write:
            if (field not in data):
                raise ValueError("Field {:} is not in data.".format(field))
            f.write(":{:}:{:}".format(field, data[field]))
        f.write("\n")
    return 0


def removeDuplicates(filename):
    # Remove duplicates
    uniqlines = set(open(filename).readlines())
    with open(filename, "w") as file:
        for line in uniqlines:
            file.write(line)
    return 0


def getReferenceTrainingTime(rtt, btt):
    reference_train_time = 60 * 60 * (rtt - btt)
    print("[utils_processing] Reference train time:")
    print("[utils_processing] " +
          utils_time.secondsToTimeStr(reference_train_time))
    return reference_train_time


def getErrorLabelsDict(model):
    # error_dict = {
    #     "MSE": [],
    #     "RMSE": [],
    #     "ABS": [],
    #     "PSNR": [],
    #     "SSIM": [],
    # }
    error_dict = {}
    for key in model.data_info_dict["errors_to_compute"]:
        error_dict[key] = []
    return error_dict


def computeErrors(target, prediction, data_info, single_sample=False):
    assert "errors_to_compute" in data_info
    errors_to_compute = data_info["errors_to_compute"]
    if single_sample:
        spatial_dims = tuple([*range(len(np.shape(target)))])
    else:
        spatial_dims = tuple([*range(len(np.shape(target)))[1:]])
    # ABSOLUTE ERROR
    abserror = np.abs(target - prediction)
    abserror = np.mean(abserror, axis=spatial_dims)
    # SQUARE ERROR
    serror = np.square(target - prediction)
    # MEAN (over-space) SQUARE ERROR
    mse = np.mean(serror, axis=spatial_dims)
    # ROOT MEAN SQUARE ERROR
    rmse = np.sqrt(mse)

    error_dict = {}

    if "CORR" in errors_to_compute:
        if single_sample:
            corr = np.corrcoef(np.reshape(target, (-1)),
                               np.reshape(prediction, (-1)))[0, 1]
        else:
            corr = np.array([
                np.corrcoef(np.reshape(target[t], (-1)),
                            np.reshape(prediction[t], (-1)))[0, 1]
                for t in range(len(target))
            ])

        error_dict["CORR"] = corr

    if "MSE" in errors_to_compute:
        error_dict["MSE"] = mse

    if "RMSE" in errors_to_compute:
        error_dict["RMSE"] = rmse

    if "NNAD" in errors_to_compute:
        assert "data_std" in data_info, "ERROR: data_std needed to compute the NNAD not found in the data_info_dict."
        assert "data_max" in data_info, "ERROR: data_max needed to compute the NNAD not found in the data_info_dict."
        assert "data_min" in data_info, "ERROR: data_min needed to compute the NNAD not found in the data_info_dict."
        data_std = data_info["data_std"]
        data_max = data_info["data_max"]
        data_min = data_info["data_min"]

        data_norm = data_max - data_min
        # print(np.shape(rmse))
        # print(np.shape(mse))
        # print(np.shape(mse))
        # print(np.shape(serror))
        temp = len(np.shape(target))
        num_channels = temp-1 if single_sample else temp-2
        for i in range(num_channels):
            data_norm = np.expand_dims(data_norm, 1)

        # print(single_sample)
        # print(num_channels)
        if not single_sample: data_norm = data_norm[np.newaxis] # Adding the ic axis
        # print(np.shape(data_norm))
        # print(np.shape(serror))
        nrmse = np.sqrt(serror / np.power(data_norm, 2.0))
        nrmse = np.mean(nrmse, axis=spatial_dims)
        error_dict["NNAD"] = nrmse
        # print(ark)
    if "ABS" in errors_to_compute:
        error_dict["ABS"] = abserror

    if "NAD" in errors_to_compute:
        nad_error = np.mean(np.abs(target - prediction) /
                            (np.max(target) - np.min(target)),
                            axis=spatial_dims)
        error_dict["NAD"] = nad_error

    if "PSNR" in errors_to_compute:
        if single_sample:
            psnr = utils_metrics.PSNR(target, prediction)
        else:
            psnr = np.array([
                utils_metrics.PSNR(target[i], prediction[i])
                for i in range(np.shape(target)[0])
            ])
        error_dict["PSNR"] = psnr

    if "SSIM" in errors_to_compute:
        if single_sample:
            ssim = utils_metrics.SSIM(target, prediction)
        else:
            ssim = np.array([
                utils_metrics.SSIM(target[i], prediction[i])
                for i in range(np.shape(target)[0])
            ])
        error_dict["SSIM"] = ssim
    return error_dict


def comptuteErrorsInTime(model, targets_all, predictions_all):
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dx]
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dy, Dx] OR
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dz, Dy, Dx]

    if model.data_info_dict["structured"]:
        T = np.shape(targets_all)[1]
        N_ICS = np.shape(targets_all)[0]
        error_mean_dict_in_time = getErrorLabelsDict(model)
        error_std_dict_in_time = getErrorLabelsDict(model)
        for t in range(T):

            error_dict = getErrorLabelsDict(model)
            for ic in range(N_ICS):
                target_path = targets_all[ic][t]
                prediction_path = predictions_all[ic][t]

                target = utils_data.getDataHDF5Field(target_path[0],
                                                     target_path[1])
                prediction = utils_data.getDataHDF5Field(
                    prediction_path[0], prediction_path[1])

                error_dict_t_ic = computeErrors(
                    target,
                    prediction,
                    model.data_info_dict,
                    single_sample=True)
                for key in error_dict_t_ic:
                    error_dict[key].append(error_dict_t_ic[key])

            # Mean over initial conditions
            error_mean_dict = {}
            error_std_dict = {}

            for key in error_dict:
                error_mean_dict_in_time[key].append(
                    np.mean(np.array(error_dict[key])))
                error_std_dict_in_time[key].append(
                    np.std(np.array(error_dict[key])))
    else:
        targets_all = np.swapaxes(targets_all, 0, 1)
        predictions_all = np.swapaxes(predictions_all, 0, 1)
        # print(np.shape(targets_all))
        # T, N_ICS, input_dim, Dx, Dy = np.shape(targets_all)

        T = np.shape(targets_all)[0]
        N_ICS = np.shape(targets_all)[1]
        input_dim = np.shape(targets_all)[2]
        error_mean_dict_in_time = getErrorLabelsDict(model)
        error_std_dict_in_time = getErrorLabelsDict(model)

        for t in range(T):
            target = targets_all[t]
            prediction = predictions_all[t]
            error_dict = computeErrors(
                target, prediction, model.data_info_dict)
            # Mean over initial conditions
            error_mean_dict = {}
            error_std_dict = {}
            for key in error_dict:
                error_mean_dict_in_time[key].append(
                    np.mean(np.array(error_dict[key])))
                error_std_dict_in_time[key].append(
                    np.std(np.array(error_dict[key])))
    return error_mean_dict_in_time, error_std_dict_in_time


# def computeStateDistributionStatisticsPerStateOutput(state_dist_statistics,
#                                                      targets_all,
#                                                      predictions_all):

#     assert (len(np.shape(targets_all)) == 3)
#     n_ics, T, N = np.shape(targets_all)
#     targets_all = np.reshape(targets_all, (-1, *targets_all.shape[-1:]))
#     predictions_all = np.reshape(predictions_all,
#                                  (-1, *targets_all.shape[-1:]))
#     l1_hist_errors, hist_data, wasserstein_distance_data, KS_errors, bounds, LL, nbins = [], [], [], [], [], [], []
#     for n in range(N):
#         target_data = targets_all[:, n]
#         output_data = predictions_all[:, n]
#         # min_ = np.min(target_data)
#         # max_ = np.max(target_data)

#         min_ = np.min([np.min(target_data), np.min(output_data)])
#         max_ = np.max([np.max(target_data), np.max(output_data)])
#         bounds_ = [min_, max_]
#         # print("Bounds of state {:} = {:}".format(n, bounds_))
#         LL_ = max_ - min_
#         N_samples = np.shape(target_data)[0]
#         nbins_ = utils_statistics.getNumberOfBins(N_samples, LL_)
#         # print("Number of bins = {:}".format(nbins_))
#         # L1_hist_error, error_vec, density_target_, density_pred_, bin_centers_ = utils_statistics.evaluateL1HistErrorVector(target_data, output_data, nbins_, bounds_)
#         hist_data_ = utils_statistics.evaluateL1HistErrorVector(
#             target_data, output_data, nbins_, bounds_)
#         wasserstein_distances_ = utils_statistics.evaluateWassersteinDistance(
#             target_data, output_data)
#         KS_error = utils_statistics.evaluateKSError(target_data, output_data)

#         L1_hist_error = hist_data_[0]

#         KS_errors.append(KS_error)
#         l1_hist_errors.append(L1_hist_error)
#         wasserstein_distance_data.append(wasserstein_distances_)
#         hist_data.append(hist_data_)
#         LL.append(LL_)
#         nbins.append(nbins_)
#         bounds.append(bounds_)
#     KS_error = np.mean(KS_errors)
#     L1_hist_error = np.mean(l1_hist_errors)
#     wasserstein_distance = np.mean(wasserstein_distance_data)
#     print("[utils_processing] Wasserstein distance = {:}".format(wasserstein_distance))
#     print("[utils_processing] KS_error = {:}".format(KS_error))
#     print("[utils_processing] L1_hist_error = {:}".format(L1_hist_error))
#     state_dist_statistics.update({
#         "state_dist_L1_hist_error": L1_hist_error,
#         "state_dist_hist_data": hist_data,
#         "state_dist_wasserstein_distance": wasserstein_distance,
#         "state_dist_wasserstein_distance_data": wasserstein_distance_data,
#         "state_dist_KS_error": KS_error,
#         "state_dist_N": N_samples,
#         "state_dist_LL": LL,
#         "state_dist_nbins": nbins,
#         "state_dist_bounds": bounds,
#     })
#     return state_dist_statistics

# def computeStateDistributionStatisticsCumulative(state_dist_statistics, targets_all, predictions_all):
#     predictions_ = np.reshape(predictions_all, (-1))
#     targets_ = np.reshape(targets_all, (-1))
#     N_samples = np.shape(predictions_)[0]
#     min_ = np.min([np.min(targets_), np.min(predictions_)])
#     max_ = np.max([np.max(targets_), np.max(predictions_)])
#     bounds = [min_, max_]
#     LL = max_ - min_
#     nbins = utils_statistics.getNumberOfBins(N_samples, LL)
#     # L1_hist_error, error_vec, density_target, density_pred, bin_centers = evaluateL1HistErrorVector(targets_, predictions_, nbins, bounds)
#     hist_data = utils_statistics.evaluateL1HistErrorVector(
#         targets_, predictions_, nbins, bounds)
#     L1_hist_error = hist_data[0]
#     wasserstein_distance = utils_statistics.evaluateWassersteinDistance(
#         targets_, predictions_)
#     KS_error = utils_statistics.evaluateKSError(targets_, predictions_)
#     print("[utils_processing] Wasserstein distance = {:}".format(wasserstein_distance))
#     print("[utils_processing] KS_error = {:}".format(KS_error))
#     print("[utils_processing] L1_hist_error = {:}".format(L1_hist_error))
#     state_dist_statistics.update({
#         "state_dist_hist_data": hist_data,
#         "state_dist_L1_hist_error": L1_hist_error,
#         "state_dist_wasserstein_distance": wasserstein_distance,
#         "state_dist_KS_error": KS_error,
#         "state_dist_N": N_samples,
#         "state_dist_LL": LL,
#         "state_dist_bounds": bounds,
#     })
#     return state_dist_statistics


def computeStateDistributionStatistics(model, targets_all, predictions_all):
    # Computing statistical errors on distributions.
    # In case on output states that are all the same (Kuramoto-Sivashinsky) the state distribution is calculated with respect to all states.
    # In case on outputs that are different (Alanine), the state distributions are calculated with respect to each output separately.
    state_dist_statistics = {}

    if model.data_info_dict["structured"]:
        warnings.warn(
            "[computeStateDistributionStatistics()] Not implemented for structured data."
        )
        return state_dist_statistics
        # raise ValueError("Not implemented.")

    # if model.data_info_dict["statistics_per_state"]:
    #     state_dist_statistics = computeStateDistributionStatisticsPerStateOutput(
    #         state_dist_statistics, targets_all, predictions_all)

    # elif model.data_info_dict["statistics_cummulative"]:
    #     state_dist_statistics = computeStateDistributionStatisticsCumulative(
    #         state_dist_statistics, targets_all, predictions_all)

    return state_dist_statistics


def printErrors(error_dict):
    print("[utils_processing] " + "_" * 30)
    for key in error_dict:
        print("[utils_processing] " + "{:} = {:}".format(key, error_dict[key]))
    print("[utils_processing] " + "_" * 30)
    return 0


def computeAdditionalResults(model, predictions_all, targets_all, dt):
    additional_errors_dict = {}
    additional_results_dict = {}

    if model.params["compute_spectrum"]:
        if model.data_info_dict["structured"]:
            raise ValueError("Not implemented.")

        freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(
            predictions_all, targets_all, dt)
        additional_results_dict["freq_pred"] = freq_pred
        additional_results_dict["freq_true"] = freq_true
        additional_results_dict["sp_true"] = sp_true
        additional_results_dict["sp_pred"] = sp_pred
        additional_errors_dict["error_freq"] = error_freq

    if model.data_info_dict["compute_errors_in_time"]:

        error_mean_dict_in_time, error_std_dict_in_time = comptuteErrorsInTime(
            model, predictions_all, targets_all)
        additional_results_dict[
            "error_mean_dict_in_time"] = error_mean_dict_in_time
        additional_results_dict[
            "error_std_dict_in_time"] = error_std_dict_in_time

    return additional_results_dict, additional_errors_dict


def postprocessComputeStateDensitiesSingleTimeInstant(model,
                                                      results,
                                                      autoencoder=True):
    ##############################################
    ### Post-processing single sequence results
    ### ONLY for plotting purposes !
    ##############################################
    print(
        "[utils_processing] # postprocessComputeStateDensitiesSingleTimeInstant() #"
    )
    if autoencoder:
        assert ("dt" in results)
        assert ("latent_states_all" in results)
        assert ("inputs_all" in results)
        assert ("outputs_all" in results)
        # Taking into account only one initial condition
        target = results["inputs_all"][0]
        output = results["outputs_all"][0]
    else:
        target = results["Reference"]
        output = results["prediction"]

    if model.data_info_dict["density_plots"]:
        N_samples = np.shape(target)[1]
        print("Number of samples {:}".format(N_samples))
        max_target = np.amax(target)
        min_target = np.amin(target)
        bounds = [min_target, max_target]
        print("bounds {:}".format(bounds))
        LL = bounds[1] - bounds[0]
        nbins = utils_statistics.getNumberOfBins(N_samples, LL)

        hist_data = np.array([
            utils_statistics.evaluateL1HistErrorVector(target[t], output[t],
                                                       nbins, bounds)
            for t in range(target.shape[0])
        ])
        hist_data = np.stack(hist_data)

        error_density_total = np.array([temp[0] for temp in hist_data])
        error_density = np.array([temp[1] for temp in hist_data])
        density_target = np.array([temp[2] for temp in hist_data])
        density_output = np.array([temp[3] for temp in hist_data])
        # Mean error density over time
        error_density_mean = np.mean(error_density_total)

        wasserstein_distance_data = np.array([
            utils_statistics.evaluateWassersteinDistance(target[t], output[t])
            for t in range(target.shape[0])
        ])
        wasserstein_distance_mean = np.mean(wasserstein_distance_data)
        results["fields_2_save_2_logfile"] += [
            "error_density_mean",
            "wasserstein_distance_mean",
        ]
    else:
        # No statistics are needed
        N_samples, bounds, nbins, LL, hist_data, wasserstein_distance_data, wasserstein_distance_mean, error_density_total, error_density_mean = None, None, None, None, None, None, None, None, None,
    results.update({
        "N": N_samples,
        "bounds": bounds,
        "nbins": nbins,
        "LL": LL,
        "hist_data": hist_data,
        "wasserstein_distance_data": wasserstein_distance_data,
        "wasserstein_distance_mean": wasserstein_distance_mean,
        "error_density_total": error_density_total,
        "error_density_mean": error_density_mean,
    })
    return results


def computeFrequencyError(predictions_all, targets_all, dt):
    spatial_dims = len(np.shape(predictions_all)[2:])
    # print(spatial_dims)
    if spatial_dims == 1:
        sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
        sp_true, freq_true = computeSpectrum(targets_all, dt)
        # s_dbfs = 20 * np.log10(s_mag)
        # TRANSFORM TO AMPLITUDE FROM DB
        sp_pred = np.exp(sp_pred / 20.0)
        sp_true = np.exp(sp_true / 20.0)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return freq_pred, freq_true, sp_true, sp_pred, error_freq
    elif spatial_dims == 3:
        # RGB Image channells (Dz) of Dx x Dy
        # Applying two dimensional FFT
        sp_true = computeSpectrum2D(targets_all)
        sp_pred = computeSpectrum2D(predictions_all)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return None, None, sp_pred, sp_true, error_freq
    elif spatial_dims == 2:
        nics, T, n_o, Dx = np.shape(predictions_all)
        predictions_all = np.reshape(predictions_all, (nics, T, n_o * Dx))
        targets_all = np.reshape(targets_all, (nics, T, n_o * Dx))
        sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
        sp_true, freq_true = computeSpectrum(targets_all, dt)
        # s_dbfs = 20 * np.log10(s_mag)
        # TRANSFORM TO AMPLITUDE FROM DB
        sp_pred = np.exp(sp_pred / 20.0)
        sp_true = np.exp(sp_true / 20.0)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return freq_pred, freq_true, sp_true, sp_pred, error_freq
    else:
        raise ValueError(
            "Not implemented. Shape of predictions_all is {:}, with spatial_dims={:}."
            .format(np.shape(predictions_all), spatial_dims))


def computeSpectrum2D(data_all):
    # Of the form [n_ics, T, n_dim]
    spectrum_db = []
    for data_ic in data_all:
        # data_ic shape = T, 1(Dz), 65(Dx), 65(Dy)
        for data_t in data_ic:
            # Taking accoung only the first channel
            s_dbfs = dbfft2D(data_t[0])
            spectrum_db.append(s_dbfs)
    # MEAN OVER ALL ICS AND ALL TIME-STEPS
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    return spectrum_db


def dbfft2D(x):
    # !! SPATIAL FFT !!
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1, :-1]
        N = len(x)
    x = np.reshape(x, (N, N))
    # Calculate real FFT and frequency vector
    sp = np.fft.fft2(x)
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp)
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    return s_dbfs


def computeSpectrum(data_all, dt):
    # Of the form [n_ics, T, n_dim]
    spectrum_db = []
    for data in data_all:
        data = np.transpose(data)
        for d in data:
            freq, s_dbfs = dbfft(d, 1 / dt)
            spectrum_db.append(s_dbfs)
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    return spectrum_db, freq


def dbfft(x, fs):
    # !!! TIME DOMAIN FFT !!!
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1]
        N = len(x)
    x = np.reshape(x, (1, N))
    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)
    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / N
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    s_dbfs = s_dbfs[0]
    return freq, s_dbfs
