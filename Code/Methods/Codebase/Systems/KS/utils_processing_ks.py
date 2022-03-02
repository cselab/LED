#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from ... import Utils as utils

import matplotlib
import matplotlib.pyplot as plt


def addResultsSystemKS(model, results, statistics):

    results["fields_2_save_2_logfile"].append("ux_uxx_l1_hist_error")
    results["fields_2_save_2_logfile"].append("ux_uxx_wasserstein_distance")

    results["fields_2_save_2_logfile"].append("L1_hist_error_mean")
    results["fields_2_save_2_logfile"].append("wasserstein_distance_mean")

    return results


def computeStateDistributionStatisticsSystemKS(state_dist_statistics,
                                               targets_all, predictions_all):
    n_ics, T, K, D = np.shape(targets_all)

    L1_hist_error = []
    wasserstein_distance = []
    KS_error = []
    for n in range(n_ics):
        predictions_ic = predictions_all[n]
        targets_ic = targets_all[n]

        predictions_ = np.reshape(predictions_ic, (-1))
        targets_ = np.reshape(targets_ic, (-1))
        N_samples = np.shape(predictions_)[0]
        min_ = np.min([np.min(targets_), np.min(predictions_)])
        max_ = np.max([np.max(targets_), np.max(predictions_)])
        bounds = [min_, max_]
        LL = max_ - min_
        nbins = utils.getNumberOfBins(N_samples, LL)
        # L1_hist_error, error_vec, density_target, density_pred, bin_centers = evaluateL1HistErrorVector(targets_, predictions_, nbins, bounds)
        hist_data = utils.evaluateL1HistErrorVector(targets_, predictions_,
                                                    nbins, bounds)
        L1_hist_error_ic = hist_data[0]
        wasserstein_distance_ic = utils.evaluateWassersteinDistance(
            targets_, predictions_)
        KS_error_ic = utils.evaluateKSError(targets_, predictions_)

        L1_hist_error.append(L1_hist_error_ic)
        wasserstein_distance.append(wasserstein_distance_ic)
        KS_error.append(KS_error_ic)

    L1_hist_error = np.array(L1_hist_error)
    wasserstein_distance = np.array(wasserstein_distance)
    KS_error = np.array(KS_error)

    L1_hist_error_mean = np.mean(L1_hist_error)
    wasserstein_distance_mean = np.mean(wasserstein_distance)
    KS_error_mean = np.mean(KS_error)

    print("[utils_processing_ks] Wasserstein distance = {:}".format(
        wasserstein_distance_mean))
    print("[utils_processing_ks] KS_error = {:}".format(KS_error_mean))
    print(
        "[utils_processing_ks] L1_hist_error = {:}".format(L1_hist_error_mean))
    state_dist_statistics.update({
        # "state_dist_hist_data": hist_data,
        "L1_hist_error_mean": L1_hist_error_mean,
        "wasserstein_distance_mean": wasserstein_distance_mean,
        "state_dist_L1_hist_error": L1_hist_error,
        "state_dist_wasserstein_distance": wasserstein_distance,
        "state_dist_KS_error": KS_error,
        # "state_dist_N": N_samples,
        # "state_dist_LL": LL,
        # "state_dist_bounds": bounds,
    })

    return state_dist_statistics


def computeStateDistributionStatisticsSystemKSUxUxx(state_dist_statistics,
                                                    targets_all,
                                                    predictions_all,
                                                    rule="sturges",
                                                    nbins=None):
    print("# computeStateDistributionStatisticsSystemKSUxUxx() #")
    n_ics, T, K, D = np.shape(targets_all)
    assert K == 1
    targets_ = np.reshape(targets_all, (n_ics * T, D))
    predictions_ = np.reshape(predictions_all, (n_ics * T, D))
    L = 22
    D = 64
    dx = L / (D - 1)
    targets_ux = np.diff(targets_, axis=1) / dx
    targets_uxx = np.diff(targets_ux, axis=1) / dx
    targets_ux = targets_ux[:, :-1]
    targets_uxx = np.reshape(targets_uxx, (-1))
    targets_ux = np.reshape(targets_ux, (-1))

    predictions_ux = np.diff(predictions_, axis=1) / dx
    predictions_uxx = np.diff(predictions_ux, axis=1) / dx
    predictions_ux = predictions_ux[:, :-1]
    predictions_uxx = np.reshape(predictions_uxx, (-1))
    predictions_ux = np.reshape(predictions_ux, (-1))

    num_samples = np.shape(targets_ux)[0]

    if nbins == None: nbins = utils.getNumberOfBins(num_samples, rule=rule)

    # plt.hist2d(targets_ux, targets_uxx, (nbins,nbins), cmap=plt.get_cmap("Reds"))
    # plt.colorbar()
    # plt.show()
    # plt.hist2d(predictions_ux, predictions_uxx, (nbins,nbins), cmap=plt.get_cmap("Reds"))
    # plt.colorbar()
    # plt.show()

    data1 = np.concatenate((targets_ux[np.newaxis], targets_uxx[np.newaxis]),
                           axis=0).T
    data2 = np.concatenate(
        (predictions_ux[np.newaxis], predictions_uxx[np.newaxis]), axis=0).T
    ux_uxx_bounds = [[np.min(data1[:, 0]),
                      np.max(data1[:, 0])],
                     [np.min(data1[:, 1]),
                      np.max(data1[:, 1])]]
    ux_uxx_l1_hist_error, ux_uxx_l1_hist_error_vec, ux_uxx_density_target, ux_uxx_density_predicted, ux_uxx_bin_centers = utils.evaluateL1HistErrorVector(
        data1, data2, nbins, ux_uxx_bounds)
    ux_uxx_wasserstein_distance = utils.evaluateWassersteinDistance(
        data1, data2)
    # print("ux_uxx_l1_hist_error = {:}".format(ux_uxx_l1_hist_error))
    # print("ux_uxx_wasserstein_distance = {:}".format(ux_uxx_wasserstein_distance))
    ux_uxx_mesh = np.meshgrid(ux_uxx_bin_centers[0], ux_uxx_bin_centers[1])

    assert ("ux_uxx_density_target" not in state_dist_statistics)
    assert ("ux_uxx_density_predicted" not in state_dist_statistics)
    assert ("ux_uxx_l1_hist_error" not in state_dist_statistics)
    assert ("ux_uxx_l1_hist_error_vec" not in state_dist_statistics)
    assert ("ux_uxx_wasserstein_distance" not in state_dist_statistics)
    assert ("ux_uxx_mesh" not in state_dist_statistics)
    state_dist_statistics.update({
        "ux_uxx_density_target": ux_uxx_density_target,
        "ux_uxx_density_predicted": ux_uxx_density_predicted,
        "ux_uxx_l1_hist_error": ux_uxx_l1_hist_error,
        "ux_uxx_l1_hist_error_vec": ux_uxx_l1_hist_error_vec,
        "ux_uxx_wasserstein_distance": ux_uxx_wasserstein_distance,
        "ux_uxx_mesh": ux_uxx_mesh,
    })
    return state_dist_statistics
