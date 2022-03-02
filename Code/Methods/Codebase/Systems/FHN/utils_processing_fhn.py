#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from ... import Utils as utils


def addResultsSystemFHN(model, results, testing_mode):

    if ("autoencoder" in testing_mode) or ("dimred" in testing_mode):
        targets_all = results["inputs_all"]
        predictions_all = results["outputs_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    if model.data_info_dict["structured"]:
        targets_all = utils.getDataFromHDF5PathArrayMultipleICs(targets_all)
        predictions_all = utils.getDataFromHDF5PathArrayMultipleICs(
            predictions_all)

    # ONLY RELEVANT FOR THE FHN SYSTEM
    targets_all_act = targets_all[:, :, 0]
    predictions_all_act = predictions_all[:, :, 0]

    # print(np.shape(targets_all_act))
    # print(np.shape(predictions_all_act))
    mnad_act = np.mean(np.abs(targets_all_act - predictions_all_act) /
                       (np.max(targets_all_act) - np.min(targets_all_act)),
                       axis=2)

    mnad_act_avg = np.mean(mnad_act)
    print(
        "[utils_processing_fhn] (MNAD) Mean normalised absolute difference on the activator density: {:}"
        .format(mnad_act_avg))

    targets_all_in = targets_all[:, :, 1]
    predictions_all_in = predictions_all[:, :, 1]

    # print(np.shape(targets_all_in))
    # print(np.shape(predictions_all_in))
    mnad_in = np.mean(np.abs(targets_all_in - predictions_all_in) /
                      (np.max(targets_all_in) - np.min(targets_all_in)),
                      axis=2)

    mnad_in_avg = np.mean(mnad_in)
    print(
        "[utils_processing_fhn] (MNAD) Mean normalised absolute difference on the inhibitor density: {:}"
        .format(mnad_in_avg))

    # Adding the computed results
    results["fields_2_save_2_logfile"].append("mnad_act_avg")
    results["fields_2_save_2_logfile"].append("mnad_in_avg")
    results.update({
        "mnad_act": mnad_act,
        "mnad_in": mnad_in,
        "mnad_act_avg": mnad_act_avg,
        "mnad_in_avg": mnad_in_avg,
    })
    return results
