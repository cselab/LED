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


def predictIndexes(mclass, data_loader, dt, set_name, testing_mode):
    print("[utils_multiscale_unsctructured] # predictIndexes() #")
    model = mclass.model

    num_test_ICS = model.num_test_ICS

    predictions_all = []
    targets_all = []
    latent_states_all = []

    predictions_augmented_all = []
    targets_augmented_all = []
    latent_states_augmented_all = []

    time_total_per_iter_all = []

    # Dictionary of error lists
    error_dict = utils.getErrorLabelsDict(model)

    if num_test_ICS > len(data_loader):
        num_test_ICS = len(data_loader)
        # raise ValueError("Not enough ICs in the dataset {:}.".format(set_name))

    print(
        "[utils_multiscale_unsctructured] # predictIndexes() on {:}/{:} initial conditions."
        .format(num_test_ICS, len(data_loader)))

    ic_num = 1
    ic_indexes = []
    for sequence in data_loader:
        if ic_num > num_test_ICS: break
        if model.display_output:
            print(
                "[utils_multiscale_unsctructured] IC {:}/{:}, {:2.3f}%".format(
                    ic_num, num_test_ICS, ic_num / num_test_ICS * 100))
        sequence = sequence[0]

        # STARTING TO PREDICT THE SEQUENCE IN model.predict_on=model.sequence_length
        # Warming-up with sequence_length
        model.predict_on = model.n_warmup
        assert (model.predict_on - model.n_warmup >= 0)

        if model.predict_on + model.prediction_horizon > np.shape(sequence)[0]:
            prediction_horizon = np.shape(sequence)[0] - model.predict_on
            warnings.warn(
                "[utils_multiscale_unsctructured] model.predict_on ({:}) + model.prediction_horizon ({:}) > np.shape(sequence)[0] ({:}). Not enough timesteps in the {:} data. Using a prediction horizon of {:}."
                .format(model.predict_on, model.prediction_horizon,
                        np.shape(sequence)[0], set_name, prediction_horizon))
        else:
            prediction_horizon = model.prediction_horizon

        # assert model.predict_on + model.prediction_horizon <= np.shape(sequence)[0], "model.predict_on ({:}) + model.prediction_horizon ({:}) > np.shape(sequence)[0] ({:}). Not enough timesteps in the {:} data.".format(model.predict_on, model.prediction_horizon, np.shape(sequence)[0], set_name)
        # assert model.predict_on - model.n_warmup >= 0
        # assert model.predict_on <= np.shape(sequence)[0]

        sequence = sequence[model.predict_on -
                            model.n_warmup:model.predict_on +
                            prediction_horizon]

        prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter = predictSequence(
            mclass,
            sequence,
            testing_mode,
            dt,
            prediction_horizon=prediction_horizon)

        prediction = model.data_info_dict["scaler"].descaleData(
            prediction,
            single_sequence=True,
            check_bounds=False,
            verbose=False)
        target = model.data_info_dict["scaler"].descaleData(
            target, single_sequence=True, check_bounds=False, verbose=False)

        prediction_augment = model.data_info_dict["scaler"].descaleData(
            prediction_augment,
            single_sequence=True,
            check_bounds=False,
            verbose=False)
        target_augment = model.data_info_dict["scaler"].descaleData(
            target_augment,
            single_sequence=True,
            check_bounds=False,
            verbose=False)

        errors = utils.computeErrors(target, prediction,
                                     model.data_info_dict)
        # Updating the error
        for error in errors:
            error_dict[error].append(errors[error])

        latent_states_all.append(latent_states)
        predictions_all.append(prediction)
        targets_all.append(target)

        latent_states_augmented_all.append(latent_states_augmented)
        predictions_augmented_all.append(prediction_augment)
        targets_augmented_all.append(target_augment)

        time_total_per_iter_all.append(time_total_per_iter)
        ic_indexes.append(ic_num)
        ic_num += 1

    time_total_per_iter_all = np.array(time_total_per_iter_all)
    time_total_per_iter = np.mean(time_total_per_iter_all)

    predictions_all = np.array(predictions_all)
    targets_all = np.array(targets_all)
    latent_states_all = np.array(latent_states_all)

    predictions_augmented_all = np.array(predictions_augmented_all)
    targets_augmented_all = np.array(targets_augmented_all)
    latent_states_augmented_all = np.array(latent_states_augmented_all)

    print("[utils_multiscale_unsctructured] Shape of trajectories:")
    print("[utils_multiscale_unsctructured] {:}:".format(
        np.shape(targets_all)))
    print("[utils_multiscale_unsctructured] {:}:".format(
        np.shape(predictions_all)))

    # Computing the average over time
    error_dict_avg = {}
    for key in error_dict:
        # print(np.shape(error_dict[key]))
        error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
    utils.printErrors(error_dict_avg)

    # Computing additional errors based on all predictions (e.g. frequency spectra)
    additional_results_dict, additional_errors_dict = utils.computeAdditionalResults(
        model, predictions_all, targets_all, dt)
    error_dict_avg = {**error_dict_avg, **additional_errors_dict}

    state_statistics = utils.computeStateDistributionStatistics(
        model, targets_all, predictions_all)
    state_statistics = systems.computeStateDistributionStatisticsSystem(
        model, state_statistics, targets_all, predictions_all)

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
        **state_statistics
    }

    results = systems.addResultsSystem(model, results, testing_mode)

    return results


def predictSequence(
    mclass,
    input_sequence,
    testing_mode=None,
    dt=1,
    ic_idx=None,
    set_name=None,
    param=None,
    prediction_horizon=None,
):
    print("[utils_multiscale_unsctructured] # predictSequence() #")
    print("[utils_multiscale_unsctructured] {:}:".format(
        np.shape(input_sequence)))
    model = mclass.model

    if prediction_horizon is None:
        prediction_horizon = model.prediction_horizon

    N = np.shape(input_sequence)[0]
    """ Prediction length """
    if N - model.n_warmup != prediction_horizon:
        raise ValueError(
            "Error! N ({:}) - model.n_warmup ({:}) != prediction_horizon ({:})"
            .format(N, model.n_warmup, prediction_horizon))

    assert model.n_warmup > 1, "Warm up steps cannot be <= 1. Increase the iterative prediction length."

    initial_hidden_states = model.getInitialRNNHiddenState(1)

    warmup_data_input = input_sequence[:model.n_warmup - 1]
    warmup_data_input = warmup_data_input[np.newaxis, :]

    warmup_data_target = input_sequence[1:model.n_warmup]
    warmup_data_target = warmup_data_target[np.newaxis, :]

    target = input_sequence[model.n_warmup:model.n_warmup + prediction_horizon]

    if torch.is_tensor(target): target = target.detach().cpu().numpy()
    if torch.is_tensor(target):
        warmup_data_target = warmup_data_target.detach().cpu().numpy()

    warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _ = model.forward(
        warmup_data_input, initial_hidden_states)

    # elif model_class in ["cnn_sindy"]:

    #     if model.n_warmup > 1:
    #         warmup_latent_states = model.model_autoencoder.forwardEncoder(warmup_data_input)
    #         warmup_data_output = model.model_autoencoder.forwardDecoder(warmup_latent_states)
    #         latent_states_pred = warmup_latent_states
    #         last_hidden_state = []
    #     else:
    #         pass

    # elif model_class in ["dimred_rc"]:
    #     if model.n_warmup > 1:
    #         warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _ = model.forward(warmup_data_input, initial_hidden_states)
    #     else:
    #         # In case of predictor with n_warmup=1 (no warmup)
    #         # assert(model.has_predictor)
    #         last_hidden_state = initial_hidden_states

    # print(np.shape(warmup_data_target))
    # print(np.shape(warmup_data_output))

    # print(np.shape(warmup_latent_states))
    # print(np.shape(latent_states_pred))
    """ Multiscale forecasting """
    iterative_propagation_is_latent = True
    input_latent = latent_states_pred[:, -1:, :]
    input_t = input_latent

    time_start = time.time()

    multiscale_rounds, macro_steps_per_round, micro_steps_per_round, _, _ = mclass.getMultiscaleParams(
        testing_mode, prediction_horizon)
    # print("[utils_multiscale_unsctructured] Macroscale timesteps per round: {:}, Microscale timesteps per round: {:}".format(macro_steps_per_round, micro_steps_per_round))
    # iterative_predictions_per_round = multiscale_macro_steps + multiscale_micro_steps

    # model.multiscale_rounds = multiscale_rounds

    prediction = []

    time_dynamics = 0.0
    time_latent_prop = 0.0

    for round_ in range(multiscale_rounds):

        multiscale_macro_steps = macro_steps_per_round[round_]

        if multiscale_macro_steps > 0:

            prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forecast(
                input_t,
                last_hidden_state,
                horizon=multiscale_macro_steps,
            )

            # elif model_class in ["dimred_rc"]:
            #     prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forecast(
            #         input_t,
            #         last_hidden_state,
            #         multiscale_macro_steps,
            #         )

            # elif model_class in ["cnn_sindy"]:

            #     prediction_model_dyn, latent_states_pred, time_latent_prop_t = model.forecast(
            #         input_t,
            #         multiscale_macro_steps,
            #         )
            #     # latent_states_ = latent_states_pred

            # else:
            #     raise ValueError("Not implemented.")

            time_latent_prop += time_latent_prop_t

            if round_ == 0:
                prediction = prediction_model_dyn
                # latent_states = latent_states_
                latent_states = latent_states_pred
            else:
                prediction = np.concatenate((prediction, prediction_model_dyn),
                                            axis=1)
                latent_states = np.concatenate(
                    (latent_states, latent_states_pred), axis=1)

            # init_state = prediction_model_dyn.cpu().detach().numpy()[-1][-1]
            init_state = prediction_model_dyn[-1][-1]

        # else:
        # raise ValueError("Not know what to do...")
        elif round_ == 0:
            init_state = input_sequence[model.n_warmup - 1]
            # init_state = init_state.cpu().detach().numpy()
            init_state = init_state  #.cpu().detach().numpy()

        else:
            # Correcting the shape
            init_state = prediction_sys_dyn[:, -1:, :]
            init_state = init_state[0, 0]

        if round_ < len(micro_steps_per_round):
            multiscale_micro_steps = micro_steps_per_round[round_]

            init_state = np.reshape(init_state, (1, *np.shape(init_state)))
            # print(np.shape(init_state))
            """ How much time to cover with the fine scale dynamics """
            total_time = multiscale_micro_steps * dt
            """ Upsampling to the high dimensional space and using solver (dynamics) to evolve in time """
            if total_time > 0.0:

                init_state = model.data_info_dict["scaler"].descaleData(
                    init_state,
                    single_sequence=True,
                    check_bounds=False,
                    verbose=False,
                )

                time_dynamics_start = time.time()

                prediction_sys_dyn = systems.evolveSystem(
                    mclass,
                    init_state,
                    total_time,
                    dt,
                    round_=round_,
                    micro_steps=multiscale_micro_steps,
                    macro_steps=multiscale_macro_steps,
                )

                prediction_sys_dyn = model.data_info_dict["scaler"].scaleData(
                    prediction_sys_dyn,
                    single_sequence=True,
                    check_bounds=False,
                )
                """ Scaling back the initial state """
                init_state = model.data_info_dict["scaler"].scaleData(
                    init_state,
                    single_sequence=True,
                    check_bounds=False,
                )
                """ Measuring the time for the dynamics """
                time_dynamics_end = time.time()
                time_dynamics_round = time_dynamics_end - time_dynamics_start
                time_dynamics += time_dynamics_round

                prediction_sys_dyn = prediction_sys_dyn[np.newaxis]
                # prediction_sys_dyn_tensor = utils.transform2Tensor(model, prediction_sys_dyn)
                """ Concatenating the prediction """
                # prediction = prediction_sys_dyn_tensor if (
                #     multiscale_macro_steps == 0
                #     and round_ == 0) else np.concatenate(
                #         (prediction, prediction_sys_dyn_tensor), axis=1)

                prediction = prediction_sys_dyn if (
                    multiscale_macro_steps == 0
                    and round_ == 0) else np.concatenate(
                        (prediction, prediction_sys_dyn), axis=1)

                init_state = np.reshape(init_state, (1, *np.shape(init_state)))
                """
                In order to update the last hidden state,
                we need to feed the high scale dynamics up to input_t
                to the network
                """

                idle_dynamics = prediction_sys_dyn[:, :-1, :].copy()
                idle_dynamics = np.concatenate((init_state, idle_dynamics),
                                               axis=1)
                # idle_dynamics = utils.transform2Tensor(model, idle_dynamics)

                _, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forward(
                    idle_dynamics,
                    last_hidden_state,
                )

                # elif model_class in ["dimred_rc"]:
                #     _, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forward(
                #         idle_dynamics,
                #         last_hidden_state,
                #         )

                # elif model_class in ["cnn_sindy"]:
                #     time_start = time.time()
                #     latent_states_ = model.model_autoencoder.forwardEncoder(idle_dynamics)
                #     time_latent_prop_t = time.time() - time_start
                #     # _ = model.model_autoencoder.forwardDecoder(latent_states_)
                #     latent_states_pred = latent_states_
                # else:
                #     raise ValueError("Not implemented.")

                time_latent_prop += time_latent_prop_t

                if (multiscale_macro_steps == 0 and round_ == 0):
                    latent_states = latent_states_pred
                else:
                    # latent_states = self.concatenateStates(latent_states, latent_states_pred, axis=1)
                    latent_states = np.concatenate(
                        (latent_states, latent_states_pred), axis=1)

            else:
                prediction_sys_dyn = prediction_model_dyn

            if iterative_propagation_is_latent:
                # input_t = latent_states_[:, -1:, :].clone()

                if torch.is_tensor(input_t):
                    raise ValueError("Not supposed to happen.")
                    input_t = latent_states_pred[:, -1:, :].clone()

                input_t = latent_states_pred[:, -1:, :]

            else:
                raise ValueError("Not supposed to happen.")
                # Next to feed, the last predicted state (from the dynamics)
                input_t = prediction_sys_dyn[:, -1:, :].clone()
                # input_t = model.transform2Tensor(input_t)

    time_end = time.time()
    time_total = time_end - time_start
    """
    Correcting the time-measurement in case of evolution of the original system
    (in this case, we do not need to internally propagate and update the latent space of the RNN)
    """
    if "multiscale" in testing_mode:
        if multiscale_macro_steps == 0:
            print(
                "[utils_multiscale_unsctructured] Tracking the time when using the original dynamics..."
            )
            time_total = time_dynamics
        else:
            time_total = time_latent_prop + time_dynamics
    else:
        time_total = time_latent_prop

    time_total_per_iter = time_total / prediction_horizon

    prediction = prediction[0]

    latent_states = latent_states[0]

    prediction = np.array(prediction)
    latent_states = np.array(latent_states)
    target = np.array(target)

    print(
        "[utils_multiscale_unsctructured_unsctructured] Shapes of prediction/target/latent_states = {:}/{:}/{:}"
        .format(
            np.shape(prediction),
            np.shape(target),
            np.shape(latent_states),
        ))

    # print("Min/Max")
    # print("Target:")
    # print(np.max(target[:,0]))
    # print(np.min(target[:,0]))
    # print("Prediction:")
    # print(np.max(prediction[:,0]))
    # print(np.min(prediction[:,0]))
    # # print(ark)

    if model.n_warmup > 1:
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
