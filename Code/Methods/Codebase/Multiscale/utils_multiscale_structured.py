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


def predictIndexesOnStructured(mclass, data_set, dt, set_name, testing_mode):
    model = mclass.model

    print(
        "[utils_multiscale] # predictIndexes() (Memory efficient implementation for structured data) #"
    )
    assert (testing_mode in mclass.getMultiscaleTestingModes())
    if testing_mode not in mclass.getMultiscaleTestingModes():
        raise ValueError("Mode {:} is invalid".format(testing_mode))

    num_test_ICS = model.num_test_ICS
    num_max_ICS = len(data_set)
    if num_test_ICS > num_max_ICS:
        warnings.warn(
            "Not enough ({:}) ICs in the dataset {:}. Using {:} possible ICs.".
            format(num_test_ICS, set_name, num_max_ICS))
        num_test_ICS = num_max_ICS
    """ Maximum prediction horizon """
    n_steps = data_set.seq_paths[0]['num_timesteps']
    prediction_horizon_max = n_steps - model.n_warmup - 1
    prediction_horizon = model.prediction_horizon
    if prediction_horizon > prediction_horizon_max:
        warnings.warn(
            "[utils_multiscale] Prediction horizon ({:}) is larger than the timesteps in the structured dataset  ({:}) - n_warmup ({:}) - 1 = ({:}). Setting prediction_horizon={:}."
            .format(prediction_horizon, n_steps, model.n_warmup,
                    prediction_horizon_max, prediction_horizon_max))
        prediction_horizon = prediction_horizon_max
    assert prediction_horizon > 0, "[utils_multiscale] Prediction horizon needs to be positive. Reduce it."

    ic_indexes = np.arange(num_test_ICS)
    # ic_indexes = np.arange(num_test_ICS) + 1

    predictions_all = []
    targets_all = []
    latent_states_all = []

    predictions_augmented_all = []
    targets_augmented_all = []
    latent_states_augmented_all = []

    # Dictionary of error lists
    error_dict = utils.getErrorLabelsDict(model)

    print(
        "[utils_multiscale] # predictIndexes() on {:}/{:} initial conditions.".
        format(num_test_ICS, num_max_ICS))

    prediction_field_name_in_hdf5 = "{:}_{:}".format(
        testing_mode, set_name) + "_ic{:}_t{:}_prediction"
    target_field_name_in_hdf5 = "{:}_{:}".format(
        testing_mode, set_name) + "_ic{:}_t{:}_target"
    latent_state_field_name_in_hdf5 = "{:}_{:}".format(
        testing_mode, set_name) + "_ic{:}_t{:}_latent"

    # Create .hdf5 file
    results_path_base = utils.getResultsDir(model) + "/{:}_{:}.h5".format(
        testing_mode, set_name)
    utils.deleteFile(results_path_base)

    time_total_per_iter_all = []

    for ic_num in range(len(ic_indexes)):
        ic_index = ic_indexes[ic_num]

        predictions_all_ic = []
        targets_all_ic = []
        latent_states_all_ic = []
        error_dict_ic = utils.getErrorLabelsDict(model)

        input_sequence = []
        # Getting the required samples for the sequence length
        timestep_start = 0
        for t in range(model.n_warmup):
            t_load = timestep_start + t
            t_save = t_load - model.n_warmup
            # print("# loading timestep from data t={:} (warm-up), saving as t={:} (used as input)".format(t_load, t_save))
            sample = data_set.getSequencePart(ic_index, t_load, t_load + 1)
            input_ = sample[0]

            input_sequence.append(input_)
            wfn = "{:}_{:}_ic{:}_t{:}_warmup".format(testing_mode, set_name,
                                                     ic_index, t_save)
            with h5py.File(results_path_base, 'a') as h5f:
                h5f.create_dataset(wfn,
                                   data=input_,
                                   compression="gzip",
                                   chunks=True)

        input_sequence = np.array(input_sequence)
        input_sequence = input_sequence[np.newaxis]

        init_hidden_state = model.getInitialRNNHiddenState(1)
        # """ In case of crnns get initial hidden states """
        # if mclass.model_class in ["crnn", "dimred_rnn"]:
        #     """ Hidden of RNN """
        #     init_hidden_state = model.getInitialRNNHiddenState(1)

        # elif mclass.model_class in ["dimred_rc"]:
        #     """ Hidden of reservoir computer """
        #     init_hidden_state = np.zeros((1, model.rc_reservoir_size, 1))

        # elif mclass.model_class in ["cnn_sindy"]:
        #     """ No hidden state in SINDy """
        #     init_hidden_state = []
        #     pass

        # input_sequence = model.transform2Tensor(input_sequence)
        # init_hidden_state = model.transform2Tensor(init_hidden_state)

        assert model.n_warmup >= 2, "Warm-up timesteps {:} need to be larger than 1.".format(
            model.n_warmup)

        warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _ = model.forward(
            input_sequence, init_hidden_state)

        # # print(ark)
        # """ Propagation for the hidden state """
        # if mclass.model_class in ["crnn", "dimred_rnn"]:

        #     if model.n_warmup > 1:
        #         warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _, _, _, _, _ = model.model.forward(
        #         input_sequence, init_hidden_state, is_train=False)
        #     else:
        #         # In case of predictor with n_warmup=1 (no warmup)
        #         # assert(model.has_predictor)
        #         last_hidden_state = init_hidden_state

        # elif mclass.model_class in ["cnn_sindy"]:

        #     if model.n_warmup > 1:
        #         warmup_latent_states = model.model_autoencoder.forwardEncoder(input_sequence)
        #         warmup_data_output = model.model_autoencoder.forwardDecoder(warmup_latent_states)
        #         latent_states_pred = warmup_latent_states
        #         last_hidden_state = []
        #     else:
        #         pass

        # elif mclass.model_class in ["dimred_rc"]:
        #     if model.n_warmup > 1:
        #         warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _ = model.forward(input_sequence, init_hidden_state)
        #     else:
        #         # In case of predictor with n_warmup=1 (no warmup)
        #         # assert(model.has_predictor)
        #         last_hidden_state = init_hidden_state
        # else:
        #     raise ValueError("Unknown model class {:}.".format(mclass.model_class))
        """ Multiscale forecasting """
        iterative_propagation_is_latent = True
        input_is_latent = True
        input_latent = latent_states_pred[:, -1:, :]
        input_t = input_latent

        timestep_start += model.n_warmup
        time_start = time.time()

        multiscale_rounds, macro_steps_per_round, micro_steps_per_round, micro_steps_main, macro_steps_main = mclass.getMultiscaleParams(
            testing_mode, prediction_horizon)

        prediction = []
        target = []

        time_dynamics = 0.0
        time_latent_prop = 0.0

        for round_ in range(multiscale_rounds):

            multiscale_macro_steps = macro_steps_per_round[round_]

            if multiscale_macro_steps > 0:

                for t in range(multiscale_macro_steps):

                    prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forecast(
                        input_t,
                        last_hidden_state,
                        horizon=1,
                    )

                    # if mclass.model_class in ["crnn", "dimred_rnn"]:

                    #     prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, RNN_outputs_, input_decoded, time_latent_prop_t, _, _ = model.model.forward(
                    #         input_t,
                    #         last_hidden_state,
                    #         is_train=False,
                    #         is_iterative_forecasting=False,
                    #         iterative_forecasting_prob=0,
                    #         horizon=None,
                    #         iterative_propagation_is_latent=iterative_propagation_is_latent,
                    #         input_is_latent=input_is_latent,
                    #         )

                    # elif mclass.model_class in ["dimred_rc"]:
                    #     prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forecast(
                    #         input_t,
                    #         last_hidden_state,
                    #         1,
                    #         )

                    # elif mclass.model_class in ["cnn_sindy"]:

                    #     prediction_model_dyn, latent_states_pred, time_latent_prop_t = model.forecast(
                    #         input_t,
                    #         1,
                    #         )

                    time_latent_prop += time_latent_prop_t
                    """ One batch, one timestep """
                    # assert prediction_model_dyn.size(0) == 1
                    # assert prediction_model_dyn.size(1) == 1

                    assert np.shape(prediction_model_dyn)[0] == 1
                    assert np.shape(prediction_model_dyn)[1] == 1

                    # if torch.is_tensor(latent_states_pred):
                    #     assert latent_states_pred.size(0) == 1
                    #     assert latent_states_pred.size(1) == 1
                    #     latent_states_pred_save = latent_states_pred.detach().cpu().numpy()[0, -1]
                    # else:
                    assert np.shape(latent_states_pred)[0] == 1
                    assert np.shape(latent_states_pred)[1] == 1
                    latent_states_pred_save = latent_states_pred[0, -1]
                    """ Saving prediction """
                    # prediction_save = prediction_model_dyn.detach().cpu().numpy()[0, -1]
                    prediction_save = prediction_model_dyn[0, -1]

                    t_load = timestep_start + t
                    t_save = t_load - model.n_warmup
                    # print("# loading timestep from data t={:} (warm-up), saving as t={:} (used as target)".format(t_load, t_save))
                    sample = data_set.getSequencePart(ic_index, t_load,
                                                      t_load + 1)
                    target_save = sample[0]

                    prev_target = target_save

                    prediction_save = model.data_info_dict[
                        "scaler"].descaleData(prediction_save,
                                              single_sequence=True,
                                              single_batch=True,
                                              verbose=False,
                                              check_bounds=False)
                    target_save = model.data_info_dict["scaler"].descaleData(
                        target_save,
                        single_sequence=True,
                        single_batch=True,
                        verbose=False,
                        check_bounds=False)

                    pfn = prediction_field_name_in_hdf5.format(
                        ic_index, t_save)
                    tfn = target_field_name_in_hdf5.format(ic_index, t_save)
                    lfn = latent_state_field_name_in_hdf5.format(
                        ic_index, t_save)

                    utils.createDatasetsInHDF5File(
                        results_path_base,
                        [pfn, tfn, lfn],
                        [
                            prediction_save, target_save,
                            latent_states_pred_save
                        ],
                    )

                    predictions_all_ic.append((results_path_base, pfn))
                    targets_all_ic.append((results_path_base, tfn))
                    latent_states_all_ic.append((results_path_base, lfn))

                    input_t = latent_states_pred

                    errors = utils.computeErrors(
                        target_save,
                        prediction_save,
                        model.data_info_dict,
                        single_sample=True)

                    # Updating the error
                    for error in errors:
                        error_dict_ic[error].append(errors[error])
                """ Initial state for the micro dynamics """
                # init_state = prediction_model_dyn.cpu().detach().numpy()[-1][-1]
                # print(input_t.size())

                init_state = prediction_model_dyn[-1][-1]

            elif round_ == 0:
                t_load = timestep_start - 1
                # print("# loading timestep from data t={:} (warm-up) (used as initial input)".format(t_load))
                sample = data_set.getSequencePart(ic_index,
                                                  t_load,
                                                  t_load + 1,
                                                  scale=True)
                init_state = sample[0]

            else:
                # Correcting the shape
                init_state = prediction_sys_dyn[:, -1:, :]
                init_state = init_state[0, 0]

            timestep_start += multiscale_macro_steps

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
                        verbose=False)
                    """ Debugging, checking the difference between initial condition and saved data """
                    # print("# loading timestep from data t={:} (warm-up) (used as initial input / checking)".format(t_load))
                    # temp = data_set.getSequencePart(ic_index, t_load, t_load+1, scale=False)
                    # print(np.shape(temp))
                    # print(np.shape(init_state))
                    # print(np.linalg.norm(temp-init_state))
                    # init_state = temp

                    time_dynamics_start = time.time()
                    prediction_sys_dyn = systems.evolveSystem(
                        mclass,
                        init_state,
                        total_time,
                        dt,
                        t0=timestep_start,
                        round_=round_,
                        micro_steps=micro_steps_main,
                        macro_steps=macro_steps_main,
                    )
                    # if multiscale_micro_steps > 1:
                    #     prediction_sys_dyn = np.repeat(init_state, multiscale_micro_steps, axis=0)
                    # else:
                    #     prediction_sys_dyn = init_state

                    # print(np.shape(prediction_sys_dyn))

                    assert np.shape(
                        prediction_sys_dyn
                    )[0] == multiscale_micro_steps, "np.shape(prediction_sys_dyn)={:}, multiscale_micro_steps={:}".format(
                        np.shape(prediction_sys_dyn), multiscale_micro_steps)

                    for tt in range(np.shape(prediction_sys_dyn)[0]):
                        """ Saving the prediction (unscaled) """
                        prediction_save = prediction_sys_dyn[tt]
                        t_load = timestep_start + tt
                        t_save = t_load - model.n_warmup
                        """ No need for scaling """
                        # print("# loading timestep from data t={:} (warm-up), saving as t={:} (used as target)".format(t_load, t_save))
                        sample = data_set.getSequencePart(ic_index,
                                                          t_load,
                                                          t_load + 1,
                                                          scale=False)
                        target_save = sample[0]

                        prev_target = target_save

                        pfn = prediction_field_name_in_hdf5.format(
                            ic_index, t_save)
                        tfn = target_field_name_in_hdf5.format(
                            ic_index, t_save)

                        utils.createDatasetsInHDF5File(
                            results_path_base,
                            [pfn, tfn],
                            [prediction_save, target_save],
                        )

                        predictions_all_ic.append((results_path_base, pfn))
                        targets_all_ic.append((results_path_base, tfn))
                        # print("Saving targets/predictions at timestep {:}".format(t_save))

                        errors = utils.computeErrors(
                            target_save,
                            prediction_save,
                            model.data_info_dict,
                            single_sample=True)
                        # Updating the error
                        for error in errors:
                            error_dict_ic[error].append(errors[error])

                    prediction_sys_dyn = model.data_info_dict[
                        "scaler"].scaleData(
                            prediction_sys_dyn,
                            single_sequence=True,
                            check_bounds=False,
                        )
                    """ Saling back the initial state """
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
                    # prediction_sys_dyn = model.toPrecision(prediction_sys_dyn)
                    # prediction_sys_dyn_tensor = utils.transform2Tensor(model, prediction_sys_dyn)

                    init_state = np.reshape(init_state,
                                            (1, *np.shape(init_state)))
                    """
                    In order to update the last hidden state,
                    we need to feed the high scale dynamics up to input_t
                    to the network
                    """

                    idle_dynamics = prediction_sys_dyn[:, :-1, :].copy()
                    idle_dynamics = np.concatenate((init_state, idle_dynamics),
                                                   axis=1)
                    # idle_dynamics = model.transform2Tensor(idle_dynamics)
                    # idle_dynamics = utils.transform2Tensor(model, idle_dynamics)

                    _, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forward(
                        idle_dynamics,
                        last_hidden_state,
                    )

                    # if mclass.model_class in ["crnn", "dimred_rnn"]:
                    #     _, last_hidden_state, latent_states_, latent_states_pred, RNN_outputs_, _, time_latent_prop_t, _, _ = model.model.forward(
                    #         idle_dynamics,
                    #         last_hidden_state,
                    #         is_train=False,
                    #         is_iterative_forecasting=False,
                    #         )

                    # elif mclass.model_class in ["dimred_rc"]:
                    #     _, last_hidden_state, latent_states_, latent_states_pred, time_latent_prop_t = model.forward(
                    #         idle_dynamics,
                    #         last_hidden_state,
                    #         )

                    # elif mclass.model_class in ["cnn_sindy"]:
                    #     time_start = time.time()
                    #     latent_states_pred = model.model_autoencoder.forwardEncoder(idle_dynamics)
                    #     time_latent_prop_t = time.time() - time_start
                    #     # _ = model.model_autoencoder.forwardDecoder(latent_states_pred)
                    # else:
                    #     raise ValueError("Not implemented.")

                    time_latent_prop += time_latent_prop_t

                    # if torch.is_tensor(latent_states_pred):
                    #     assert latent_states_pred.size(0) == 1
                    #     assert latent_states_pred.size(1) == multiscale_micro_steps, "latent_states_pred.size(1)={:}, multiscale_micro_steps={:}".format(latent_states_pred.size(1), multiscale_micro_steps)
                    # else:
                    assert np.shape(latent_states_pred)[0] == 1
                    assert np.shape(
                        latent_states_pred
                    )[1] == multiscale_micro_steps, "np.shape(latent_states_pred)[1]={:}, multiscale_micro_steps={:}".format(
                        np.shape(latent_states_pred)[1],
                        multiscale_micro_steps)

                    for tt in range(np.shape(latent_states_pred)[1]):
                        """ Saving the prediction (unscaled) """
                        latent_states_pred_save = latent_states_pred[0, tt]
                        # if torch.is_tensor(latent_states_pred_save): latent_states_pred_save = latent_states_pred_save.detach().cpu().numpy()
                        t_load = timestep_start + tt
                        t_save = t_load - model.n_warmup
                        # print("Saving latent state at timestep {:}".format(t_save))
                        lfn = latent_state_field_name_in_hdf5.format(
                            ic_index, t_save)
                        utils.createDatasetsInHDF5File(
                            results_path_base, [lfn],
                            [latent_states_pred_save])
                        latent_states_all_ic.append((results_path_base, lfn))

                else:
                    prediction_sys_dyn = prediction_model_dyn

                if iterative_propagation_is_latent:
                    # if torch.is_tensor(latent_states_pred):
                    #     input_t = latent_states_pred[:, -1:, :].clone()
                    # else:
                    input_t = latent_states_pred[:, -1:, :]

                else:
                    raise ValueError("Not supposed to happen.")
                    # Next to feed, the last predicted state (from the dynamics)
                    # input_t = prediction_sys_dyn[:, -1:, :].clone()
                    # input_t = model.transform2Tensor(input_t)
                    input_t = prediction_sys_dyn[:, -1:, :]
                    # input_t = utils.transform2Tensor(model, input_t)

                timestep_start += multiscale_micro_steps

        time_end = time.time()
        time_total = time_end - time_start
        time_total_per_iter = time_total / prediction_horizon
        time_total_per_iter_all.append(time_total_per_iter)

        predictions_all.append(predictions_all_ic)
        predictions_augmented_all.append(predictions_all_ic)
        targets_all.append(targets_all_ic)
        targets_augmented_all.append(targets_all_ic)
        latent_states_all.append(latent_states_all_ic)
        latent_states_augmented_all.append(latent_states_all_ic)

        # Updating the error
        for error in error_dict.keys():
            error_dict[error].append(error_dict_ic[error])

    time_total_per_iter_all = np.array(time_total_per_iter_all)
    time_total_per_iter = np.mean(time_total_per_iter_all)

    # Computing the average over time
    error_dict_avg = {}
    for key in error_dict:
        error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
    utils.printErrors(error_dict_avg)

    print(
        "[utils_multiscale] Shapes of predictions_all/targets_all/latent_states_all = {:}/{:}/{:}"
        .format(
            np.shape(predictions_all),
            np.shape(targets_all),
            np.shape(latent_states_all),
        ))
    # Computing additional errors based on all predictions (e.g. frequency spectra)
    additional_results_dict, additional_errors_dict = utils.computeAdditionalResults(
        mclass.model, predictions_all, targets_all, dt)
    error_dict_avg = {**error_dict_avg, **additional_errors_dict}

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
        # **state_statistics
    }

    results = systems.addResultsSystem(mclass.model, results, testing_mode)
    return results
