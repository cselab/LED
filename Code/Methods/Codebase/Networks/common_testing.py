#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import os
import numpy as np

from .. import Utils as utils
from .. import Systems as systems

from tqdm import tqdm
import time
import warnings


def testIterativeOnHDF5Structured(model, data_set, dt, set_name, testing_mode):
    print(
        "[common_testing] # predictIndexes() (Memory efficient implementation for structured data) #"
    )

    num_test_ICS = model.num_test_ICS
    num_max_ICS = len(data_set)
    if num_test_ICS > num_max_ICS:
        warnings.warn(
            "Not enough ({:}) ICs in the dataset {:}. Using {:} possible ICs.".
            format(num_test_ICS, set_name, num_max_ICS))
        num_test_ICS = num_max_ICS

    prediction_horizon = model.prediction_horizon
    """ Maximum prediction horizon """
    n_steps = data_set.seq_paths[0]['num_timesteps']
    prediction_horizon_max = n_steps - model.n_warmup - 1
    if prediction_horizon > prediction_horizon_max:
        warnings.warn(
            "[common_testing] Prediction horizon ({:}) is larger than the timesteps in the structured dataset  ({:}) - n_warmup ({:}) - 1 = ({:}). Setting prediction_horizon={:}."
            .format(prediction_horizon, n_steps, model.n_warmup,
                    prediction_horizon_max, prediction_horizon_max))
        prediction_horizon = prediction_horizon_max
    assert prediction_horizon > 0, "[common_testing] Prediction horizon needs to be positive. Reduce it."

    ic_indexes = np.arange(num_test_ICS)

    predictions_all = []
    targets_all = []
    latent_states_all = []

    predictions_augmented_all = []
    targets_augmented_all = []
    latent_states_augmented_all = []

    # Dictionary of error lists
    error_dict = utils.getErrorLabelsDict(model)

    print("[common_testing] # predictIndexes() on {:}/{:} initial conditions.".
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

    tqdm_ = tqdm(total=len(ic_indexes) * (model.n_warmup + prediction_horizon))
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
            # print("# loading timestep from data t={:} (warm-up), saving as t={:}".format(t_load, t_save))
            sample = data_set.getSequencePart(ic_index, t_load, t_load + 2)
            input_ = sample[0]
            target_ = sample[1]

            input_sequence.append(input_)

            wfn = "{:}_{:}_ic{:}_t{:}_warmup".format(testing_mode, set_name,
                                                     ic_index, t_save)
            utils.createDatasetsInHDF5File(results_path_base, [wfn], [input_])

            tqdm_.update(1)

        input_sequence = np.array(input_sequence)
        input_sequence = input_sequence[np.newaxis]

        init_hidden_state = model.getInitialRNNHiddenState(1)

        # warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _ = model.warmup(input_sequence, init_hidden_state)

        # input_sequence = model.transform2Tensor(input_sequence)
        # init_hidden_state = model.transform2Tensor(init_hidden_state)

        timestep_start += model.n_warmup
        time_start = time.time()
        # First input is always high-dimensional
        input_is_latent = False
        iterative_propagation_is_latent = False
        for t in range(prediction_horizon):
            # outputs, next_hidden_state, latent_states, latent_states_pred, _, _, _, _, _ = model.model.forward(input_sequence, init_hidden_state, is_train=False, is_iterative_forecasting=False, iterative_forecasting_prob=0, iterative_forecasting_gradient=0, horizon=None, input_is_latent=input_is_latent, iterative_propagation_is_latent=iterative_propagation_is_latent)

            outputs, next_hidden_state, latent_states, latent_states_pred, _ = model.forward(
                input_sequence,
                init_hidden_state,
                input_is_latent=input_is_latent,
                iterative_propagation_is_latent=iterative_propagation_is_latent
            )

            # prediction_ = outputs.detach().cpu().numpy()[0, -1]
            # latent_states_pred_ = latent_states_pred.detach().cpu().numpy()[0, -1]
            # print(np.shape(input_sequence))
            # print(np.shape(outputs))
            # print(ark)
            prediction_ = outputs[0, -1]
            latent_states_pred_ = latent_states_pred[0, -1]

            prediction_ = prediction_[np.newaxis][np.newaxis]
            prediction_ = np.reshape(prediction_, np.shape(target_))

            prev_target = target_

            prediction_ = model.data_info_dict["scaler"].descaleData(
                prediction_,
                single_sequence=True,
                single_batch=True,
                verbose=False,
                check_bounds=False)
            target_ = model.data_info_dict["scaler"].descaleData(
                target_,
                single_sequence=True,
                single_batch=True,
                verbose=False,
                check_bounds=False)

            t_save = t_load - model.n_warmup
            pfn = prediction_field_name_in_hdf5.format(ic_index, t_save)
            tfn = target_field_name_in_hdf5.format(ic_index, t_save)
            lfn = latent_state_field_name_in_hdf5.format(ic_index, t_save)

            utils.createDatasetsInHDF5File(
                results_path_base,
                [pfn, tfn, lfn],
                [prediction_, target_, latent_states_pred_],
            )

            predictions_all_ic.append((results_path_base, pfn))
            targets_all_ic.append((results_path_base, tfn))
            latent_states_all_ic.append((results_path_base, lfn))

            errors = utils.computeErrors(
                target_,
                prediction_,
                model.data_info_dict,
                single_sample=True)

            # Updating the error
            for error in errors:
                error_dict_ic[error].append(errors[error])

            # Loading the next scaled input and output sample
            t_load = timestep_start + t
            t_save = t_load - model.n_warmup
            # print("# loading timestep from data t={:}, saving as t={:}".format(t_load, t_save))
            sample = data_set.getSequencePart(ic_index, t_load, t_load + 2)
            input_ = sample[0]
            target_ = sample[1]

            # print(prev_target==input_)

            # Adding the prediction to the input
            if "iterative_state" in testing_mode:
                input_sequence = np.concatenate(
                    (input_sequence, outputs[:, -1:]), axis=1)
                input_sequence = input_sequence[:, 1:]
                init_hidden_state = next_hidden_state

            elif "teacher_forcing" in testing_mode:
                input_ = input_[np.newaxis]
                input_ = input_[np.newaxis]
                # input_ = model.transform2Tensor(input_)
                # if model.gpu: input_ = input_.cuda()
                input_sequence = np.concatenate((input_sequence, input_),
                                                axis=1)
                # input_sequence = torch.cat([input_sequence, input_], dim=1)
                input_sequence = input_sequence[:, 1:]
                init_hidden_state = next_hidden_state

            elif "iterative_latent" in testing_mode:
                # input_sequence = latent_states.detach()
                # input_sequence = torch.cat([input_sequence, latent_states_pred[:, -1:]], dim=1)
                # input_sequence = input_sequence[:, 1:]
                # init_hidden_state = next_hidden_state

                input_sequence = latent_states
                # input_sequence = torch.cat([input_sequence, latent_states_pred[:, -1:]], dim=1)
                input_sequence = np.concatenate(
                    (input_sequence, latent_states_pred[:, -1:]), axis=1)
                input_sequence = input_sequence[:, 1:]
                init_hidden_state = next_hidden_state

                input_is_latent = True
                iterative_propagation_is_latent = True
            else:
                raise ValueError("Not yet implemented.")

            tqdm_.update(1)

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

    tqdm_.close()
    time_total_per_iter_all = np.array(time_total_per_iter_all)
    time_total_per_iter = np.mean(time_total_per_iter_all)

    error_dict_avg = utils.getErrorDictAvg(error_dict)

    results = utils.addResultsIterative(
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
    )
    results = systems.addResultsSystem(model, results, testing_mode)
    results = systems.computeStateDistributionStatisticsSystem(
        model, results, targets_all, predictions_all)
    return results


def testIterativeOnHDF5(model, data_loader, dt, set_name, testing_mode):
    print("[common_testing] # testIterativeOnHDF5() #")
    assert (testing_mode in model.getTestingModes())

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

    num_max_ICS = len(data_loader)
    if num_test_ICS > num_max_ICS:
        warnings.warn(
            "Not enough ({:}) ICs in the dataset {:}. Using {:} possible ICs.".
            format(num_test_ICS, set_name, num_max_ICS))
        num_test_ICS = len(data_loader)

    print("[common_testing] # predictIndexes() on {:}/{:} initial conditions.".
          format(num_test_ICS, len(data_loader)))

    ic_num = 1
    ic_indexes = []

    tqdm_ = tqdm(total=num_test_ICS)

    for sequence in data_loader:
        if ic_num > num_test_ICS: break
        if model.params["display_output"]:
            print("[common_testing] IC {:}/{:}, {:2.3f}%".format(
                ic_num, num_test_ICS, ic_num / num_test_ICS * 100))
        sequence = sequence[0]

        # STARTING TO PREDICT THE SEQUENCE IN model.predict_on=model.sequence_length
        # Warming-up with sequence_length
        model.predict_on = model.n_warmup
        # assert model.predict_on + model.prediction_horizon <= np.shape(sequence)[0], "model.predict_on ({:}) + model.prediction_horizon ({:}) > np.shape(sequence)[0] ({:}). Not enough timesteps in the {:} data.".format(model.predict_on, model.prediction_horizon, np.shape(sequence)[0], set_name)
        assert (model.predict_on - model.n_warmup >= 0)

        if model.predict_on + model.prediction_horizon > np.shape(sequence)[0]:
            prediction_horizon = np.shape(sequence)[0] - model.predict_on
            warnings.warn(
                "[common_testing] model.predict_on ({:}) + model.prediction_horizon ({:}) > np.shape(sequence)[0] ({:}). Not enough timesteps in the {:} data. Using a prediction horizon of {:}."
                .format(model.predict_on, model.prediction_horizon,
                        np.shape(sequence)[0], set_name, prediction_horizon))
        else:
            prediction_horizon = model.prediction_horizon

        sequence = sequence[model.predict_on -
                            model.n_warmup:model.predict_on +
                            prediction_horizon]

        prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter = model.predictSequence(
            sequence,
            testing_mode,
            dt=dt,
            prediction_horizon=prediction_horizon)

        print(np.shape(prediction))
        print(np.shape(target))
        prediction = model.data_info_dict["scaler"].descaleData(
            prediction, single_sequence=True, check_bounds=False)
        target = model.data_info_dict["scaler"].descaleData(
            target, single_sequence=True, check_bounds=False)

        prediction_augment = model.data_info_dict["scaler"].descaleData(
            prediction_augment, single_sequence=True, check_bounds=False)
        target_augment = model.data_info_dict["scaler"].descaleData(
            target_augment, single_sequence=True, check_bounds=False)

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
        tqdm_.update(1)

    tqdm_.close()
    time_total_per_iter_all = np.array(time_total_per_iter_all)
    time_total_per_iter = np.mean(time_total_per_iter_all)

    predictions_all = np.array(predictions_all)
    targets_all = np.array(targets_all)
    latent_states_all = np.array(latent_states_all)

    predictions_augmented_all = np.array(predictions_augmented_all)
    targets_augmented_all = np.array(targets_augmented_all)
    latent_states_augmented_all = np.array(latent_states_augmented_all)

    print("[common_testing] Shape of trajectories:")
    print("[common_testing] {:}:".format(np.shape(targets_all)))
    print("[common_testing] {:}:".format(np.shape(predictions_all)))

    error_dict_avg = utils.getErrorDictAvg(error_dict)

    results = utils.addResultsIterative(
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
    )
    results = systems.addResultsSystem(model, results, testing_mode)
    results = systems.computeStateDistributionStatisticsSystem(
        model, results, targets_all, predictions_all)
    return results


def testEncodeDecodeOnHDF5Structured(model, data_set, dt, set_name,
                                     testing_mode):

    prediction_horizon = model.prediction_horizon
    num_test_ICS = model.num_test_ICS
    num_max_ICS = len(data_set)
    if num_test_ICS > num_max_ICS:
        warnings.warn(
            "[common_testing] Not enough ICs ({:}) in the dataset {:}. Using {:} possible ICs."
            .format(num_test_ICS, set_name, num_max_ICS))
        num_test_ICS = num_max_ICS
    assert num_test_ICS > 0
    print(
        "[testEncodeDecodeOnHDF5Structured()] Testing only the Autoencoder on {:}/{:} initial conditions from the data."
        .format(model.num_test_ICS, len(data_set)))
    """ Maximum prediction horizon """
    prediction_horizon_max = data_set.seq_paths[0]['num_timesteps']
    if prediction_horizon > prediction_horizon_max:
        warnings.warn(
            "[common_testing] Prediction horizon ({:}) is larger than the timesteps in the structured dataset ({:}). Setting prediction_horizon={:}."
            .format(prediction_horizon, prediction_horizon_max,
                    prediction_horizon_max))
        prediction_horizon = prediction_horizon_max

    ic_indexes = np.arange(num_test_ICS)

    outputs_all = []
    inputs_all = []
    latent_states_all = []

    latent_states_all_data = []
    # Dictionary of error lists
    error_dict = utils.getErrorLabelsDict(model)

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

    tqdm_ = tqdm(total=len(ic_indexes) * prediction_horizon)

    for ic_num in range(len(ic_indexes)):
        ic_index = ic_indexes[ic_num]

        outputs_all_ic = []
        targets_all_ic = []
        latent_states_all_ic = []
        latent_states_all_ic_data = []
        error_dict_ic = utils.getErrorLabelsDict(model)

        # init_hidden_state = model.getInitialRNNHiddenState(1)

        for t in range(prediction_horizon):

            t_load = t
            sample = data_set.getSequencePart(ic_index, t_load, t_load + 1)
            input_ = sample[0]
            target_ = sample[0]

            input_sequence = input_[np.newaxis]
            input_sequence = input_sequence[np.newaxis]

            # input_sequence = model.transform2Tensor(input_sequence)
            # init_hidden_state = model.transform2Tensor(init_hidden_state)

            # _, _, latent_states, _, _, outputs, _, _, _ = model.model.forward(input_sequence, init_hidden_state, is_train=False)

            # prediction_ = outputs.detach().cpu().numpy()[0, -1]
            # latent_states_ = latent_states.detach().cpu().numpy()[0, -1]
            prediction_, latent_states_ = model.encodeDecode(input_sequence)
            latent_states_ = latent_states_[0, -1]
            prediction_ = prediction_[0, -1]

            prediction_ = prediction_[np.newaxis]
            prediction_ = prediction_[np.newaxis]

            prediction_ = np.reshape(prediction_, np.shape(target_))

            prediction_ = model.data_info_dict["scaler"].descaleData(
                prediction_,
                single_sequence=True,
                single_batch=True,
                verbose=False,
                check_bounds=False)
            target_ = model.data_info_dict["scaler"].descaleData(
                target_,
                single_sequence=True,
                single_batch=True,
                verbose=False,
                check_bounds=False)
            """ Creating hdf5 file for each timestep """
            pfn = prediction_field_name_in_hdf5.format(ic_index, t)
            tfn = target_field_name_in_hdf5.format(ic_index, t)
            lfn = latent_state_field_name_in_hdf5.format(ic_index, t)
            utils.createDatasetsInHDF5File(
                results_path_base,
                [pfn, tfn, lfn],
                [prediction_, target_, latent_states_],
            )
            """ Saving the hdf5 files in arrays """
            outputs_all_ic.append((results_path_base, pfn))
            targets_all_ic.append((results_path_base, tfn))
            latent_states_all_ic.append((results_path_base, lfn))
            latent_states_all_ic_data.append(latent_states_)

            errors = utils.computeErrors(
                target_,
                prediction_,
                model.data_info_dict,
                single_sample=True)

            # Updating the error
            for error in errors:
                error_dict_ic[error].append(errors[error])

            tqdm_.update(1)

        outputs_all.append(outputs_all_ic)
        inputs_all.append(targets_all_ic)
        latent_states_all.append(latent_states_all_ic)

        latent_states_all_data.append(latent_states_all_ic_data)

        # Updating the error
        for error in error_dict.keys():
            error_dict[error].append(error_dict_ic[error])

    tqdm_.close()

    error_dict_avg = utils.getErrorDictAvg(error_dict)
    results = utils.addResultsAutoencoder(
        model,
        outputs_all,
        inputs_all,
        latent_states_all,
        dt,
        error_dict_avg,
        latent_states_all_data=latent_states_all_data)
    results = systems.addResultsSystem(model, results, testing_mode)
    results = systems.computeStateDistributionStatisticsSystem(
        model, results, inputs_all, outputs_all)
    return results


def testEncodeDecodeOnHDF5(model,
                           data_loader,
                           dt,
                           set_name,
                           testing_mode,
                           dataset=None):
    if model.num_test_ICS > len(data_loader):
        num_test_ICS = len(data_loader)
    else:
        num_test_ICS = model.num_test_ICS

    assert num_test_ICS > 0
    print(
        "[testEncodeDecodeOnHDF5()] # Testing on {:}/{:} initial conditions.".
        format(num_test_ICS, len(data_loader)))

    latent_states_all = []
    outputs_all = []
    inputs_all = []
    num_seqs_tested_on = 0

    error_dict = utils.getErrorLabelsDict(model)
    for input_sequence_ in data_loader:

        if num_seqs_tested_on >= num_test_ICS: break

        assert np.shape(input_sequence_)[0] == 1

        if model.data_info_dict["structured"]:
            input_sequence = dataset.getSequencesPart(input_sequence_, 0,
                                                      model.prediction_horizon)
            input_sequence = input_sequence[0]
        else:
            input_sequence = input_sequence_[0]

            if model.prediction_horizon <= 0:
                raise ValueError("Prediction horizon cannot be {:}.".format(
                    model.prediction_horizon))
            input_sequence = input_sequence[:model.prediction_horizon]

        if model.prediction_horizon > np.shape(input_sequence)[0]:
            warnings.warn(
                "[common_testing] Warning: model.prediction_horizon={:} is bigger than the length of the sequence {:}."
                .format(model.prediction_horizon,
                        np.shape(input_sequence)[0]))

        # initial_hidden_states = model.getInitialRNNHiddenState(1)
        input_sequence = input_sequence[np.newaxis, :]

        # input_sequence = model.transform2Tensor(input_sequence)
        # initial_hidden_states = model.transform2Tensor(initial_hidden_states)
        # _, _, latent_states, _, _, outputs, _, _, _ = model.model.forward(input_sequence, initial_hidden_states, is_train=False)

        outputs, latent_states = model.encodeDecode(input_sequence)

        input_sequence = input_sequence[0]
        latent_states = latent_states[0]
        outputs = outputs[0]

        input_sequence = model.data_info_dict["scaler"].descaleData(
            input_sequence,
            single_sequence=True,
            check_bounds=False,
            verbose=False)
        outputs = model.data_info_dict["scaler"].descaleData(
            outputs, single_sequence=True, check_bounds=False, verbose=False)

        errors = utils.computeErrors(input_sequence, outputs,
                                     model.data_info_dict)
        # Updating the error
        for error in errors:
            error_dict[error].append(errors[error])

        latent_states_all.append(latent_states)
        outputs_all.append(outputs)
        inputs_all.append(input_sequence)
        num_seqs_tested_on += 1

    inputs_all = np.array(inputs_all)
    outputs_all = np.array(outputs_all)

    error_dict_avg = utils.getErrorDictAvg(error_dict)
    results = utils.addResultsAutoencoder(model, outputs_all, inputs_all,
                                          latent_states_all, dt,
                                          error_dict_avg)
    results = systems.addResultsSystem(model, results, testing_mode)
    results = systems.computeStateDistributionStatisticsSystem(
        model, results, inputs_all, outputs_all)
    return results


def testOnMode(model, data_loader, dt, set_, testing_mode, data_set):
    assert (testing_mode in model.getTestingModes())
    assert (set_ in ["train", "test", "val"])
    print("[testOnMode()] ---- Testing on Mode {:} ----".format(testing_mode))

    if testing_mode in ["dimred_testing", "autoencoder_testing"]:
        if model.data_info_dict["structured"]:
            results = testEncodeDecodeOnHDF5Structured(model, data_set, dt,
                                                       set_, testing_mode)
        else:
            results = testEncodeDecodeOnHDF5(model, data_loader, dt, set_,
                                             testing_mode)
    elif testing_mode in [
            "iterative_state_forecasting",
            "iterative_latent_forecasting",
            "teacher_forcing_forecasting",
    ]:
        if model.data_info_dict["structured"]:
            results = testIterativeOnHDF5Structured(model, data_set, dt, set_,
                                                    testing_mode)

        else:
            results = testIterativeOnHDF5(model, data_loader, dt, set_,
                                          testing_mode)

    data_path = utils.getResultsDir(model) + "/results_{:}_{:}".format(
        testing_mode, set_)
    utils.saveData(results, data_path, model.save_format)
    return 0


def testingRoutine(
    model,
    data_loader,
    dt,
    set_,
    data_set,
    testing_modes,
):
    for testing_mode in testing_modes:
        testOnMode(model, data_loader, dt, set_, testing_mode, data_set)
    return 0


def testModesOnSet(model,
                   set_="train",
                   print_=False,
                   rank_str="",
                   gpu=False,
                   testing_modes=[]):
    print("[testModesOnSet()] #####     Testing on set: {:}     ######".format(
        set_))
    dt = model.data_info_dict["dt"]
    if set_ == "test":
        data_path = model.data_path_test
    elif set_ == "val":
        data_path = model.data_path_val
    elif set_ == "train":
        data_path = model.data_path_train
    else:
        raise ValueError("Invalid set {:}.".format(set_))

    data_loader_test, _, data_set = utils.getDataLoader(
        data_path,
        model.data_info_dict,
        batch_size=1,
        shuffle=False,
        print_=print_,
        rank_str=rank_str,
        gpu=gpu,
    )
    testingRoutine(model, data_loader_test, dt, set_, data_set, testing_modes)
    return 0
