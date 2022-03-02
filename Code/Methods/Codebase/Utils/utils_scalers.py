#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np


class scaler(object):
    def __init__(
        self,
        scaler_type,
        data_min,
        data_max,
        common_scaling_per_input_dim=False,
        common_scaling_per_channels=False,
        channels=0,
    ):
        super(scaler, self).__init__()
        # Data are of three possible types:
        # Type 1:
        #         (T, input_dim)
        # Type 2:
        #         (T, input_dim, Cx) (input_dim is the N_particles, perm_inv, etc.)
        # Type 3:
        #         (T, input_dim, Cx, Cy, Cz)
        self.scaler_type = scaler_type

        if self.scaler_type not in ["MinMaxZeroOne", "MinMaxMinusOneOne"]:
            raise ValueError("Scaler {:} not implemented.".format(
                self.scaler_type))

        self.data_min = np.array(data_min)
        self.data_max = np.array(data_max)
        self.data_range = self.data_max - self.data_min
        self.common_scaling_per_input_dim = common_scaling_per_input_dim
        self.common_scaling_per_channels = common_scaling_per_channels
        self.channels = channels

    def scaleData(self,
                  batch_of_sequences,
                  reuse=None,
                  single_sequence=False,
                  check_bounds=True):
        if single_sequence: batch_of_sequences = batch_of_sequences[np.newaxis]
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]
        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)

        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_max)))
            batch_of_sequences_scaled = np.array(
                (batch_of_sequences - data_min) / (data_max - data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= 0.0))
                assert (np.all(batch_of_sequences_scaled <= 1.0))

        elif self.scaler_type == "MinMaxMinusOneOne":
            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_max)))

            batch_of_sequences_scaled = np.array(
                (2.0 * batch_of_sequences - data_max - data_min) /
                (data_max - data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= -1.0))
                assert (np.all(batch_of_sequences_scaled <= 1.0))

        elif self.scaler_type == "Standard":
            data_mean = self.repeatScalerParam(
                self.data_mean, self.data_shape)
            data_std = self.repeatScalerParam(
                self.data_std, self.data_shape)

            assert (np.all(
                np.shape(batch_of_sequences) == np.shape(data_mean)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_std)))

            batch_of_sequences_scaled = np.array(
                (batch_of_sequences - data_mean) / data_std)

        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def repeatScalerParam(self, data, shape):
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]

        common_scaling_per_channels = self.common_scaling_per_channels
        common_scaling_per_input_dim = self.common_scaling_per_input_dim
        # Running through the shape in reverse order
        if common_scaling_per_input_dim:
            D = shape[2]
            # Commong scaling for all inputs !
            data = np.repeat(data[np.newaxis], D, 0)

        # Running through the shape in reverse order
        if common_scaling_per_channels:
            # Repeating the scaling for each channel
            assert (len(shape[::-1][:-3]) == self.channels)
            for channel_dim in shape[::-1][:-3]:
                data = np.repeat(data[np.newaxis], channel_dim, 0)
                data = np.swapaxes(data, 0, 1)

        T = shape[1]
        data = np.repeat(data[np.newaxis], T, 0)
        K = shape[0]
        data = np.repeat(data[np.newaxis], K, 0)
        return data

    def descaleData(self,
                    batch_of_sequences_scaled,
                    single_sequence=True,
                    single_batch=False,
                    verbose=True,
                    check_bounds=True):
        if verbose:
            print("[utils_scalers] # descaleData() #")
            print("[utils_scalers] max = {:} ".format(
                np.max(batch_of_sequences_scaled)))
            print("[utils_scalers] min = {:} ".format(
                np.min(batch_of_sequences_scaled)))

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        if single_batch:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]

        # Size of the batch_of_sequences_scaled is [K, T, ...]
        # Size of the batch_of_sequences_scaled is [K, T, D]
        # Size of the batch_of_sequences_scaled is [K, T, D, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C, C]
        self.data_shape = np.shape(batch_of_sequences_scaled)
        self.data_shape_length = len(self.data_shape)
        if self.scaler_type == "MinMaxZeroOne":

            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape)

            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_min)))
            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_max)))

            batch_of_sequences = np.array(batch_of_sequences_scaled *
                                          (data_max - data_min) + data_min)

            if check_bounds:
                assert (np.all(batch_of_sequences >= data_min))
                assert (np.all(batch_of_sequences <= data_max))

        elif self.scaler_type == "MinMaxMinusOneOne":

            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape)

            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_min)))
            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_max)))

            batch_of_sequences = np.array(batch_of_sequences_scaled *
                                          (data_max - data_min) + data_min +
                                          data_max) / 2.0

            if check_bounds:
                assert (np.all(batch_of_sequences >= data_min))
                assert (np.all(batch_of_sequences <= data_max))

        elif self.scaler_type == "Standard":

            data_mean = self.repeatScalerParam(
                self.data_mean, self.data_shape)
            data_std = self.repeatScalerParam(
                self.data_std, self.data_shape)

            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_mean)))
            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_std)))

            batch_of_sequences = np.array(batch_of_sequences_scaled *
                                          data_std + data_mean)

        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence: batch_of_sequences = batch_of_sequences[0]
        if single_batch: batch_of_sequences = batch_of_sequences[0]
        return np.array(batch_of_sequences)
