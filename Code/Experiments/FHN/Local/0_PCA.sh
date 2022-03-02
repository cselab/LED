#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=0

system_name=FHN
input_dim=2
channels=1
Dx=101

truncate_data_batches=0

dimred_method="pca"
scaler=MinMaxZeroOne
latent_state_dim=6
random_seed=7
write_to_log=1

num_test_ICS=1000
prediction_horizon=10000

plot_testing_ics_examples=1
plot_latent_dynamics=1

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred \
--system_name $system_name \
--input_dim $input_dim \
--channels $channels \
--Dx $Dx \
--write_to_log $write_to_log \
--mode $mode \
--dimred_method $dimred_method \
--latent_state_dim $latent_state_dim  \
--scaler $scaler \
--num_test_ICS $num_test_ICS \
--prediction_horizon $prediction_horizon \
--display_output 1 \
--random_seed $random_seed \
--plot_latent_dynamics $plot_latent_dynamics \
--truncate_data_batches $truncate_data_batches \
--plot_testing_ics_examples $plot_testing_ics_examples \
--test_on_test 1 \
--test_on_val 1

# dimred_method=diffmaps
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred \
# --system_name $system_name \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --write_to_log $write_to_log \
# --mode $mode \
# --dimred_method $dimred_method \
# --latent_state_dim $latent_state_dim  \
# --scaler $scaler \
# --num_test_ICS $num_test_ICS \
# --prediction_horizon $prediction_horizon \
# --display_output 1 \
# --random_seed $random_seed \
# --plot_latent_dynamics $plot_latent_dynamics \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 1 \
# --test_on_val 1

