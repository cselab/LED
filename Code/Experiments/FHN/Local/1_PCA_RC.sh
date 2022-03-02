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

num_test_ICS=5
prediction_horizon=41

plot_testing_ics_examples=1
plot_latent_dynamics=1

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot

rc_solver=pinv
rc_degree=9
rc_radius=0.9
rc_sigma_input=1.0
rc_dynamics_length=10
rc_regularization=0.0001
rc_noise_level_per_mill=1
rc_approx_reservoir_size=200
n_warmup=10

iterative_state_forecasting=1
iterative_latent_forecasting=1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rc \
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
--test_on_val 1 \
--rc_solver $rc_solver \
--rc_approx_reservoir_size $rc_approx_reservoir_size \
--rc_degree $rc_degree \
--rc_radius $rc_radius \
--rc_sigma_input $rc_sigma_input \
--rc_dynamics_length $rc_dynamics_length \
--rc_regularization $rc_regularization \
--rc_noise_level_per_mill $rc_noise_level_per_mill \
--n_warmup $n_warmup \
--iterative_state_forecasting $iterative_state_forecasting \
--iterative_latent_forecasting $iterative_latent_forecasting \




# MULTISCALE TESTING #

truncate_timesteps=20
prediction_horizon=4
n_warmup=2
num_test_ICS=1


# # MULTISCALE TESTING #

# mode=multiscale
# # mode=plotMultiscale
# plot_multiscale_results_comparison=1
# plot_testing_ics_examples=0
# plot_latent_dynamics=0

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py \
# --multiscale_testing 1 \
# --plot_multiscale_results_comparison $plot_multiscale_results_comparison \
# --multiscale_micro_steps_list 1 \
# --multiscale_macro_steps_list 0 \
# --multiscale_macro_steps_list 3 \
# --multiscale_macro_steps_list 100 \
# dimred_rc \
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
# --test_on_val 1 \
# --rc_solver $rc_solver \
# --rc_approx_reservoir_size $rc_approx_reservoir_size \
# --rc_degree $rc_degree \
# --rc_radius $rc_radius \
# --rc_sigma_input $rc_sigma_input \
# --rc_dynamics_length $rc_dynamics_length \
# --rc_regularization $rc_regularization \
# --rc_noise_level_per_mill $rc_noise_level_per_mill \
# --n_warmup $n_warmup \
# --iterative_state_forecasting $iterative_state_forecasting \
# --iterative_latent_forecasting $iterative_latent_forecasting \



