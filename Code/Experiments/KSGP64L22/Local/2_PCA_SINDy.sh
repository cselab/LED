#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=0

system_name=KSGP64L22
input_dim=1
channels=1
Dx=64

truncate_data_batches=0

dimred_method="pca"
scaler=MinMaxZeroOne
latent_state_dim=10
random_seed=7
write_to_log=1

num_test_ICS=1
prediction_horizon=11

plot_testing_ics_examples=1
plot_latent_dynamics=1

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot


# sindy_integrator_type="discrete"
sindy_integrator_type="continuous"
sindy_degree=2
sindy_threshold=0.0
sindy_library="poly"
sindy_interp_factor=1
sindy_smoother_polyorder=0
sindy_smoother_window_size=7
n_warmup=10

iterative_state_forecasting=1
iterative_latent_forecasting=1
teacher_forcing_forecasting=1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_sindy \
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
--sindy_integrator_type $sindy_integrator_type \
--sindy_degree $sindy_degree \
--sindy_threshold $sindy_threshold \
--sindy_library $sindy_library \
--sindy_interp_factor $sindy_interp_factor \
--sindy_smoother_polyorder $sindy_smoother_polyorder \
--sindy_smoother_window_size $sindy_smoother_window_size \
--n_warmup $n_warmup \
--iterative_state_forecasting $iterative_state_forecasting \
--iterative_latent_forecasting $iterative_latent_forecasting \
--teacher_forcing_forecasting $teacher_forcing_forecasting



# # # MULTISCALE TESTING #

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
# dimred_sindy \
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
# --sindy_integrator_type $sindy_integrator_type \
# --sindy_degree $sindy_degree \
# --sindy_threshold $sindy_threshold \
# --sindy_library $sindy_library \
# --sindy_interp_factor $sindy_interp_factor \
# --sindy_smoother_polyorder $sindy_smoother_polyorder \
# --sindy_smoother_window_size $sindy_smoother_window_size \
# --n_warmup $n_warmup \
# --iterative_state_forecasting $iterative_state_forecasting \
# --iterative_latent_forecasting $iterative_latent_forecasting \
# --teacher_forcing_forecasting $teacher_forcing_forecasting






