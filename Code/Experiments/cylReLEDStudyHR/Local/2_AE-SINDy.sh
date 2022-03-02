#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=2

random_seed_in_name=1

max_epochs=10
max_rounds=2
overfitting_patience=20

retrain=0
batch_size=32
write_to_log=1

learning_rate=0.001
# weight_decay=0.0001
weight_decay=0.0
# dropout_keep_prob=0.95
dropout_keep_prob=1.0

# mode=train
# mode=test
mode=all
# mode=plot

system_name=cylRe100HR
# system_name=cylRe100HR
# system_name=cylRe2500
Dy=512
Dx=1024
channels=2
input_dim=4
truncate_data_batches=0
channels=2


make_videos=0
cudnn_benchmark=1

activation_str_general=celu
activation_str_output=tanhplus
scaler=MinMaxZeroOne
optimizer_str=adabelief

AE_convolutional=1
AE_conv_transpose=1
AE_pool_type="avg"
AE_size_factor=1
AE_batch_norm=0

latent_state_dim=12
reconstruction_loss=1
output_forecasting_loss=0

prediction_horizon=100
sequence_length=1

latent_forecasting_loss=0

num_test_ICS=2
plot_testing_ics_examples=0

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot
make_videos=0

beta_vae=0
beta_vae_weight_max=1.0
random_seed=114
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.0
# sequence_length=4
retrain=0
AE_conv_architecture=conv_latent_1


# ####################################
# # SINDY
# ####################################



truncate_data_batches=1
truncate_timesteps=51
batch_size=1
max_epochs=2

# mode=train
# mode=debug
mode=all
# mode=test

learning_rate_AE=$learning_rate

random_seed_in_AE_name=$random_seed
random_seed=7
n_warmup=10

write_to_log=1

# sindy_integrator_type="discrete"
sindy_integrator_type="continuous"
sindy_degree=2
sindy_threshold=0.01
sindy_library="poly"
# sindy_library="fourier"
sindy_interp_factor=5
sindy_smoother_polyorder=3
sindy_smoother_window_size=7

prediction_horizon=10
num_test_ICS=1
plot_testing_ics_examples=0





# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py cnn_sindy \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --Dy $Dy \
# --optimizer_str $optimizer_str \
# --beta_vae $beta_vae \
# --beta_vae_weight_max $beta_vae_weight_max \
# --c1_latent_smoothness_loss $c1_latent_smoothness_loss \
# --c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
# --reconstruction_loss $reconstruction_loss \
# --activation_str_general $activation_str_general \
# --activation_str_output $activation_str_output \
# --AE_convolutional $AE_convolutional \
# --AE_batch_norm $AE_batch_norm \
# --AE_conv_transpose $AE_conv_transpose \
# --AE_pool_type $AE_pool_type \
# --AE_conv_architecture $AE_conv_architecture \
# --latent_state_dim $latent_state_dim  \
# --sequence_length $sequence_length \
# --scaler $scaler \
# --learning_rate_AE $learning_rate_AE \
# --weight_decay $weight_decay \
# --dropout_keep_prob $dropout_keep_prob \
# --batch_size $batch_size \
# --overfitting_patience $overfitting_patience \
# --max_epochs $max_epochs \
# --max_rounds $max_rounds \
# --num_test_ICS $num_test_ICS \
# --prediction_horizon $prediction_horizon \
# --display_output 1 \
# --random_seed $random_seed \
# --random_seed_in_name $random_seed_in_name \
# --random_seed_in_AE_name $random_seed_in_AE_name \
# --teacher_forcing_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --make_videos $make_videos \
# --compute_spectrum 0 \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --truncate_timesteps $truncate_timesteps \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 1 \
# --test_on_val 0 \
# --sindy_integrator_type $sindy_integrator_type \
# --sindy_degree $sindy_degree \
# --sindy_threshold $sindy_threshold \
# --sindy_library $sindy_library \
# --sindy_interp_factor $sindy_interp_factor \
# --sindy_smoother_polyorder $sindy_smoother_polyorder \
# --sindy_smoother_window_size $sindy_smoother_window_size




prediction_horizon=10
n_warmup=2
num_test_ICS=1
plot_testing_ics_examples=1

# --multiscale_macro_steps_list 0 \

# # MULTISCALE TESTING #

mode=multiscale
# mode=plotMultiscale
plot_multiscale_results_comparison=0


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py \
--multiscale_testing 1 \
--plot_multiscale_results_comparison $plot_multiscale_results_comparison \
--multiscale_micro_steps_list 1 \
--multiscale_macro_steps_list 0 \
--multiscale_macro_steps_list 2 \
--multiscale_macro_steps_list 100 \
cnn_sindy \
--mode $mode \
--system_name $system_name \
--cudnn_benchmark $cudnn_benchmark \
--write_to_log $write_to_log \
--input_dim $input_dim \
--channels $channels \
--Dx $Dx \
--Dy $Dy \
--optimizer_str $optimizer_str \
--beta_vae $beta_vae \
--beta_vae_weight_max $beta_vae_weight_max \
--c1_latent_smoothness_loss $c1_latent_smoothness_loss \
--c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
--reconstruction_loss $reconstruction_loss \
--activation_str_general $activation_str_general \
--activation_str_output $activation_str_output \
--AE_convolutional $AE_convolutional \
--AE_batch_norm $AE_batch_norm \
--AE_conv_transpose $AE_conv_transpose \
--AE_pool_type $AE_pool_type \
--AE_conv_architecture $AE_conv_architecture \
--latent_state_dim $latent_state_dim  \
--sequence_length $sequence_length \
--scaler $scaler \
--learning_rate_AE $learning_rate_AE \
--weight_decay $weight_decay \
--dropout_keep_prob $dropout_keep_prob \
--batch_size $batch_size \
--overfitting_patience $overfitting_patience \
--max_epochs $max_epochs \
--max_rounds $max_rounds \
--num_test_ICS $num_test_ICS \
--prediction_horizon $prediction_horizon \
--display_output 1 \
--random_seed $random_seed \
--random_seed_in_name $random_seed_in_name \
--random_seed_in_AE_name $random_seed_in_AE_name \
--teacher_forcing_forecasting 1 \
--iterative_latent_forecasting 1 \
--make_videos $make_videos \
--compute_spectrum 0 \
--plot_state_distributions 0 \
--plot_system 0 \
--plot_latent_dynamics 1 \
--truncate_data_batches $truncate_data_batches \
--truncate_timesteps $truncate_timesteps \
--plot_testing_ics_examples $plot_testing_ics_examples \
--test_on_test 1 \
--test_on_val 0 \
--sindy_integrator_type $sindy_integrator_type \
--sindy_degree $sindy_degree \
--sindy_threshold $sindy_threshold \
--sindy_library $sindy_library \
--sindy_interp_factor $sindy_interp_factor \
--sindy_smoother_polyorder $sindy_smoother_polyorder \
--sindy_smoother_window_size $sindy_smoother_window_size \
--n_warmup $n_warmup
