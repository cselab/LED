#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=2

random_seed_in_name=1

max_epochs=50
max_rounds=10
overfitting_patience=20

retrain=0
batch_size=32
write_to_log=1

learning_rate=0.001
# weight_decay=0.0001
weight_decay=0.0
# dropout_keep_prob=1.0
dropout_keep_prob=0.99

# mode=train
# mode=test
mode=all
# mode=plot

system_name=FHN
Dx=101
channels=1
input_dim=2
truncate_data_batches=0


make_videos=0
cudnn_benchmark=1

activation_str_general=celu
activation_str_output=tanhplus
scaler=MinMaxZeroOne
optimizer_str=adabelief

AE_convolutional=1
AE_batch_norm=0
AE_conv_transpose=0
# AE_conv_transpose=0
AE_pool_type="avg"


latent_state_dim=4
reconstruction_loss=1
output_forecasting_loss=0

prediction_horizon=100
sequence_length=1

latent_forecasting_loss=0

num_test_ICS=2
plot_testing_ics_examples=1

# mode=train
# mode=debug
# mode=all
mode=test
# mode=plot
make_videos=0

beta_vae=0
beta_vae_weight_max=1.0
random_seed=114
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.1
sequence_length=4
retrain=0
AE_conv_architecture=conv_latent_1


# ####################################
# # RESERVOIR COMPUTER
# ####################################


# mode=train
# mode=debug
mode=all
# mode=test


learning_rate_AE=$learning_rate
random_seed_in_AE_name=$random_seed

random_seed=7
n_warmup=10

write_to_log=1


# prediction_horizon=41
prediction_horizon=1000
num_test_ICS=2
plot_testing_ics_examples=1







for rc_solver in "pinv"
do
for rc_approx_reservoir_size in 1000
do
for rc_degree in 10
do
for rc_radius in 1
do
for rc_sigma_input in 2
do
for rc_dynamics_length in 60
do
for rc_regularization in 0.001
do
for rc_noise_level_per_mill in 1
do
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rc \
--mode $mode \
--dimred_method "ae" \
--system_name $system_name \
--cudnn_benchmark $cudnn_benchmark \
--write_to_log $write_to_log \
--input_dim $input_dim \
--channels $channels \
--Dx $Dx \
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
--plot_testing_ics_examples $plot_testing_ics_examples \
--test_on_test 1 \
--test_on_val 0 \
--rc_solver $rc_solver \
--rc_approx_reservoir_size $rc_approx_reservoir_size \
--rc_degree $rc_degree \
--rc_radius $rc_radius \
--rc_sigma_input $rc_sigma_input \
--rc_dynamics_length $rc_dynamics_length \
--rc_regularization $rc_regularization \
--rc_noise_level_per_mill $rc_noise_level_per_mill \
--n_warmup $n_warmup
done
done
done
done
done
done
done
done







