#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=0

random_seed_in_name=1

max_epochs=10
max_rounds=4
overfitting_patience=5

retrain=0
batch_size=16
write_to_log=0

learning_rate=0.001
weight_decay=0.0
# dropout_keep_prob=1.0
dropout_keep_prob=0.99
noise_level=0.0

Dx=64
channels=1
input_dim=1
truncate_data_batches=64
batch_size=16

make_videos=0
cudnn_benchmark=1

activation_str_general=celu
# activation_str_general=relu
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

sequence_length=1

plot_testing_ics_examples=1

beta_vae=0
beta_vae_weight_max=1.0
random_seed=114
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.1
# sequence_length=10
retrain=0

AE_conv_architecture=conv_latent_1


prediction_horizon=1000
num_test_ICS=10
system_name=KSGP64L22

# mode=train
# mode=debug
mode=all
# mode=test

plotting=0
precision=double
# precision=float

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn \
--mode $mode \
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
--output_forecasting_loss $output_forecasting_loss \
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
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--dropout_keep_prob $dropout_keep_prob \
--noise_level $noise_level \
--batch_size $batch_size \
--overfitting_patience $overfitting_patience \
--max_epochs $max_epochs \
--max_rounds $max_rounds \
--num_test_ICS $num_test_ICS \
--prediction_horizon $prediction_horizon \
--display_output 1 \
--random_seed $random_seed \
--random_seed_in_name $random_seed_in_name \
--teacher_forcing_forecasting 1 \
--iterative_latent_forecasting 1 \
--make_videos $make_videos \
--retrain $retrain \
--compute_spectrum 0 \
--plot_state_distributions 0 \
--plot_system 0 \
--plot_latent_dynamics 1 \
--truncate_data_batches $truncate_data_batches \
--precision $precision \
--plot_testing_ics_examples $plot_testing_ics_examples \
--plotting $plotting \
--test_on_test 1 \
--test_on_val 1




