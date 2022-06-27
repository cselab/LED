#!/bin/bash

############### NOTE #############################################
# Make sure the AE is trained with the 4_AE.sh script.
# The trained autoencoder is loaded here. The AE loaded
# here has to have the same parametrization
# as the pretrained autoencoder, else the script will complain
# that it could not find the trained model,
# so be careful to select the same hyper-parameters.
##################################################################
#
cd ../../../Methods


CUDA_DEVICES=0

random_seed_in_name=1

max_epochs=2
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

Dx=101
channels=1
input_dim=2
truncate_data_batches=0

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
num_test_ICS=1000
system_name=FHN

# [crnn] ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.01-L2_0.0-RS_114-CHANNELS_1-2-5-KERNELS_3-3-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_0.99-LD_4
# [crnn] ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.01-L2_0.0-RS_114-CHANNELS_1-2-5-KERNELS_3-3-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_0.99-LD_4

truncate_data_batches=128
batch_size=32

# mode=train
# mode=debug
mode=all
# mode=test

plotting=0
precision=double
# precision=float




# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --optimizer_str $optimizer_str \
# --beta_vae $beta_vae \
# --beta_vae_weight_max $beta_vae_weight_max \
# --c1_latent_smoothness_loss $c1_latent_smoothness_loss \
# --c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
# --output_forecasting_loss $output_forecasting_loss \
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
# --learning_rate $learning_rate \
# --weight_decay $weight_decay \
# --dropout_keep_prob $dropout_keep_prob \
# --noise_level $noise_level \
# --batch_size $batch_size \
# --overfitting_patience $overfitting_patience \
# --max_epochs $max_epochs \
# --max_rounds $max_rounds \
# --num_test_ICS $num_test_ICS \
# --prediction_horizon $prediction_horizon \
# --display_output 1 \
# --random_seed $random_seed \
# --random_seed_in_name $random_seed_in_name \
# --teacher_forcing_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --make_videos $make_videos \
# --retrain $retrain \
# --compute_spectrum 0 \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --precision $precision \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --plotting $plotting \
# --test_on_test 1 \
# --test_on_val 1













random_seed_in_AE_name=$random_seed

random_seed=7


iterative_loss_validation=0
iterative_loss_schedule_and_gradient=none

# iterative_loss_validation=1
# iterative_loss_schedule_and_gradient=inverse_sigmoidal_with_gradient

overfitting_patience=10
max_rounds=5
max_epochs=20

iterative_propagation_during_training_is_latent=1

prediction_horizon=41
num_test_ICS=1
# ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.01-NL_0.01-L2_0.0-NL_0.0-RS_114-CHANNELS_1-2-5-KERNELS_3-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_0.99-LD_3
# ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.01-NL_0.01-L2_0.0-RS_114-CHANNELS_1-2-5-KERNELS_3-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_0.99-LD_3

# retrain=0
# train_RNN_only=0
# load_trained_AE=1
# reconstruction_loss=0
# output_forecasting_loss=1
# latent_forecasting_loss=0
# CUDA_DEVICES=0

retrain=0
train_RNN_only=1
load_trained_AE=1
reconstruction_loss=0
output_forecasting_loss=1
latent_forecasting_loss=1
CUDA_DEVICES=0

learning_rate_AE=$learning_rate
learning_rate=0.001
noise_level_AE=$noise_level
noise_level=0.0
overfitting_patience=10


latent_space_scaler=Standard
RNN_cell_type="lstm"
RNN_layers_num=1
RNN_layers_size=16
RNN_activation_str_output="identity"
sequence_length=10
prediction_length=6
n_warmup_train=0
# n_warmup_train=0
n_warmup=10

mode=all
# mode=test
# mode=train




# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --optimizer_str $optimizer_str \
# --beta_vae $beta_vae \
# --beta_vae_weight_max $beta_vae_weight_max \
# --c1_latent_smoothness_loss $c1_latent_smoothness_loss \
# --c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
# --iterative_loss_validation $iterative_loss_validation \
# --iterative_loss_schedule_and_gradient $iterative_loss_schedule_and_gradient \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
# --reconstruction_loss $reconstruction_loss \
# --activation_str_general $activation_str_general \
# --activation_str_output $activation_str_output \
# --AE_convolutional $AE_convolutional \
# --AE_batch_norm $AE_batch_norm \
# --AE_conv_transpose $AE_conv_transpose \
# --AE_pool_type $AE_pool_type \
# --AE_conv_architecture $AE_conv_architecture \
# --train_RNN_only $train_RNN_only \
# --load_trained_AE $load_trained_AE \
# --RNN_cell_type $RNN_cell_type \
# --RNN_layers_num $RNN_layers_num \
# --RNN_layers_size $RNN_layers_size \
# --RNN_activation_str_output $RNN_activation_str_output \
# --latent_state_dim $latent_state_dim  \
# --sequence_length $sequence_length \
# --prediction_length $prediction_length \
# --scaler $scaler \
# --learning_rate_AE $learning_rate_AE \
# --learning_rate $learning_rate \
# --weight_decay $weight_decay \
# --dropout_keep_prob $dropout_keep_prob \
# --noise_level $noise_level \
# --noise_level_AE $noise_level_AE \
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
# --iterative_state_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --iterative_propagation_during_training_is_latent $iterative_propagation_during_training_is_latent \
# --make_videos $make_videos \
# --retrain $retrain \
# --compute_spectrum 0 \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 1 \
# --n_warmup_train $n_warmup_train \
# --n_warmup $n_warmup \
# --latent_space_scaler $latent_space_scaler \
# --precision $precision \
# --test_on_test 1
















# MULTISCALE TESTING #

truncate_timesteps=20
prediction_horizon=4
n_warmup=2
num_test_ICS=1

# --multiscale_macro_steps_list 0 \

# # MULTISCALE TESTING #

mode=multiscale
# mode=plotMultiscale
plot_multiscale_results_comparison=1
plot_testing_ics_examples=0
plot_latent_dynamics=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py \
--multiscale_testing 1 \
--plot_multiscale_results_comparison $plot_multiscale_results_comparison \
--multiscale_micro_steps_list 1 \
--multiscale_macro_steps_list 0 \
--multiscale_macro_steps_list 3 \
--multiscale_macro_steps_list 100 \
crnn \
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
--iterative_loss_validation $iterative_loss_validation \
--iterative_loss_schedule_and_gradient $iterative_loss_schedule_and_gradient \
--output_forecasting_loss $output_forecasting_loss \
--latent_forecasting_loss $latent_forecasting_loss \
--reconstruction_loss $reconstruction_loss \
--activation_str_general $activation_str_general \
--activation_str_output $activation_str_output \
--AE_convolutional $AE_convolutional \
--AE_batch_norm $AE_batch_norm \
--AE_conv_transpose $AE_conv_transpose \
--AE_pool_type $AE_pool_type \
--AE_conv_architecture $AE_conv_architecture \
--train_RNN_only $train_RNN_only \
--load_trained_AE $load_trained_AE \
--RNN_cell_type $RNN_cell_type \
--RNN_layers_num $RNN_layers_num \
--RNN_layers_size $RNN_layers_size \
--RNN_activation_str_output $RNN_activation_str_output \
--latent_state_dim $latent_state_dim  \
--sequence_length $sequence_length \
--prediction_length $prediction_length \
--scaler $scaler \
--learning_rate_AE $learning_rate_AE \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--dropout_keep_prob $dropout_keep_prob \
--noise_level $noise_level \
--noise_level_AE $noise_level_AE \
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
--iterative_state_forecasting 1 \
--iterative_latent_forecasting 1 \
--iterative_propagation_during_training_is_latent $iterative_propagation_during_training_is_latent \
--make_videos $make_videos \
--retrain $retrain \
--compute_spectrum 0 \
--plot_state_distributions 0 \
--plot_system 0 \
--plot_latent_dynamics $plot_latent_dynamics \
--truncate_data_batches $truncate_data_batches \
--plot_testing_ics_examples $plot_testing_ics_examples \
--test_on_test 1 \
--n_warmup_train $n_warmup_train \
--n_warmup $n_warmup \
--latent_space_scaler $latent_space_scaler \
--precision $precision \
--test_on_test 1

























