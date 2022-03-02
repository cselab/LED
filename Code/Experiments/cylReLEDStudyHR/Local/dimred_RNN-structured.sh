#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=0

random_seed_in_name=1

max_epochs=20
max_rounds=4
overfitting_patience=5

retrain=0
batch_size=16
write_to_log=0

learning_rate=0.0001
noise_level=0.01

system_name=cylRe100HR_demo
Dx=1024
Dy=512
channels=2
input_dim=4
truncate_data_batches=10
truncate_timesteps=10

make_videos=0
cudnn_benchmark=1

scaler=MinMaxZeroOne
optimizer_str=adabelief


latent_state_dim=5
reconstruction_loss=1
output_forecasting_loss=0

sequence_length=1

random_seed=114
retrain=0
plotting=1

num_test_ICS=1
prediction_horizon=41
plot_testing_ics_examples=1

# mode=train
# mode=debug
# mode=all
mode=test
# mode=plot

dimred_method="pca"

perform_dim_red=1

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --dimred_method $dimred_method \
# --perform_dim_red $perform_dim_red \
# --input_dim $input_dim \
# --channels $channels \
# --Dy $Dy \
# --Dx $Dx \
# --optimizer_str $optimizer_str \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_state_dim $latent_state_dim  \
# --sequence_length $sequence_length \
# --scaler $scaler \
# --learning_rate $learning_rate \
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
# --plotting $plotting \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 0 \
# --test_on_val 1







batch_size=1
max_epochs=10
learning_rate=0.001

truncate_data_batches=1
truncate_timesteps=11

perform_dim_red=0
train_AE=1
AE_convolutional=1
AE_batch_norm=1
AE_batch_norm_affine=1
AE_conv_transpose=1
AE_pool_type=avg
AE_conv_architecture=conv_latent_3
activation_str_general=celu
activation_str_output=tanh
mode=train
# mode=test

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rnn \
--mode $mode \
--system_name $system_name \
--cudnn_benchmark $cudnn_benchmark \
--write_to_log $write_to_log \
--dimred_method $dimred_method \
--perform_dim_red $perform_dim_red \
--train_AE $train_AE \
--input_dim $input_dim \
--channels $channels \
--Dy $Dy \
--Dx $Dx \
--activation_str_general $activation_str_general \
--activation_str_output $activation_str_output \
--AE_convolutional $AE_convolutional \
--AE_batch_norm $AE_batch_norm \
--AE_batch_norm_affine $AE_batch_norm_affine \
--AE_conv_transpose $AE_conv_transpose \
--AE_pool_type $AE_pool_type \
--AE_conv_architecture $AE_conv_architecture \
--optimizer_str $optimizer_str \
--output_forecasting_loss $output_forecasting_loss \
--latent_state_dim $latent_state_dim  \
--sequence_length $sequence_length \
--scaler $scaler \
--learning_rate $learning_rate \
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
--truncate_timesteps $truncate_timesteps \
--plot_testing_ics_examples $plot_testing_ics_examples \
--test_on_test 0 \
--test_on_val 1






random_seed_in_AE_name=$random_seed
learning_rate_AE=$learning_rate
noise_level_AE=$noise_level

learning_rate=0.001
max_epochs=10
max_rounds=4
overfitting_patience=5

RNN_activation_str="tanh"
RNN_activation_str_output="tanhplus"
RNN_cell_type="gru"
RNN_layers_size=16
RNN_layers_num=1
n_warmup=10
output_forecasting_loss=1

train_RNN=1
train_AE=0
perform_dim_red=0

# mode=train
# mode=test
mode=all

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --dimred_method $dimred_method \
# --perform_dim_red $perform_dim_red \
# --train_RNN $train_RNN \
# --train_AE $train_AE \
# --input_dim $input_dim \
# --channels $channels \
# --Dy $Dy \
# --Dx $Dx \
# --activation_str_general $activation_str_general \
# --random_seed_in_AE_name $random_seed_in_AE_name \
# --learning_rate_AE $learning_rate_AE \
# --noise_level_AE $noise_level_AE \
# --AE_convolutional $AE_convolutional \
# --AE_batch_norm $AE_batch_norm \
# --AE_conv_transpose $AE_conv_transpose \
# --AE_pool_type $AE_pool_type \
# --AE_conv_architecture $AE_conv_architecture \
# --optimizer_str $optimizer_str \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_state_dim $latent_state_dim  \
# --sequence_length $sequence_length \
# --scaler $scaler \
# --learning_rate $learning_rate \
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
# --RNN_activation_str $RNN_activation_str \
# --RNN_activation_str_output $RNN_activation_str_output \
# --RNN_cell_type $RNN_cell_type \
# --RNN_layers_size $RNN_layers_size \
# --RNN_layers_num $RNN_layers_num \
# --make_videos $make_videos \
# --retrain $retrain \
# --compute_spectrum 0 \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 0 \
# --test_on_val 1 \
# --n_warmup $n_warmup




