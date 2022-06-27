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
prediction_horizon=100

plot_testing_ics_examples=1
plot_latent_dynamics=1

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot


# for file in ./* ; do mv $file ${file//GPU-DimRedRNN/DimRedRNN} ; done

latent_forecasting_loss=1
RNN_cell_type="lstm"
RNN_activation_str="tanh"
RNN_activation_str_output="tanhplus"
RNN_layers_size=16
RNN_layers_num=1
latent_space_scaler=MinMaxZeroOne
sequence_length=40
learning_rate=0.01
batch_size=32
overfitting_patience=40
max_epochs=10
max_rounds=5
optimizer_str="adabelief"

iterative_state_forecasting=1
iterative_latent_forecasting=1
teacher_forcing_forecasting=1
n_warmup=10

mode=all

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rnn \
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
--n_warmup $n_warmup \
--iterative_state_forecasting $iterative_state_forecasting \
--iterative_latent_forecasting $iterative_latent_forecasting \
--teacher_forcing_forecasting $teacher_forcing_forecasting \
--latent_forecasting_loss $latent_forecasting_loss \
--RNN_cell_type $RNN_cell_type \
--RNN_activation_str $RNN_activation_str \
--RNN_activation_str_output $RNN_activation_str_output \
--RNN_layers_size $RNN_layers_size \
--RNN_layers_num $RNN_layers_num \
--latent_space_scaler $latent_space_scaler \
--optimizer_str $optimizer_str \
--sequence_length $sequence_length \
--learning_rate $learning_rate \
--batch_size $batch_size \
--max_epochs $max_epochs \
--overfitting_patience $overfitting_patience \
--max_rounds $max_rounds


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
# dimred_rnn \
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
# --n_warmup $n_warmup \
# --iterative_state_forecasting $iterative_state_forecasting \
# --iterative_latent_forecasting $iterative_latent_forecasting \
# --teacher_forcing_forecasting $teacher_forcing_forecasting \
# --latent_forecasting_loss $latent_forecasting_loss \
# --RNN_cell_type $RNN_cell_type \
# --RNN_activation_str $RNN_activation_str \
# --RNN_activation_str_output $RNN_activation_str_output \
# --RNN_layers_size $RNN_layers_size \
# --RNN_layers_num $RNN_layers_num \
# --latent_space_scaler $latent_space_scaler \
# --optimizer_str $optimizer_str \
# --sequence_length $sequence_length \
# --learning_rate $learning_rate \
# --batch_size $batch_size \
# --max_epochs $max_epochs \
# --overfitting_patience $overfitting_patience \
# --max_rounds $max_rounds







