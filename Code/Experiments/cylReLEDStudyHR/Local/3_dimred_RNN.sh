#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=2

random_seed_in_name=1

max_epochs=200
max_rounds=4
overfitting_patience=5

retrain=0
batch_size=4
write_to_log=1

learning_rate=0.001
noise_level=0.0

system_name=cylRe100HR_demo
Dy=512
Dx=1024
channels=2
input_dim=4
truncate_data_batches=2
truncate_data_batches=4

make_videos=0
cudnn_benchmark=1

scaler=MinMaxZeroOne
optimizer_str=adabelief


latent_state_dim=5
reconstruction_loss=1
output_forecasting_loss=0

sequence_length=1

random_seed=114


num_test_ICS=1
prediction_horizon=4
plot_testing_ics_examples=1

# mode=train
# mode=debug
mode=all
# mode=test
# mode=plot

dimred_method="pca"
# dimred_method="diffmaps"

diffmaps_weight=0.1
diffmaps_num_neighbors=2

truncate_timesteps=200
truncate_data_batches=1
batch_size=1

plot_latent_dynamics=1
plot_testing_ics_examples=1
mode=all

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred \
--mode $mode \
--system_name $system_name \
--write_to_log $write_to_log \
--dimred_method $dimred_method \
--diffmaps_weight $diffmaps_weight \
--diffmaps_num_neighbors $diffmaps_num_neighbors \
--input_dim $input_dim \
--channels $channels \
--Dx $Dx \
--Dy $Dy \
--latent_state_dim $latent_state_dim  \
--scaler $scaler \
--num_test_ICS $num_test_ICS \
--prediction_horizon $prediction_horizon \
--display_output 1 \
--truncate_data_batches $truncate_data_batches \
--truncate_timesteps $truncate_timesteps \
--plot_testing_ics_examples $plot_testing_ics_examples \
--plot_latent_dynamics $plot_latent_dynamics \
--test_on_test 1 \
--test_on_val 1 \
--compute_spectrum 0


max_epochs=10

RNN_activation_str="tanh"
RNN_activation_str_output="tanhplus"
RNN_cell_type="lstm"
RNN_layers_size=32
RNN_layers_num=1
n_warmup=2
output_forecasting_loss=1
latent_forecasting_loss=0
sequence_length=2
learning_rate=0.0001

train_RNN=1
perform_dim_red=0

truncate_data_batches=0
batch_size=1

mode=train
# mode=test
# mode=all

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py dimred_rnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --dimred_method $dimred_method \
# --diffmaps_weight $diffmaps_weight \
# --diffmaps_num_neighbors $diffmaps_num_neighbors \
# --perform_dim_red $perform_dim_red \
# --train_RNN $train_RNN \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --Dy $Dy \
# --optimizer_str $optimizer_str \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
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
# --truncate_timesteps $truncate_timesteps \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 0 \
# --test_on_val 1 \
# --n_warmup $n_warmup








# # MULTISCALE TESTING #

# truncate_timesteps=20
# prediction_horizon=4
# n_warmup=2
# num_test_ICS=1

# # --multiscale_macro_steps_list 0 \

# # # MULTISCALE TESTING #

# mode=multiscale
# # mode=plotMultiscale
# plot_multiscale_results_comparison=1



# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py \
# --multiscale_testing 1 \
# --plot_multiscale_results_comparison $plot_multiscale_results_comparison \
# --multiscale_micro_steps_list 1 \
# --multiscale_macro_steps_list 0 \
# --multiscale_macro_steps_list 3 \
# --multiscale_macro_steps_list 100 \
# dimred_rnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --dimred_method $dimred_method \
# --diffmaps_weight $diffmaps_weight \
# --diffmaps_num_neighbors $diffmaps_num_neighbors \
# --perform_dim_red $perform_dim_red \
# --train_RNN $train_RNN \
# --input_dim $input_dim \
# --channels $channels \
# --Dx $Dx \
# --Dy $Dy \
# --optimizer_str $optimizer_str \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
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


