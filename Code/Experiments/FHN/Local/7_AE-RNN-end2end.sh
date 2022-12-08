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

# whether to include the random seed in the name
random_seed_in_name=1

# maximum number of epochs to train
max_epochs=10000

# maximum rounds reducing the learning rate
max_rounds=20

# the overfitting patience of the learning rate scheduler
overfitting_patience=20

# whether to retrain (if you want to resume training from a trained model)
retrain=0

# the batch size
batch_size=32

# whether to write log files
write_to_log=0

# the learning rate
learning_rate=0.001

# weight decay
weight_decay=0.0

# the dropout keep probability
dropout_keep_prob=1.0

# noise sometimes helps training
noise_level=0.0

# the dimensions of the data
Dx=101
channels=1
input_dim=2
truncate_data_batches=0

# advanced use: don't use
make_videos=0
cudnn_benchmark=1

# the activation used throughout the networks internally
activation_str_general=celu

# the activation at the output (has to match the image of the scaler)
activation_str_output=tanhplus

# the scaler. Since it maps to [0,1] the activation at the output is tanhplus (check the code for definition)
scaler=MinMaxZeroOne

# the optimizer
optimizer_str=adabelief

# Whether the autoencoder is convolutional or not
AE_convolutional=0

# whether the autoencoder has batch normalization
AE_batch_norm=0

# whether the autoencoder uses transpose convolutions
AE_conv_transpose=0

# the type of pooling used (here averaged)
AE_pool_type="avg"

# the dimension of the latent state
latent_state_dim=2

# whether to plot exaples from some initial conditions
plot_testing_ics_examples=1

# whether to use a beta variational autoencoder (not extensively tested)
beta_vae=0

# the weight of the beta variational autoencoder
beta_vae_weight_max=1.0

# the random seed
random_seed=7

# for advanced use: additional loss encouraging smoothness on latent space
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.0

# whether to retrain
retrain=0

# for advanced use: the convolutional architecture to use 
AE_conv_architecture=conv_latent_1


# the prediction horizon for testing
prediction_horizon=1000

# the number of different initial conditions for testing
# num_test_ICS=1000
num_test_ICS=10

# the name of the system
system_name=FHN

# whether plotting is enabled
plotting=1

# the precision
precision=double


# advanced use: don't change
iterative_loss_validation=0
iterative_loss_schedule_and_gradient=none

# whether the iterative propagation in training is latent, don't change.
iterative_propagation_during_training_is_latent=1

# whether to train the RNN only (load trained AE)
train_RNN_only=0

# whether to load a trained AE (should have been trained first)
load_trained_AE=0

# WHether to add a reconstruction loss
reconstruction_loss=0

# Whether to add a forecasting loss at the decoded output
output_forecasting_loss=1

# Whether to add a forecasting loss at the encoded output (RNN only / no AE)
latent_forecasting_loss=1

# the cuda devide to use
CUDA_DEVICES=0

# the noise level used in the RNN tranining
noise_level=0.0

# scaler used in the latent space (don't change)
latent_space_scaler=Standard

# the cell type of the RNN
RNN_cell_type="lstm"

# the number of RNN layers
RNN_layers_num=1

# the size of the RNN layers
RNN_layers_size=32

# the activation at the output of the RNN
RNN_activation_str_output="tanhplus"

# the sequence length of the RNN
sequence_length=40

# the prediction length of the RNN
prediction_length=1

# the number of warm-up steps (to warm-up the hidden state)
n_warmup_train=0

# the warm up steps in testing
n_warmup=20


# which mode to run
# train: to train the network
# test: to test the network
# plot: to plot results
# all: to perform all aforementioned modes

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
# --iterative_state_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --iterative_propagation_during_training_is_latent $iterative_propagation_during_training_is_latent \
# --make_videos $make_videos \
# --retrain $retrain \
# --compute_spectrum 0 \
# --plot_state_distributions 1 \
# --plot_system 1 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --test_on_test 1 \
# --n_warmup_train $n_warmup_train \
# --n_warmup $n_warmup \
# --latent_space_scaler $latent_space_scaler \
# --precision $precision \
# --reference_train_time 20. \
# --buffer_train_time 1.0 \
# --test_on_test 1












########################################################################
########################################################################
# Multiscale testing
#########################################################################
#########################################################################

# whether to truncate the time sequences (for debugging). Set to zero.
# truncate_timesteps=20
truncate_timesteps=0

# the prediction horizon
prediction_horizon=4

# the number of steps to warm up the hidden state of the RNN
n_warmup=2

# the number of testing initial conditions
num_test_ICS=1

# the mode to use, either multiscale, or plotMultiscale to plot the results
mode=multiscale
# mode=plotMultiscale

# whether to plot the comparison results of multiscale runs
plot_multiscale_results_comparison=1

# whether to plot testing initial condition examples
plot_testing_ics_examples=0

# whether to plot the latent dynamics
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

























