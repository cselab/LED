#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=0

# whether to include the random seed in the name
random_seed_in_name=1

# maximum number of epochs to train
max_epochs=10000

# maximum rounds reducing the learning rate
max_rounds=20

# the overfitting patience of the learning rate scheduler
overfitting_patience=10

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

# whether to truncate the data batches (set to 0 to use all data and get a good result)
truncate_data_batches=0

# advanced use: don't use
make_videos=0

# advanced use: don't use
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

# whether to add a reconstruction loss (should be 1 if the autoencoder is to be trained)
reconstruction_loss=1

# whether to add a forecasting loss at the decoded output (should be 1 if the RNN is to be trainwed)
output_forecasting_loss=0

# the sequence length (if the autoencoder is trained only, it should be 1, if the RNN is trained also, it should be >>1, approx. 10-50)
sequence_length=1

# whether to plot exaples from some initial conditions
plot_testing_ics_examples=1

# whether to use a beta variational autoencoder (not extensively tested)
beta_vae=0

# the weight of the beta variational autoencoder
beta_vae_weight_max=1.0

# the random seed
random_seed=10

# for advanced use: additional loss encouraging smoothness on latent space
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.1

# whether to retrain
retrain=0

# for advanced use: the convolutional architecture to use 
AE_conv_architecture=conv_latent_1

# the prediction horizon for testing
prediction_horizon=1000

# the number of different initial conditions for testing
num_test_ICS=1000

# the name of the system
system_name=FHN

# whether plotting is enabled
plotting=1

# the precision
precision=double

# which mode to run
# train: to train the network
# test: to test the network
# plot: to plot results
# all: to perform all aforementioned modes

mode=all
# mode=train
# mode=plot
# mode=test


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







