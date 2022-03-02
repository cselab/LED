#!/bin/bash

cd ../../../Methods

CUDA_DEVICES=3

module load python/3.8.3
source $HOME/venv-python-3.8/bin/activate

mode=all
num_test_ICS=1
# prediction_horizon=10
prediction_horizon=1000
plotting=1
plot_system=1
max_epochs=2

# TEST AUTOENCODER
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode test --system_name cylRe1000HR --cudnn_benchmark 1 --write_to_log 1 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 1 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --AE_size_factor 1 --AE_convolutional 1 --AE_batch_norm 1 --AE_conv_transpose 0 --AE_pool_type avg --AE_conv_architecture conv_latent_1 --latent_state_dim 3 --noise_level 0.0 --sequence_length 20 --n_warmup_train 0 --n_warmup 20 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 32 --overfitting_patience 20 --max_epochs $max_epochs --max_rounds 20 --display_output 1 --random_seed 10 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 1 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 1 --test_on_train 0 --plot_state_distributions 0 --plot_system $plot_system --plot_latent_dynamics 1 --plot_testing_ics_examples 1 --reference_train_time 20.0 --buffer_train_time 1.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS $num_test_ICS --prediction_horizon $prediction_horizon --plotting $plotting;

# TEST RNN
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode $mode --system_name cylRe1000HR --cudnn_benchmark 1 --write_to_log 1 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 1 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --AE_size_factor 1 --AE_convolutional 1 --AE_batch_norm 1 --AE_conv_transpose 0 --AE_pool_type avg --AE_conv_architecture conv_latent_1 --latent_state_dim 3 --noise_level 0.0 --sequence_length 20 --n_warmup_train 0 --n_warmup 20 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 32 --overfitting_patience 20 --max_epochs 10000 --max_rounds 20 --display_output 1 --random_seed 7 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 1 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 0 --test_on_train 0 --plot_state_distributions 0 --plot_system $plot_system --plot_latent_dynamics 1 --plot_testing_ics_examples 1 --reference_train_time 20.0 --buffer_train_time 1.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS $num_test_ICS --prediction_horizon $prediction_horizon --plotting $plotting --random_seed_in_AE_name 10 --train_RNN_only 1 --load_trained_AE 1 --RNN_cell_type lstm --RNN_layers_num 1 --RNN_layers_size 32 --RNN_activation_str_output tanhplus;

