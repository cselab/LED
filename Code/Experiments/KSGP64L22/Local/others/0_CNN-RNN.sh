#!/bin/bash

cd ../../../Methods

    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1",

python3 RUN.py crnn --mode plot --system_name KSGP64L22 --cudnn_benchmark 1 --write_to_log 1 --channels 1 --input_dim 1 --Dx 64 --optimizer_str adabelief --output_forecasting_loss 0 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --AE_size_factor 2 --AE_convolutional 1 --AE_batch_norm 0 --AE_conv_transpose 0 --AE_pool_type avg --AE_conv_architecture conv_latent_1 --latent_state_dim 8 --noise_level 0.0 --sequence_length 50 --n_warmup_train 0 --n_warmup 60 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 32 --overfitting_patience 40 --max_epochs 20000 --max_rounds 20 --display_output 1 --random_seed 7 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 1 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 0 --test_on_train 0 --plot_state_distributions 1 --plot_system 0 --plot_latent_dynamics 0 --plot_testing_ics_examples 1 --reference_train_time 20.0 --buffer_train_time 1.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 1 --prediction_horizon 3200 --plotting 1 --random_seed_in_AE_name 30 --train_RNN_only 1 --load_trained_AE 1 --RNN_cell_type lstm --RNN_layers_num 1 --RNN_layers_size 512 --RNN_activation_str_output tanhplus;