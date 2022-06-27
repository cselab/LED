#!/bin/bash -l

SYSTEM_NAME=cylRe100HR
EXPERIMENT_NAME=Experiment_Daint_Large






SYSTEM_NAME=cylRe100HR
MODEL_NAME=GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1
# for FIELD in "Evaluation_Data" "Figures" "Trained_Models"
# do
# mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done

FIELD=Evaluation_Data
MODEL_NAME=GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/*.pickle /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/iterative_latent_forecasting_test.h5 /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}


SYSTEM_NAME=cylRe100HR
mkdir -p /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.txt /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.pickle /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.h5 /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data




CUDA_DEVICES=1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode plot --system_name cylRe100HR --cudnn_benchmark 1 --write_to_log 1 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 0 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --precision single --AE_size_factor 2 --AE_convolutional 1 --AE_batch_norm 1 --AE_batch_norm_affine 0 --AE_conv_transpose 1 --AE_pool_type avg --AE_interp_subsampling_input 2 --AE_conv_architecture conv_latent_6 --noise_level 0.0 --sequence_length 25 --n_warmup_train 0 --n_warmup 10 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 16 --overfitting_patience 40 --max_epochs 10000 --max_rounds 10 --display_output 1 --random_seed 21 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 0 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 0 --test_on_train 0 --plot_state_distributions 0 --plot_system 1 --plot_latent_dynamics 0 --plot_testing_ics_examples 0 --reference_train_time 24.0 --buffer_train_time 12.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 4 --prediction_horizon 100 --plotting 1 --random_seed_in_AE_name 10 --train_RNN_only 1 --load_trained_AE 1 --RNN_activation_str tanh --RNN_activation_str_output tanhplus --RNN_cell_type lstm --RNN_layers_num 1 --latent_state_dim 4 --RNN_layers_size 32 --make_videos 1 --plot_errors_in_time 0;



CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode plot --system_name cylRe100HR --cudnn_benchmark 1 --write_to_log 1 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 0 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --precision single --AE_size_factor 2 --AE_convolutional 1 --AE_batch_norm 1 --AE_batch_norm_affine 0 --AE_conv_transpose 1 --AE_pool_type avg --AE_interp_subsampling_input 2 --AE_conv_architecture conv_latent_6 --noise_level 0.0 --sequence_length 25 --n_warmup_train 0 --n_warmup 10 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 16 --overfitting_patience 40 --max_epochs 10000 --max_rounds 10 --display_output 1 --random_seed 21 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 0 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 0 --test_on_val 0 --test_on_train 0 --plot_state_distributions 0 --plot_system 0 --plot_latent_dynamics 1 --plot_testing_ics_examples 1 --reference_train_time 24.0 --buffer_train_time 12.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 4 --prediction_horizon 100 --plotting 0 --random_seed_in_AE_name 10 --train_RNN_only 1 --load_trained_AE 1 --RNN_activation_str tanh --RNN_activation_str_output tanhplus --RNN_cell_type lstm --RNN_layers_num 1 --latent_state_dim 4 --RNN_layers_size 32 --make_videos 1 --plot_errors_in_time 0;

\begin{align}
&4\times 256 \times 512 \\
\to & 
20 \times 128 \times 256 \\
\to &  20 \times 64 \times 128 \\
\to &  20 \times 32 \times 64 \\
\to &  20 \times 16 \times 32 \\
\to &  20 \times 8 \times 16 \\
\to &  2 \times 4 \times 8 \\
\to &  4
\end{align}



\begin{align}
&2 \times 4 \times 8 \\
\to & 20 \times 8 \times 16 \\
\to & 20 \times 16 \times 32 \\
\to & 20 \times 32 \times 64 \\
\to & 20 \times 64 \times 128 \\
\to & 20 \times 128 \times 256 \\
\to & 4 \times 512 \times 1024 \\
\to & 4 \times 512 \times 1024]
\end{align}








SYSTEM_NAME=cylRe1000HR
MODEL_NAME=GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1

for FIELD in "Evaluation_Data" "Figures" "Trained_Models"
do
mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
done


FIELD=Evaluation_Data
MODEL_NAME=GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/*.pickle /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/iterative_latent_forecasting_test.h5 /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}


/scratch/snx3000/pvlachas/LED/Code/Results/cylRe1000HR/Evaluation_Data/GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1



SYSTEM_NAME=cylRe1000HR
mkdir -p /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.txt /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.pickle /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/*.h5 /scratch/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data





CUDA_DEVICES=1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode plot --system_name cylRe1000HR --cudnn_benchmark 1 --write_to_log 0 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 0 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --precision single --AE_size_factor 2 --AE_convolutional 1 --AE_batch_norm 1 --AE_batch_norm_affine 0 --AE_conv_transpose 1 --AE_pool_type avg --AE_interp_subsampling_input 2 --AE_conv_architecture conv_latent_6 --noise_level 0.0 --sequence_length 25 --n_warmup_train 0 --n_warmup 20 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 16 --overfitting_patience 40 --max_epochs 10000 --max_rounds 10 --display_output 1 --random_seed 21 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 0 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 0 --test_on_train 0 --plot_state_distributions 0 --plot_system 1 --plot_latent_dynamics 0 --plot_testing_ics_examples 0 --reference_train_time 24.0 --buffer_train_time 12.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 4 --prediction_horizon 100 --plotting 1 --random_seed_in_AE_name 10 --train_RNN_only 1 --load_trained_AE 1 --RNN_activation_str tanh --RNN_activation_str_output tanhplus --RNN_cell_type lstm --RNN_layers_num 1 --latent_state_dim 10 --RNN_layers_size 32 --make_videos 1 --plot_errors_in_time 0;


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py crnn  --mode plot --system_name cylRe1000HR --cudnn_benchmark 1 --write_to_log 0 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str adabelief --output_forecasting_loss 0 --latent_forecasting_loss 1 --reconstruction_loss 0 --activation_str_general celu --activation_str_output tanhplus --precision single --AE_size_factor 2 --AE_convolutional 1 --AE_batch_norm 1 --AE_batch_norm_affine 0 --AE_conv_transpose 1 --AE_pool_type avg --AE_interp_subsampling_input 2 --AE_conv_architecture conv_latent_6 --noise_level 0.0 --sequence_length 25 --n_warmup_train 0 --n_warmup 20 --scaler MinMaxZeroOne --learning_rate 0.001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size 16 --overfitting_patience 40 --max_epochs 10000 --max_rounds 10 --display_output 1 --random_seed 21 --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 0 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 0 --test_on_val 0 --test_on_train 0 --plot_state_distributions 0 --plot_system 0 --plot_latent_dynamics 1 --plot_testing_ics_examples 1 --reference_train_time 24.0 --buffer_train_time 12.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 4 --prediction_horizon 100 --plotting 1 --random_seed_in_AE_name 10 --train_RNN_only 1 --load_trained_AE 1 --RNN_activation_str tanh --RNN_activation_str_output tanhplus --RNN_cell_type lstm --RNN_layers_num 1 --latent_state_dim 10 --RNN_layers_size 32 --make_videos 1 --plot_errors_in_time 0;



/scratch/pvlachas/LED/Code/Results/cylRe1000HR/Evaluation_Data/GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1
/scratch/snx3000/pvlachas/LED/Code/Results/cylRe1000HR/Evaluation_Data/GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1/iterative_latent_forecasting_test.h5







[utils_plotting] PLOTTING HOSTNAME: barry.ethz.ch
[utils_plotting] CLUSTER=True, CLUSTER_NAME=barry
[utils_plotting] Matplotlib Version = 3.4.2
[Config] RUNNING IN BARRY CLUSTER.
[Config] HOME PATH = /home/pvlachas
[Config] SCRATCH PATH = /scratch/pvlachas
[Config] PROJECT PATH = /scratch/pvlachas/LED/Code
[RUN] Python : 3.8.3 (default, Jun 14 2020, 13:52:37)
[GCC 10.1.0]
[RUN] Torch  : 1.9.0+cu102
[RUN] ######################################
[RUN] ##########    crnn    ##########
[RUN] ######################################
[crnn] Imported Horovod.
[utils_processing] Reference train time:
[utils_processing] 12[h]:0[m]:0[s]
[CONV-NET]: Loading architecture: conv_latent_6
[CNN-MLP-AE] with padding to keep the dimensionality.
[Encoder] -------------------
[Encoder] printDimensions() #
[Encoder] [4, 512, 1024]
[Encoder] [4, 256, 512]
[Encoder] [20, 128, 256]
[Encoder] [20, 64, 128]
[Encoder] [20, 32, 64]
[Encoder] [20, 16, 32]
[Encoder] [20, 8, 16]
[Encoder] [2, 4, 8]
[Encoder] [4]
[Encoder] -------------------
[CONV-NET]: Loading architecture: conv_latent_6
[CNN-Decoder] with padding to keep the dimensionality.
[crnn_model] # AUTOENCODER #
[Encoder] -------------------
[Encoder] printDimensions() #
[Encoder] [4, 512, 1024]
[Encoder] [4, 256, 512]
[Encoder] [20, 128, 256]
[Encoder] [20, 64, 128]
[Encoder] [20, 32, 64]
[Encoder] [20, 16, 32]
[Encoder] [20, 8, 16]
[Encoder] [2, 4, 8]
[Encoder] [4]
[Encoder] -------------------
[Decoder] -------------------
[Decoder] # printDimensions() #
[Decoder] [4]
[Decoder] [2, 4, 8]
[Decoder] [20, 8, 16]
[Decoder] [20, 16, 32]
[Decoder] [20, 32, 64]
[Decoder] [20, 64, 128]
[Decoder] [20, 128, 256]
[Decoder] [4, 512, 1024]
[Decoder] [4, 512, 1024]
[Decoder] -------------------
[Encoder] -------------------
[Encoder] printDimensions() #
[Encoder] [4, 512, 1024]
[Encoder] [4, 256, 512]
[Encoder] [20, 128, 256]
[Encoder] [20, 64, 128]
[Encoder] [20, 32, 64]
[Encoder] [20, 16, 32]
[Encoder] [20, 8, 16]
[Encoder] [2, 4, 8]
[Encoder] [4]
[Encoder] -------------------
[crnn_model] Network has MinMaxZeroOne latent state scaler.
[crnn_model] setModelToHalfPrecision()
[crnn_model] Sending model to CUDA.
[crnn_model] module_list :
[crnn_model] [ModuleList(
[crnn_model]   (0): interpolationLayer()
[crnn_model]   (1): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (2): Conv2d(4, 20, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (3): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=True)
[crnn_model]   (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (5): CELU(alpha=1.0)
[crnn_model]   (6): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (7): Conv2d(20, 20, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (8): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=True)
[crnn_model]   (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (10): CELU(alpha=1.0)
[crnn_model]   (11): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (12): Conv2d(20, 20, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (13): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=True)
[crnn_model]   (14): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (15): CELU(alpha=1.0)
[crnn_model]   (16): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (17): Conv2d(20, 20, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (18): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=True)
[crnn_model]   (19): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (20): CELU(alpha=1.0)
[crnn_model]   (21): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (22): Conv2d(20, 20, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (23): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=True)
[crnn_model]   (24): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (25): CELU(alpha=1.0)
[crnn_model]   (26): ZeroPad2d(padding=(6, 6, 6, 6), value=0.0)
[crnn_model]   (27): Conv2d(20, 2, kernel_size=(13, 13), stride=(1, 1))
[crnn_model]   (28): AvgPool2d(kernel_size=2, stride=2, padding=0)
[crnn_model]   (29): CELU(alpha=1.0)
[crnn_model]   (30): Flatten(start_dim=-3, end_dim=-1)
[crnn_model]   (31): Linear(in_features=64, out_features=4, bias=True)
[crnn_model]   (32): CELU(alpha=1.0)
[crnn_model] ), ModuleList(), ModuleList(
[crnn_model]   (0): Linear(in_features=4, out_features=64, bias=True)
[crnn_model]   (1): CELU(alpha=1.0)
[crnn_model]   (2): ViewModule()
[crnn_model]   (3): ConvTranspose2d(2, 20, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (4): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=False)
[crnn_model]   (5): CELU(alpha=1.0)
[crnn_model]   (6): ConvTranspose2d(20, 20, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (7): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=False)
[crnn_model]   (8): CELU(alpha=1.0)
[crnn_model]   (9): ConvTranspose2d(20, 20, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (10): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=False)
[crnn_model]   (11): CELU(alpha=1.0)
[crnn_model]   (12): ConvTranspose2d(20, 20, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (13): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=False)
[crnn_model]   (14): CELU(alpha=1.0)
[crnn_model]   (15): ConvTranspose2d(20, 20, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (16): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=0, track_running_stats=False)
[crnn_model]   (17): CELU(alpha=1.0)
[crnn_model]   (18): ConvTranspose2d(20, 4, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), output_padding=(1, 1))
[crnn_model]   (19): interpolationLayer()
[crnn_model]   (20): Tanhplus(
[crnn_model]     (tanh): Tanh()
[crnn_model]   )
[crnn_model] ), ModuleList(
[crnn_model]   (0): ZoneoutLayer(
[crnn_model]     (RNN_cell): LSTMCell(4, 32)
[crnn_model]   )
[crnn_model] ), ModuleList(
[crnn_model]   (0): Linear(in_features=32, out_features=4, bias=True)
[crnn_model]   (1): Tanhplus(
[crnn_model]     (tanh): Tanh()
[crnn_model]   )
[crnn_model] )]
[crnn] - model_name:
[crnn] GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1
[crnn] Trainable params 587142/587142
[crnn_model] Initializing parameters.
[crnn_model] Parameters initialized.
[crnn] USING CUDA -> SENDING THE MODEL TO THE GPU.
[crnn_model] Sending model to CUDA.































