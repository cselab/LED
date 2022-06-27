#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --job-name=cylStudy_HVD_AECNN
#SBATCH --output=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_HVD_AECNN_outputfile_JID%j_A%a.txt
#SBATCH --error=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_HVD_AECNN_errorfile_JID%j_A%a.txt
#SBATCH --gres=gpu:0,craynetwork:1

# ======START=====

module load daint-gpu
module load PyTorch/1.9.0-CrayGNU-20.11
source ${HOME}/venv-python3.8-pytorch1.9/bin/activate

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1


# ----- TRAINING -----
multinode=1
mode=train
# --------------------

# # ----- TESTING -----
# multinode=0
# mode=test
# # -------------------

optimizer_str=adam
# optimizer_str=adabelief
# batch_size=32
# max_epochs=2
sequence_length=250
prediction_horizon=2
truncate_data_batches=0

hvd_compression=1
hvd_adasum=1
precision=single

AE_conv_architecture=conv_latent_6

AE_size_factor=2


random_seed=9
latent_state_dim=10
batch_size=64
max_epochs=2
sequence_length=25

random_seed=${random_seed}${batch_size}


# truncate_data_batches=16
# batch_size=16
# mode=train
# # mode=all

cd /users/pvlachas/LED/Code/Methods/

system_name=cylRe1000HRLarge

srun --wait 0 python3 RUN.py crnn --multinode $multinode --hvd_compression $hvd_compression --hvd_adasum $hvd_adasum --precision $precision --mode $mode --system_name $system_name --cudnn_benchmark 1 --write_to_log 1 --channels 2 --input_dim 4 --Dx 1024 --Dy 512 --optimizer_str $optimizer_str --output_forecasting_loss 0 --latent_forecasting_loss 0 --reconstruction_loss 1 --activation_str_general celu --activation_str_output tanhplus --AE_size_factor $AE_size_factor --AE_convolutional 1 --AE_batch_norm 1 --AE_batch_norm_affine 0 --AE_conv_transpose 1 --AE_pool_type avg --AE_conv_architecture $AE_conv_architecture  --AE_interp_subsampling_input 2 --latent_state_dim $latent_state_dim --noise_level 0.0 --sequence_length $sequence_length --n_warmup_train 0 --n_warmup 0 --scaler MinMaxZeroOne --learning_rate 0.0001 --weight_decay 0.0 --dropout_keep_prob 1.0 --batch_size $batch_size --overfitting_patience 40 --max_epochs $max_epochs --max_rounds 20 --display_output 1 --random_seed $random_seed --random_seed_in_name 1 --make_videos 0 --retrain 0 --compute_spectrum 0 --teacher_forcing_forecasting 1 --iterative_latent_forecasting 1 --iterative_state_forecasting 0 --test_on_test 1 --test_on_val 1 --test_on_train 0 --plot_state_distributions 0 --plot_system 1 --plot_latent_dynamics 1 --plot_testing_ics_examples 1 --reference_train_time 24.0 --buffer_train_time 6.0 --c1_latent_smoothness_loss 0 --c1_latent_smoothness_loss_factor 0.0 --iterative_loss_schedule_and_gradient none --iterative_loss_validation 0 --iterative_propagation_during_training_is_latent 1 --num_test_ICS 1 --prediction_horizon $prediction_horizon --plotting 1 --truncate_data_batches $truncate_data_batches;



