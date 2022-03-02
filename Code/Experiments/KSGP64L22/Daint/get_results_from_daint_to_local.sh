#!/bin/bash -l

SYSTEM_NAME=KSGP64L22


EXPERIMENT_NAME=Experiment_Daint_Large


mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs
echo '###############		   COPY Logs			###############';
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logs/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs


for SYSTEM_NAME in KSGP64L22
do

echo '###############		   COPY Logfiles			###############';
mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logfiles/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles 
# echo '###############		   COPY Figures			###############';
# mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Figures/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures 
done





# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_16" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4"
SYSTEM_NAME=KSGP64L22
for MODEL_NAME in \
"ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1"
do
# FIELD="Figures"
# FIELD="Evaluation_Data"
FIELD="Trained_Models"
GPU_MODEL_NAME="GPU-"${MODEL_NAME}
mkdir -p /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${GPU_MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
done











# |      1 | 0.0134611 | 0.116014 | 0.999755 | GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4     |
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x128-SL_100-LFO_1-LFL_1"
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_6-PRETRAIN-AE_1-RS_7-C_lstm-R_1x128-SL_100-LFO_1-LFL_1"
# GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x128-SL_25-LFO_1-LFL_1

# SYSTEM_NAME=KSGP64L22
# for MODEL_NAME in \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x128-SL_25-LFO_1-LFL_1"
# do
# FIELD="Figures"
# # FIELD="Evaluation_Data"
# # FIELD="Trained_Models"
# mkdir -p /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/"GPU-"${MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/
# done


# SYSTEM_NAME=KSGP64L22
# for MODEL_NAME in \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_2"
# do
# # FIELD="Figures"
# # FIELD="Evaluation_Data"
# FIELD="Trained_Models"
# GPU_MODEL_NAME="GPU-"${MODEL_NAME}
# mkdir -p /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${GPU_MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done



# SYSTEM_NAME=KSGP64L22
# for MODEL_NAME in \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_16" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4"
# do
# # FIELD="Figures"
# # FIELD="Evaluation_Data"
# FIELD="Trained_Models"
# GPU_MODEL_NAME="GPU-"${MODEL_NAME}
# mkdir -p /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${GPU_MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done



