#!/bin/bash -l


echo '###############          COPY MODEL DATA            ###############';

    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",

SYSTEM_NAME=cylRe100HR
for MODEL_NAME in \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2" \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
do
FIELD="Trained_Models"
mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
done

    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",

SYSTEM_NAME=cylRe1000HR
for MODEL_NAME in \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3" \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
do
FIELD="Trained_Models"
mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
done

