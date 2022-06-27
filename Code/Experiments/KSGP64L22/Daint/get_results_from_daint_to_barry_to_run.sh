#!/bin/bash -l

SYSTEM_NAME=KSGP64L22


echo '###############          COPY MODEL DATA            ###############';

# SYSTEM_NAME=KSGP64L22
# for MODEL_NAME in \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4" \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_2"
# do
# FIELD="Trained_Models"
# mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done

# SYSTEM_NAME=KSGP64L22
# for MODEL_NAME in \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_16" \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8" \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4"
# do
# FIELD="Trained_Models"
# mkdir -p /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done
