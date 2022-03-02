#!/bin/bash -l




EXPERIMENT_NAME=Experiment_Daint_Large

SYSTEM_NAME=cylReLEDStudyHR


# |      1 | 0.0134611 | 0.116014 | 0.999755 | GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4     |

# SYSTEM_NAME=cylRe100HR
SYSTEM_NAME=cylRe1000HR
for MODEL_NAME in \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_3"
do
FIELD="Figures"
# FIELD="Evaluation_Data"
# FIELD="Trained_Models"
mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/
done
# /scratch/snx3000/pvlachas/STF/Code/Results/cylRe100HR/Figures/GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_3



