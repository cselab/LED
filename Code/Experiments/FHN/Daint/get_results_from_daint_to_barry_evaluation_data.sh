#!/bin/bash -l

SYSTEM_NAME=FHN


EXPERIMENT_NAME=Experiment_Daint_Large



# "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10" \

SYSTEM_NAME=FHN
for MODEL_NAME in \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_40-LFO_0-LFL_1" \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1" \
"GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10" \
"GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_5" \
"GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_25-LFO_1-LFL_1"
do
FIELD="Evaluation_Data"
mkdir -p /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
done

