#!/bin/bash -l

SYSTEM_NAME=FHN


EXPERIMENT_NAME=Experiment_Daint_Large


mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs
echo '###############		   COPY Logs			###############';
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logs/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs


# for SYSTEM_NAME in FHN-1000 FHN-100
for SYSTEM_NAME in FHN
do

echo '###############		   COPY Logfiles			###############';
mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logfiles/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles 
# echo '###############		   COPY Figures			###############';
# mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Figures/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures 
done




# SYSTEM_NAME=FHN
# for MODEL_NAME in \
# "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_3-SOLVER_pinv-SIZE_1000-DEG_10-R_0.8-S_0.5-REG_1e-05-NS_10" \
# "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_3-TYPE_continuous-PO_5-THRES_1e-05-LIB_poly-INT_5" \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
# do
# FIELD="Evaluation_Data"
# mkdir -p /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}//${FIELD}/${MODEL_NAME} 
# done






SYSTEM_NAME=FHN
for MODEL_NAME in \
"ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
do
# FIELD="Figures"
# FIELD="Evaluation_Data"
FIELD="Trained_Models"
mkdir -p /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/"GPU-"${MODEL_NAME}/ /Users/pvlachas/STF/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/
done







