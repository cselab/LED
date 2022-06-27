#!/bin/bash -l




EXPERIMENT_NAME=Experiment_Daint_Large

SYSTEM_NAME=cylReLEDStudyHR

mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs
echo '###############		   COPY Logs			###############';
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Logs/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs


for SYSTEM_NAME in cylRe1000HRLarge
do

echo '###############          COPY Logfiles            ###############';
mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Logfiles/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles 
echo '###############          COPY Figures         ###############';
mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Figures/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures 
done



# for SYSTEM_NAME in cylRe100HR cylRe1000HR cylRe100HRDt005 cylRe1000HRDt005
# # for SYSTEM_NAME in cylRe100HR
# do

# echo '###############		   COPY Logfiles			###############';
# mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Logfiles/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles 
# echo '###############		   COPY Figures			###############';
# mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Figures/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures 
# done






# # |      1 | 0.0134611 | 0.116014 | 0.999755 | GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4     |

# SYSTEM_NAME=cylRe100HR
# for MODEL_NAME in \
# "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_3"
# do
# FIELD="Figures"
# # FIELD="Evaluation_Data"
# # FIELD="Trained_Models"
# mkdir -p /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/"GPU-"${MODEL_NAME}/ /Users/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}/
# done




# SYSTEM_NAME=cylRe100HR
# for MODEL_NAME in \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_20-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_2"
# do
# # FIELD="Figures"
# # FIELD="Evaluation_Data"
# FIELD="Trained_Models"
# GPU_MODEL_NAME="GPU-"${MODEL_NAME}
# mkdir -p /Users/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${GPU_MODEL_NAME}/ /Users/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done



# SYSTEM_NAME=cylRe100HR
# for MODEL_NAME in \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_16" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8" \
# "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0005-L2_0.0-RS_10-CHANNELS_1-4-8-16-32-1-KERNELS_11-9-7-5-3-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_4"
# do
# # FIELD="Figures"
# # FIELD="Evaluation_Data"
# FIELD="Trained_Models"
# GPU_MODEL_NAME="GPU-"${MODEL_NAME}
# mkdir -p /Users/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${GPU_MODEL_NAME}/ /Users/pvlachas/LED/Code/Results/${SYSTEM_NAME}/${FIELD}/${MODEL_NAME}
# done



