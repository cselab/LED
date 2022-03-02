#!/bin/bash -l


SYSTEM_NAME=KSGP64L22

EXPERIMENT_NAME=Experiment_Barry

mkdir -p $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs

# COPY LOGS
rsync -mzarvP barry:/scratch/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs/ $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs


SYSTEM_NAME=KSGP64L22

mkdir -p $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles

mkdir -p $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures

    
# COPY LOG-FILES RESULTS
rsync -mzarvP barry:/scratch/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logfiles/ $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles

# COPY Figures
rsync -mzarvP barry:/scratch/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Figures/ $HOME/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures





