#!/bin/bash -l

SYSTEM_NAME=KSGP64L22
EXPERIMENT_NAME=Experiment_Barry

cd $HOME/STF
git add .
git commit -m "running experiment ${EXPERIMENT_NAME}"
git push


ssh barry << 'EOF'

    # SYSTEM_NAME=MNISToy
    SYSTEM_NAME=KSGP64L22

    EXPERIMENT_NAME=Experiment_Barry
    mkdir -p /scratch/pvlachas/STF/Code/Data/${SYSTEM_NAME}/Data
    exit
EOF

rsync -mzarvP $HOME/STF/Code/Data/${SYSTEM_NAME}/Data/ barry:/scratch/pvlachas/STF/Code/Data/${SYSTEM_NAME}/Data


ssh barry << 'EOF'

    echo "(run_experiment_train.sh): SCRATCH FOLDER="$SCRATCH

    cd ${SCRATCH}/STF
    git stash save --keep-index
    git stash drop
    git pull

    SYSTEM_NAME=KSGP64L22
    
    EXPERIMENT_NAME=Experiment_Barry

    echo "(run_experiment_train.sh): PREPARING JOB..."
    echo "(run_experiment_train.sh): SYSTEM_NAME="$SYSTEM_NAME
    echo "(run_experiment_train.sh): EXPERIMENT_NAME="$EXPERIMENT_NAME
    
    cd ${SCRATCH}/STF/Code/Experiments/ScriptsBarry
    # bash barry_prepare_job.sh ${SYSTEM_NAME} ${EXPERIMENT_NAME} ${SCRATCH}


    SCRIPT_NAME=ARNN-end2end
    bash barry_run_script.sh ${SCRIPT_NAME} ${SYSTEM_NAME} ${EXPERIMENT_NAME} ${SCRATCH}
    
    SCRIPT_NAME=ARNN-sequential
    bash barry_run_script.sh ${SCRIPT_NAME} ${SYSTEM_NAME} ${EXPERIMENT_NAME} ${SCRATCH}

    SCRIPT_NAME=ConvRNN
    bash barry_run_script.sh ${SCRIPT_NAME} ${SYSTEM_NAME} ${EXPERIMENT_NAME} ${SCRATCH}

    exit
EOF




