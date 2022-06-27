#!/bin/bash -l


SYSTEM_NAME=cylRe100
# SYSTEM_NAME=cylRe1000

cd ${HOME}/LED/Code/PostProcessing/LED

################################################
### Synchronize git (push local)
################################################
git add .
git commit -m "making figures for system $SYSTEM_NAME"
git push


################################################
### RUN
################################################

ssh barry << 'EOF'

cd ${SCRATCH}/LED/Code/PostProcessing/LED

    git stash save --keep-index
    git stash drop
    git pull

    SYSTEM_NAME=cylRe100
    # SYSTEM_NAME=cylRe1000

    module load python/3.8.3
    source $HOME/venv-python-3.8/bin/activate

    python3 F3_field_wrt_multiscale_ratio.py ${SYSTEM_NAME} 1 0 None

	exit
EOF



# Get data back

REMOTE_EXPERIMENT_NAME=None
LOCAL_EXPERIMENT_NAME=Experiment_Barry
mkdir -p /Users/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${LOCAL_EXPERIMENT_NAME}/Data
rsync -mzarvP barry:/scratch/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${REMOTE_EXPERIMENT_NAME}/Data/ /Users/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${LOCAL_EXPERIMENT_NAME}/Data



cd ${HOME}/LED/Code/PostProcessing/LED

python3 F3_field_wrt_multiscale_ratio.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}




