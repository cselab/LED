#!/bin/bash -l


SYSTEM_NAME=cylReLEDStudyHR

################################################
### Synchronize git (push local)
################################################
cd $HOME/STF
git add .
git commit -m "running system (HOROVOD)"
git push


################################################
### RUN
################################################

ssh daint << 'EOF'
	cd ${HOME}/STF
	git stash save --keep-index
	git stash drop
	git pull

	SYSTEM_NAME=cylReLEDStudyHR

	echo "PREPARING JOB..."
	mkdir -p /scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logs

	# cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Horovod

    # sbatch HVD_debug.sh

    # cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Horovod100

    # sbatch HVD_AECNN_1.sh
    # sbatch HVD_AECNN_2.sh
    # sbatch HVD_AECNN_3.sh
    # sbatch HVD_AECNN_4.sh
    # sbatch HVD_AECNN_5.sh
    # sbatch HVD_AECNN_6.sh


    cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Horovod1000

    # sbatch HVD_AECNN_1.sh
    sbatch HVD_AECNN_2.sh
    sbatch HVD_AECNN_3.sh
    sbatch HVD_AECNN_4.sh
    sbatch HVD_AECNN_5.sh
    sbatch HVD_AECNN_6.sh


	exit
EOF











