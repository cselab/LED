#!/bin/bash -l
SYSTEM_NAME=KSGP64L22

cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy

################################################
### Create Tasks
################################################
# python3 S0_greasy_AE_make_tasks.py
# python3 S1_greasy_PCA_make_tasks.py
# python3 S2_greasy_DMAPS_make_tasks.py
# python3 S3_greasy_AECNN_make_tasks.py

# python3 P0_greasy_AE_RNN_make_tasks.py
# python3 P1_greasy_AE_RNN_end2end_make_tasks.py
# python3 P2_greasy_AE_RC_make_tasks.py
# python3 P3_greasy_AE_SINDy_make_tasks.py
# python3 P4_greasy_AE_MLP_make_tasks.py


python3 P0_greasy_AE_RNN_make_tasks.py


# python3 P5_greasy_RNN_make_tasks.py
# python3 P6_greasy_RC_make_tasks.py



################################################
### Synchronize git (push local)
################################################
cd $HOME/STF
git add .
git commit -m "running system $SYSTEM_NAME"
git push


################################################
### RUN
################################################

ssh daint << 'EOF'
	cd ${HOME}/STF
	git stash save --keep-index
	git stash drop
	git pull

	SYSTEM_NAME=KSGP64L22

	echo "PREPARING JOB..."
	mkdir -p /scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logs

	cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy




    # sbatch S0_greasy_AE.sh 1
    # sbatch S1_greasy_PCA.sh 1
    # sbatch S2_greasy_DMAPS.sh 1
    # sbatch S3_greasy_AECNN.sh 1





    # sbatch P0_greasy_AE_RNN.sh 1

    # sbatch P1_greasy_AE_RNN_end2end.sh 1

    # sbatch P2_greasy_AE_RC.sh 1

    # sbatch P3_greasy_AE_SINDy.sh 1

    # sbatch P4_greasy_AE_MLP.sh 1



    sbatch P0_greasy_AE_RNN.sh 1



    # sbatch P5_greasy_RNN.sh 1

    # sbatch P6_greasy_RC.sh 1




	exit
EOF






