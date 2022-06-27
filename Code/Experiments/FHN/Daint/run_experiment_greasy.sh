#!/bin/bash -l
SYSTEM_NAME=FHN

cd ${HOME}/LED/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy

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

python3 P1_greasy_AE_RNN_end2end_make_tasks.py



################################################
### Synchronize git (push local)
################################################
cd $HOME/LED
git add .
git commit -m "running system $SYSTEM_NAME"
git push


################################################
### RUN
################################################

ssh daint << 'EOF'
	cd ${HOME}/LED
	git stash save --keep-index
	git stash drop
	git pull

	SYSTEM_NAME=FHN

	echo "PREPARING JOB..."
	mkdir -p /scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/Logs

	cd ${HOME}/LED/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy



    # sbatch S0_greasy_AE.sh 1

    # sbatch S1_greasy_PCA.sh 1

    # sbatch S2_greasy_DMAPS.sh 1

    # sbatch S3_greasy_AECNN.sh 1





    # sbatch P0_greasy_AE_RNN.sh 1

    # sbatch P1_greasy_AE_RNN_end2end.sh 1

    # sbatch P2_greasy_AE_RC.sh 1

    # sbatch P3_greasy_AE_SINDy.sh 1

    # sbatch P4_greasy_AE_MLP.sh 1




    sbatch P1_greasy_AE_RNN_end2end.sh 1


	exit
EOF


# NUM_JOBS=8
# for batch_id in $(seq 1 $NUM_JOBS);
# do
#     echo "-----------------------------------"
#     echo "# Running jobs of batch "$batch_id;
#     # echo sbatch S2_greasy_DMAPS.sh $batch_id;
#     sbatch S2_greasy_DMAPS.sh $batch_id
# done








