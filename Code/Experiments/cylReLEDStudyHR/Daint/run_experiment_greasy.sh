#!/bin/bash -l
SYSTEM_NAME=cylReLEDStudyHR

cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy

################################################
### Create Tasks
################################################
# python3 A1_greasy_PCA_make_tasks.py
# python3 A2_greasy_AECNN_make_tasks.py
# python3 WW_greasy_CNNaSPACEoDYSSEY_make_tasks.py




# python3 A1P1_greasy_PCA_RNN_make_tasks.py
# python3 A1P2_greasy_PCA_RC_make_tasks.py
# python3 A1P3_greasy_PCA_SINDy_make_tasks.py

# python3 MA1P1_greasy_PCA_RNN_make_tasks.py

python3 A2P1_greasy_AECNN_RNN_make_tasks.py
# python3 A2P2_greasy_AECNN_RC_make_tasks.py
# python3 A2P3_greasy_AECNN_SINDy_make_tasks.py

# python3 MA2P1_greasy_AECNN_RNN_make_tasks.py

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

	SYSTEM_NAME=cylReLEDStudyHR

	echo "PREPARING JOB..."
	mkdir -p /scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logs

	cd ${HOME}/STF/Code/Experiments/${SYSTEM_NAME}/Daint/Greasy



    # sbatch A1_greasy_PCA.sh 1
    # sbatch A2_greasy_AECNN.sh 1





    # sbatch A1P1_greasy_PCA_RNN.sh 1
    # sbatch A1P2_greasy_PCA_RC.sh 1
    # sbatch A1P3_greasy_PCA_SINDy.sh 1

    # sbatch MA1P1_greasy_PCA_RNN.sh 1

    # NUM_JOBS=10
    # for batch_id in $(seq 1 $NUM_JOBS);
    # do
    #     sbatch MA1P1_greasy_PCA_RNN.sh $batch_id
    # done




    sbatch A2P1_greasy_AECNN_RNN.sh 1
    # sbatch A2P2_greasy_AECNN_RC.sh 1
    # sbatch A2P3_greasy_AECNN_SINDy.sh 1

    # sbatch MA2P1_greasy_AECNN_RNN.sh 1

    # NUM_JOBS=13
    # for batch_id in $(seq 1 $NUM_JOBS);
    # do
    #     sbatch MA2P1_greasy_AECNN_RNN.sh $batch_id
    # done


	exit
EOF







# python3 V1_greasy_PCACNN_make_tasks.py

# python3 C1_greasy_PCA_RNN_make_tasks.py

# python3 P0_greasy_AECNN_RNN_make_tasks.py
# python3 P2_greasy_AECNN_RC_make_tasks.py
# python3 P3_greasy_AECNN_SINDy_make_tasks.py
# python3 P4_greasy_AECNN_MLP_make_tasks.py

# python3 P0_greasy_AECNN_RNN_make_tasks.py
# python3 P0M0_greasy_AECNN_RNN_make_tasks.py


# NUM_JOBS=2
# for batch_id in $(seq 1 $NUM_JOBS);
# do
#     echo "-----------------------------------"
#     echo "# Running jobs of batch "$batch_id;
#     sbatch A1_greasy_PCA.sh $batch_id
# done





# # NUM_JOBS=12
# NUM_JOBS=6
# for batch_id in $(seq 1 $NUM_JOBS);
# do
#     echo "-----------------------------------"
#     echo "# Running jobs of batch "$batch_id;
#     sbatch WW_greasy_CNNaSPACEoDYSSEY.sh $batch_id
# done





# sbatch V1_greasy_PCACNN.sh 1

# NUM_JOBS=8
# for batch_id in $(seq 1 $NUM_JOBS);
# do
#     echo "-----------------------------------"
#     echo "# Running jobs of batch "$batch_id;
#     sbatch V1_greasy_PCACNN.sh $batch_id
# done




# sbatch C1_greasy_PCA_RNN.sh 1



# sbatch P0_greasy_AECNN_RNN.sh 1
# sbatch P2_greasy_AECNN_RC.sh 1
# sbatch P3_greasy_AECNN_SINDy.sh 1
# sbatch P4_greasy_AECNN_MLP.sh 1

# sbatch P0_greasy_AECNN_RNN.sh 1




# sbatch P0_greasy_AECNN_RNN.sh 1
# sbatch P0M0_greasy_AECNN_RNN.sh 1




