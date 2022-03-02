#!/bin/bash -l


ssh hippo << 'EOF'
    
    SYSTEM_NAME_LOGS=KSGP64L22
    
    for SYSTEM_NAME in KSGP64L22
    do

    mkdir -p /data1/users/pvlachas/STF/Code/Data/${SYSTEM_NAME}/Data
    tmux new -s copying_data_${SYSTEM_NAME} -d
    tmux send-keys -t copying_data_${SYSTEM_NAME}.0 "\
    echo '###############           COPY Data            ###############'; \
    rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Data/${SYSTEM_NAME}/Data/ /data1/users/pvlachas/STF/Code/Data/${SYSTEM_NAME}/Data; \
    " ENTER

    
    EXPERIMENT_NAME=Experiment_Daint_Large
    
    mkdir -p /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}

    mkdir -p /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logs

    mkdir -p /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles

    mkdir -p /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures

    mkdir -p /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Trained_Models

    tmux new -s copying_results_${SYSTEM_NAME} -d
    # tmux send-keys -t copying_results_${SYSTEM_NAME}.0 "rsync -mzarvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}" ENTER

    tmux send-keys -t copying_results_${SYSTEM_NAME}.0 "\
    echo '###############           COPY Logs            ###############'; \
    rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME_LOGS}/Logs/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME_LOGS}/Logs; \
    echo '###############           COPY Logfiles            ###############'; \
    rsync -mzarvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Logfiles/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Logfiles; \
    echo '###############           COPY Figures         ###############'; \
    rsync -mzarvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Figures/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Figures; \
    echo '###############           COPY Trained_Models          ###############'; \
    rsync -mzarvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Trained_Models/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Trained_Models; \
    echo '###############           COPY Evaluation_Data         ###############'; \
    rsync -marvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/Evaluation_Data/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}/Evaluation_Data; \
    echo '###############           COPY FOLDER         ###############'; \
    rsync -marvP --size-only daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/ /data1/users/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}; \
    " ENTER

    done

    exit
EOF
