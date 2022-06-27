#!/bin/bash -l


# SYSTEM_NAME=FHN
# SYSTEM_NAME=KSGP64L22
# SYSTEM_NAME=KSGP64L22Large
# SYSTEM_NAME=cylRe100
# SYSTEM_NAME=cylRe1000

# SYSTEM_NAME=cylRe100HR
# SYSTEM_NAME=cylRe1000HR
# SYSTEM_NAME=cylRe100HRDt005
# SYSTEM_NAME=cylRe1000HRDt005

# mkdir -p /Users/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/ 
# rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/dt.txt /Users/pvlachas/LED/Code/Data/${SYSTEM_NAME}/Data/ 



# cd ${HOME}/LED/Code/PostProcessing/LED

# ################################################
# ### Synchronize git (push local)
# ################################################
# git add .
# git commit -m "making figures for system $SYSTEM_NAME"
# git push












# ssh daint << 'EOF'
# 	cd ${HOME}/LED/Code/PostProcessing/LED

# 	git stash save --keep-index
# 	git stash drop
# 	git pull

#     # SYSTEM_NAME=FHN
#     # SYSTEM_NAME=KSGP64L22
#     # SYSTEM_NAME=KSGP64L22Large
#     # SYSTEM_NAME=cylRe100
#     # SYSTEM_NAME=cylRe1000
    
#     # SYSTEM_NAME=cylRe100HR
#     # SYSTEM_NAME=cylRe1000HR
#     # SYSTEM_NAME=cylRe100HRDt005
#     # SYSTEM_NAME=cylRe1000HRDt005
#     # rm -rf ./${SYSTEM_NAME}/${DUMMY_EXPERIMENT_NAME}/Data

#     module load daint-gpu
#     module load PyTorch/1.9.0-CrayGNU-20.11
#     source ${HOME}/venv-python3.8-pytorch1.9/bin/activate

#     DUMMY_EXPERIMENT_NAME=None


#     # for SYSTEM_NAME in cylRe100HR cylRe1000HR cylRe100HRDt005 cylRe1000HRDt005
#     # for SYSTEM_NAME in cylRe100HR cylRe1000HR
#     for SYSTEM_NAME in cylRe100HR
#     # for SYSTEM_NAME in FHN
#     do
#         rm -rf ./${SYSTEM_NAME}/${DUMMY_EXPERIMENT_NAME}/Data
        
#         python3 F2_field_wrt_models.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME
#         # python3 F3_field_wrt_multiscale_ratio.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME
#         # python3 F5_KS_error_wrt_models_and_time.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME

#         # python3 F9_latent_dynamics_comparison_PCA.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME
#         # python3 F9c_attractor_comparison_KS.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME
        
#         # python3 F10_contours_cylRe.py ${SYSTEM_NAME} 1 0 $DUMMY_EXPERIMENT_NAME
#     done

# 	exit
# EOF







# for SYSTEM_NAME in cylRe100HR cylRe1000HR cylRe100HRDt005 cylRe1000HRDt005
for SYSTEM_NAME in cylRe100HR cylRe1000HR
# for SYSTEM_NAME in cylRe100HR
# for SYSTEM_NAME in FHN
do
    REMOTE_EXPERIMENT_NAME=None
    LOCAL_EXPERIMENT_NAME=Experiment_Daint_Large
    mkdir -p /Users/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${LOCAL_EXPERIMENT_NAME}/Data
    # rsync -mzarvP daint:/users/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${REMOTE_EXPERIMENT_NAME}/Data/ /Users/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/${LOCAL_EXPERIMENT_NAME}/Data

    cd ${HOME}/LED/Code/PostProcessing/LED

    # python3 F2_field_wrt_models.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}
    # python3 F3_field_wrt_multiscale_ratio.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}
    # python3 F5_KS_error_wrt_models_and_time.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}

    # python3 F9_latent_dynamics_comparison_PCA.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}
    # python3 F9c_attractor_comparison_KS.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}
    python3 F11_latent_fourier.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}

    # python3 F10_contours_cylRe.py ${SYSTEM_NAME} 0 1 ${LOCAL_EXPERIMENT_NAME}
done






# # for SYSTEM_NAME in cylRe100HR cylRe1000HR cylRe100HRDt005 cylRe1000HRDt005
# for SYSTEM_NAME in cylRe100HR cylRe1000HR
# do
#     LOCAL_EXPERIMENT_NAME=Experiment_Daint_Large
#     python3 F4_speedup_wrt_multiscale_ratio.py ${SYSTEM_NAME} ${LOCAL_EXPERIMENT_NAME}
# done




# SYSTEM_NAME=FHN
# # SYSTEM_NAME=KSGP64L22
# # SYSTEM_NAME=KSGP64L22Large
# # SYSTEM_NAME=cylRe100
# # SYSTEM_NAME=cylRe1000
# # SYSTEM_NAME=cylRe1000HR
# LOCAL_EXPERIMENT_NAME=Experiment_Daint_Large
# python3 F4_speedup_wrt_multiscale_ratio.py ${SYSTEM_NAME} ${LOCAL_EXPERIMENT_NAME}












