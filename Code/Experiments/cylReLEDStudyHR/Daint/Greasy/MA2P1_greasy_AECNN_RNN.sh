#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --job-name=cylStudy_MA2P1_greasy_AECNN_RNN
#SBATCH --output=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA2P1_greasy_AECNN_RNN_outputfile_JID%j_A%a.txt
#SBATCH --error=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA2P1_greasy_AECNN_RNN_errorfile_JID%j_A%a.txt
#SBATCH --gres=gpu:0,craynetwork:1

# ======START=====


module load daint-gpu
module load GREASY
# module swap PrgEnv-cray PrgEnv-gnu;
module load cray-hdf5-parallel cray-fftw cray-petsc cudatoolkit GSL cray-python
export HYPRE_ROOT=/users/novatig/hypre/build
export GSL_ROOT=/apps/daint/UES/jenkins/7.0.UP02/gpu/easybuild/software/GSL/2.5-CrayGNU-20.08
module load PyTorch/1.9.0-CrayGNU-20.11
source ${HOME}/venv-python3.8-pytorch1.9/bin/activate

export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export GREASY_LOGFILE="/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA2P1_B"$1"_greasy_AECNN_RNN_logfile_JID"${SLURM_JOBID}".txt"
echo "GREASY - Running greasy tasks from "
echo $PWD
echo $GREASY_LOGFILE
echo "## BATCH = B"$1" ##"



greasy ./Tasks/MA2P1_B$1_greasy_AECNN_RNN_tasks.txt




#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=debug


#SBATCH --time=24:00:00
#SBATCH --nodes=6
#SBATCH --partition=normal

#SBATCH --time=24:00:00
#SBATCH --nodes=20
#SBATCH --partition=normal

#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --partition=normal