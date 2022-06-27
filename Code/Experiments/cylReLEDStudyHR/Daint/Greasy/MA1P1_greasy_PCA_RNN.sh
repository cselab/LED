#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --time=24:00:00
#SBATCH --nodes=20
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --job-name=cylStudy_MA1P1_greasy_PCA_RNN
#SBATCH --output=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA1P1_greasy_PCA_RNN_outputfile_JID%j_A%a.txt
#SBATCH --error=/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA1P1_greasy_PCA_RNN_errorfile_JID%j_A%a.txt

# ======START=====




module load daint-gpu
module load GREASY
# module swap PrgEnv-cray PrgEnv-gnu;
module load cray-hdf5-parallel cray-fftw cray-petsc cudatoolkit GSL cray-python
export HYPRE_ROOT=/users/novatig/hypre/build
export GSL_ROOT=/apps/daint/UES/jenkins/7.0.UP02/gpu/easybuild/software/GSL/2.5-CrayGNU-20.08
module load PyTorch/1.9.0-CrayGNU-20.11
source ${HOME}/venv-python3.8-pytorch1.9/bin/activate


export GREASY_LOGFILE="/scratch/snx3000/pvlachas/LED/Code/Results/cylReLEDStudyHR/Logs/cylStudy_MA1P1_B"$1"_greasy_PCA_RNN_logfile_JID"${SLURM_JOBID}".txt"
echo "GREASY - Running greasy tasks from "
echo $PWD
echo $GREASY_LOGFILE
echo "## BATCH = B"$1" ##"



greasy ./Tasks/MA1P1_B$1_greasy_PCA_RNN_tasks.txt



#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=debug

#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --partition=normal

#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --partition=normal


#SBATCH --time=24:00:00
#SBATCH --nodes=112
#SBATCH --partition=normal


