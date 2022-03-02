#!/bin/bash -l
#SBATCH --account=eth2
#SBATCH --time=24:00:00
#SBATCH --nodes=24
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=mc
#SBATCH --job-name=KSGP64L22_P6_greasy_RC
#SBATCH --output=/scratch/snx3000/pvlachas/STF/Code/Results/KSGP64L22/Logs/KSGP64L22_P6_greasy_RC_outputfile_JID%j_A%a.txt
#SBATCH --error=/scratch/snx3000/pvlachas/STF/Code/Results/KSGP64L22/Logs/KSGP64L22_P6_greasy_RC_errorfile_JID%j_A%a.txt

# ======START=====

module load daint-mc
module load GREASY

# FOR PYTORCH
module load cray-python/3.8.2.1
source ${HOME}/venv-python3.8-pytorch1.9/bin/activate

export GREASY_LOGFILE="/scratch/snx3000/pvlachas/STF/Code/Results/KSGP64L22/Logs/KSGP64L22_P6_B"$1"_greasy_RC_logfile_JID"${SLURM_JOBID}".txt"

echo "GREASY - Running greasy tasks from "
echo $PWD
echo $GREASY_LOGFILE
echo "## BATCH = B"$1" ##"
# echo "## PRINTING ALL ENV VARIABLES ##"
# env
# echo "## VARIABLES PRINTED ##"

greasy ./Tasks/P6_B$1_greasy_RC_tasks.txt
# greasy ./Tasks/P6T_B$1_greasy_RC_tasks.txt

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=normal

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=debug

#SBATCH --time=24:00:00
#SBATCH --nodes=24
#SBATCH --partition=normal




