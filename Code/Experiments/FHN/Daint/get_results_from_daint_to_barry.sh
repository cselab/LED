#!/bin/bash -l

SYSTEM_NAME=FHN


EXPERIMENT_NAME=Experiment_Daint_Large

echo '###############          COPY MODEL DATA            ###############';

EXPERIMENT_NAME=Experiment_Daint_Large
SYSTEM_NAME=FHN-100-Vort
mkdir -p /scratch/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/ /scratch/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}

EXPERIMENT_NAME=Experiment_Daint_Large
SYSTEM_NAME=FHN-1000-Vort
mkdir -p /scratch/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/STF/Code/Results/${SYSTEM_NAME}/ /scratch/pvlachas/STF/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}