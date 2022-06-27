#!/bin/bash -l

SYSTEM_NAME=KSGP64L22


EXPERIMENT_NAME=Experiment_Daint_Large

echo '###############          COPY MODEL DATA            ###############';

EXPERIMENT_NAME=Experiment_Daint_Large
SYSTEM_NAME=KSGP64L22
mkdir -p /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/ /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}

EXPERIMENT_NAME=Experiment_Daint_Large
SYSTEM_NAME=KSGP64L22
mkdir -p /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}
rsync -mzarvP daint:/scratch/snx3000/pvlachas/LED/Code/Results/${SYSTEM_NAME}/ /scratch/pvlachas/LED/Code/Results/${EXPERIMENT_NAME}/${SYSTEM_NAME}