#!/bin/bash -l

SYSTEM_NAME=KSGP64L22
PATH_SOURCE=/scratch/snx3000/pvlachas
PATH_DEST=/project/s929/pvlachas
PATH_BASE=/STF/Code/Results/${SYSTEM_NAME}/Evaluation_Data
mkdir -p ${PATH_DEST}${PATH_BASE}
mv ${PATH_SOURCE}${PATH_BASE}/* ${PATH_DEST}${PATH_BASE}/




