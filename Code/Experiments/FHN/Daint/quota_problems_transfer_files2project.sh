#!/bin/bash -l

SYSTEM_NAME=FHN-100-Vort
PATH_SOURCE=/scratch/snx3000/pvlachas
PATH_DEST=/project/s929/pvlachas
PATH_BASE=/LED/Code/Results/${SYSTEM_NAME}/Evaluation_Data
mkdir -p ${PATH_DEST}${PATH_BASE}
mv ${PATH_SOURCE}${PATH_BASE}/* ${PATH_DEST}${PATH_BASE}/




