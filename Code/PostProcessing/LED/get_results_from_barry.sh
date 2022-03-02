#!/bin/bash -l



SYSTEM_NAME=FHN

mkdir -p /Users/pvlachas/STF/Code/PostProcessing/LED/Barry/${SYSTEM_NAME}/
rsync -mzarvP barry:/scratch/pvlachas/STF/Code/PostProcessing/LED/${SYSTEM_NAME}/ /Users/pvlachas/STF/Code/PostProcessing/LED/Barry/${SYSTEM_NAME}


