#!/bin/bash -l



SYSTEM_NAME=FHN

mkdir -p /Users/pvlachas/LED/Code/PostProcessing/LED/Barry/${SYSTEM_NAME}/
rsync -mzarvP barry:/scratch/pvlachas/LED/Code/PostProcessing/LED/${SYSTEM_NAME}/ /Users/pvlachas/LED/Code/PostProcessing/LED/Barry/${SYSTEM_NAME}


