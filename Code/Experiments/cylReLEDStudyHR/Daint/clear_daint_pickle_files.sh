#!/bin/bash -l


################################################
### RUN
################################################

ssh daint << 'EOF'
    cd /scratch/snx3000/pvlachas/STF/Code/Results/

    rm -rf ./*/*/*/autoencoder_testing_test_ic*_t*_target.pickle

    find . -name "autoencoder_testing_test_ic*_t*_*.pickle" | xargs rm
    find . -name "dimred_testing_test_ic*_t*_*.pickle" | xargs rm
    
    exit
EOF



# find . -name "autoencoder_testing_test_ic*_t*_*.pickle" | xargs rm
# find . -name "dimred_testing_*_ic*_prediction.pickle" | xargs rm
# find . -name "dimred_testing_*_ic*_prediction.pickle" | xargs rm








