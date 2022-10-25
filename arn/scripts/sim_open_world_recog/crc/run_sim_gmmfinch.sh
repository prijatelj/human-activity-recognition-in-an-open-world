#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -N test_sim_kowl

# CRC script to run the simulated KOWL experiment.
module load conda
conda init bash
source activate base

# Assign the code and data paths appropriately and bind them to their
# respective expected paths within the used config
CODE_PATH="/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on"
DATA_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/har"

apptainer exec \
    --nv \
    --env CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu_card \
    --bind $CODE_PATH:/tmp/har/code \
    --bind $DATA_PATH:/tmp/har/data \
    --pwd /tmp/har/code \
    $CODE_PATH/arn/arn/data/arn_latest.sif \
    bash -c "ls -lh ./ &&
        /usr/bin/which python &&
        ls /arn
        pip install --no-deps -e ./arn/ &&
        pip install --no-deps -e ./exputils/ &&
        docstr arn/arn/scripts/exp2/configs/tsf_gmmfinch_blur.yaml \
            --feedback_amount 1.0 \
            --predictor.min_error_tol 0.05
    "