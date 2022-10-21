#!/bin/bash

# TODO The CRC things for qsub

# CRC script to run the simulated KOWL experiment.

# Assign the code and data paths appropriately and bind them to their
# respective expected paths within the used config
CODE_PATH="/scratch365/dprijate/osr/har"
DATA_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/har"

apptainer exec \
    --nv \
    --env CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu_card \
    --bind $CODE_PATH:/tmp/har/code \
    --bind $DATA_PATH:/tmp/har/data \
    --pwd /tmp/har/code \
    $CODE_PATH/data/containers/arn_latest.sif \
    docstr arn/arn/scripts/sim_open_world_recog/configs/container_test-run_exp2_2d_sim.yaml \
        --predictor.load_inc_paths "$DATA_PATH/"
