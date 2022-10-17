#!/bin/bash

# TODO The CRC things

# CRC script to run the simulated KOWL experiment.

# Assign the code and data paths appropriately and bind them to their
# respective expected paths within the used config
CODE_PATH="/scratch365/dprijate/osr/har"
DATA_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/har"

FEEDBACK_AMOUNT="1.0"

apptainer exec \
    --nv \
    --bind $CODE_PATH:/tmp/har/code \
    --bind $DATA_PATH:/tmp/har/data \
    --pwd /tmp/har/code \
    $CODE_PATH/data/containers/arn_latest.sif \
    python \
        -c \
        "docstr arn/scripts/sim_open_world_recog/configs/exp2_gmm_finch.yaml --feedback_amount $FEEDBACK_AMOUNT"
