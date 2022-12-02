#!/bin/bash

CODE_PATH="/home/jin/research/jair_code"
DATA_PATH="/media/har"

apptainer exec \
    --bind $CODE_PATH:/home/jin/har/code \
    --bind $DATA_PATH:/home/jin/har/data \
    --bind /media/mnt/scratch_3/:/home/jin/scratch_31 \
    --pwd /home/jin/har/code \
    $CODE_PATH/arn/arn/data/arn_latest.sif \
    bash -c "ls -lh ./ &&
        /usr/bin/which python &&
        ls /arn &&
        pip install -e ./arn/ &&
        docstr arn/arn/scripts/exp2/configs/visual_transforms/tsf_gmmfinch_feedback_0.0.yaml \
            --feedback_amount 0.0 \
            --predictor.min_error_tol 0.05
    "

docstr arn/arn/scripts/exp2/configs/visual_transforms/tsf_gmmfinch_feedback_0.0.yaml \
        --feedback_amount 0.0 \
        --predictor.min_error_tol 0.05

docstr arn/scripts/exp2/configs/visual_transforms/tsf_gmmfinch_feedback_1.0_local.yaml \
        --feedback_amount 0.0 \
        --predictor.min_error_tol 0.05

%run /home/jin/anaconda3/envs/jair_updated/bin/docstr arn/scripts/exp2/configs/visual_transforms/tsf_gmmfinch_feedback_1.0_local.yaml \
        --feedback_amount 0.0 \
        --predictor.min_error_tol 0.05
