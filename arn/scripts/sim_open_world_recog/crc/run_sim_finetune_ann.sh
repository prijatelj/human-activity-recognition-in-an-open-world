#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -l h=!qa-a10-001
#$ -l h=!qa-a10-002
#$ -N tsf_ann_fb_1.0

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
    --bind $CODE_PATH:/afs/crc.nd.edu/user/j/jhuang24/har/code \
    --bind $DATA_PATH:/afs/crc.nd.edu/user/j/jhuang24/har/data \
    --bind /afs/crc.nd.edu/group/cvrl/scratch_31:/afs/crc.nd.edu/user/j/jhuang24/scratch_31 \
    --pwd /afs/crc.nd.edu/user/j/jhuang24/har/code \
    $CODE_PATH/arn/arn/data/arn_latest.sif \
    bash -c "ls -lh ./ &&
        /usr/bin/which python &&
        ls /arn
        pip install -e ./arn/ &&
        pip install exputils==0.1.7&&
        docstr arn/arn/scripts/exp2/configs/visual_transforms/tsf_ann_feedback_1.0.yaml \
            --feedback_amount 1.0 \
    "
