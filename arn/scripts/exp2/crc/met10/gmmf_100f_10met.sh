#!/bin/bash

#$ -pe smp 24
#$ -N gmm_1_10
#$ -q long
#$ -o $HOME/scratch_365/osr/har/logs/gmmf/met10/
#$ -e $HOME/scratch_365/osr/har/logs/gmmf/met10/

# CRC script to run the simulated KOWL experiment.
#module load conda
#conda init bash
#source activate base

# Assign the code and data paths appropriately and bind them to their
# respective expected paths within the used config
CODE_PATH="/scratch365/dprijate/osr/har"
DATA_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/har"

#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044

apptainer exec \
    --nv \
    --env CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu_card \
    --bind $CODE_PATH:/tmp/har/code \
    --bind $DATA_PATH:/tmp/har/data \
    --pwd /tmp/har/code \
    $CODE_PATH/data/containers/arn_latest.sif \
    dsp_arn/arn/scripts/exp2/crc/met10/apptainer_exec_gmm_100_10met.sh
