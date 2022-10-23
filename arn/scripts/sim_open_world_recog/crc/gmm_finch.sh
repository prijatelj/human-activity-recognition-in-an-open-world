#!/bin/bash

#$ -pe smp 8
#$ -N sim_gmm
#$ -q long
#$ -o $HOME/scratch365/osr/har/logs/sims/gmmf/
#$ -e $HOME/scratch365/osr/har/logs/sims/gmmf/

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
    dsp_arn/arn/scripts/sim_open_world_recog/crc/apptainer_exec_gmm.sh
