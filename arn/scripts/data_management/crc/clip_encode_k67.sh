#!/bin/bash

#$ -pe smp 4
#$ -N k67clipEncCVRL
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/k67/
#$ -e $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/k67/
#$ -t 1-6

# Check if given an arg
if [[ "$#" -eq 1 ]]; then
    SGE_TASK_ID="$1"
    echo "Given an argument and thus set SGE_TASK_ID to $SGE_TASK_ID"
fi

echo "SGE_TASK_ID = $SGE_TASK_ID"

# Set up environment
module add conda
conda activate arn

# Profile the CRC Job on the node in which it is running.
crc_profile

# Path variables
BASE="/scratch365/dprijate/osr/har"

# Add to path variables for this given run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    SPLIT="train"
    DSET_NUM="600"
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    SPLIT="validate"
    DSET_NUM="600"
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    SPLIT="test"
    DSET_NUM="600"
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    SPLIT="train"
    DSET_NUM="700_2020"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    SPLIT="validate"
    DSET_NUM="700_2020"
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    SPLIT="test"
    DSET_NUM="700_2020"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

AUG=""
DATA="$BASE/data/kinetics/kinetics$DSET_NUM"
VIDS_BASE="$DATA/videos"
VIDS="$VIDS_BASE/$SPLIT"

python3 "$BASE/arn/arn/scripts/research/clip/kinetics_clip_img_encode.py" \
    "$VIDS/" \
    "$DATA/$SPLIT.json" \
    "$DATA/unique_labels.txt" \
    "$BASE/models/clip/clip_ViT-B_32.pt" \
    --label_path "$DATA/clip_encoded_k$DSET_NUM"_classes.pt \
    --image_path "$DATA/clip/$SPLIT/$AUG/images_enc_clip_ViT-B_32.pt" \
    --pred_path "$DATA/clip/$SPLIT/$AUG/preds_K$DSET_NUM"_clip_ViT-B_32_preds.pt
