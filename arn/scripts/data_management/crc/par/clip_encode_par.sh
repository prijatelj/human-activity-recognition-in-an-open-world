#!/bin/bash

#$ -pe smp 4
#$ -N clipEncPAR
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/par/
#$ -e $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/par/
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
DATA="$BASE/data/par"
VIDS_BASE="/afs/crc.nd.edu/group/cvrl/scratch_34/sgrieggs"

DSET_NUM="par"
AUG=""

# Add to path variables for this given run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    # Simply encode all of training together.
    SPLIT="train"
    VIDS="$VIDS_BASE/all_vids_and_augs"
    DATA="$DATA/5fold_training_m24/train-0_tr-val/5folds_$SPLIT-0.csv"

# Additional videos: Encode in their batches as they are.
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    SPLIT="train"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    SPLIT="val"
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    SPLIT="test"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

python3 "$BASE/arn/arn/scripts/research/clip/kinetics_clip_img_encode.py" \
    "$VIDS/" \
    "$DATA" \
    "$BASE/models/clip/clip_ViT-B_32.pt" \
    --label_path "$DATA/clip/clip_encoded_$DSET_NUM"_classes.pt \
    --image_path "$DATA/clip/$SPLIT/$AUG/images_enc_clip_ViT-B_32.pt" \
    --pred_path "$DATA/clip/$SPLIT/$AUG/preds_$DSET_NUM"_clip_ViT-B_32_preds.pt
