#!/bin/bash

#$ -pe smp 4
#$ -N k4clipEnc
#$ -q gpu
#$ -o $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/
#$ -e $HOME/scratch_48/har/kinetics/clip_encode_pred/logs/
#$ -t 1-19

# Set up environment
module add conda
conda activate arn

# Path variables
BASE="/scratch365/dprijate/osr/har"
DATA="$BASE/data/kinetics/kinetics400/"
VIDS_BASE="/scratch365/sgrieggs"

# Add to path variables for this given run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    SPLIT="train"
    AUG=""
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT"
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    SPLIT="validate"
    AUG=""
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT"
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    SPLIT="test"
    AUG=""
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT"

# Augmentations: Validation
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    SPLIT="validate"
    AUG="normal"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    SPLIT="validate"
    AUG="blur"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    SPLIT="validate"
    AUG="flip"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "7" ]; then
    SPLIT="validate"
    AUG="invert"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "8" ]; then
    SPLIT="validate"
    AUG="noise"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "9" ]; then
    SPLIT="validate"
    AUG="perspective"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "10" ]; then
    SPLIT="validate"
    AUG="rotation"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "11" ]; then
    SPLIT="validate"
    AUG="jitter"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"

# Augmentations: Test
elif [ "$SGE_TASK_ID" -eq "12" ]; then
    SPLIT="test"
    AUG="normal"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "13" ]; then
    SPLIT="test"
    AUG="blur"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "14" ]; then
    SPLIT="test"
    AUG="flip"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "15" ]; then
    SPLIT="test"
    AUG="invert"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "16" ]; then
    SPLIT="test"
    AUG="noise"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "17" ]; then
    SPLIT="test"
    AUG="perspective"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "18" ]; then
    SPLIT="test"
    AUG="rotation"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
elif [ "$SGE_TASK_ID" -eq "19" ]; then
    SPLIT="test"
    AUG="jitter"
    VIDS="VIDS_BASE/kinetics-dataset-400-$SPLIT-$AUG"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

mkdir -p "$BASE/$SPLIT/"

cd "$BASE/tar_$SPLIT/"

python3 "$BASE/arn/arn/scripts/research/clip/kinetics_clip_img_encode.py" \
    "$VIDS" \
    "$DATA/$SPLIT.json" \
    "$DATA/unique_labels.txt" \
    "$BASE/models/clip/clip_ViT-B_32.pt" \
    --label_path "$DATA/clip_encoded_k400_classes.pt" \
    --image_path "$DATA/clip/$SPLIT/$AUG/images_enc_clip_ViT-B_32.pt" \
    --pred_path "$DATA/clip/$SPLIT/$AUG/preds_K400_clip_ViT-B_32_preds.pt"
