


CODE_PATH="/home/jin/research/jair_code"
DATA_PATH="/media/har"

apptainer exec -B /afs --no-home \
    --nv \
    --env CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu_card \
    --bind $CODE_PATH:/home/jin/har/code \
    --bind $DATA_PATH:/home/jin/har/data \
    --bind /media/mnt/scratch_3/:/home/jin/scratch_31 \
    --pwd /home/jin/har/code \
    $CODE_PATH/arn/arn/data/arn_latest.sif \
    bash -c "ls -lh ./ &&
        /usr/bin/which python &&
        ls /arn
        pip install -e ./arn/ &&
        pip install exputils==0.1.7 &&
        docstr arn/arn/scripts/exp2/configs/visual_transforms/tsf_gmmfinch.yaml \
            --feedback_amount 1.0 \
            --predictor.min_error_tol 0.05
    "