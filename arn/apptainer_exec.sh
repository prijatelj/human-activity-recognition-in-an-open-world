pip install --no-deps -e ./arn/ &&
pip install --no-deps -e ./exputils/ &&
docstr arn/arn/scripts/exp2/configs/tfs_fine_tune_blur.yaml \
    --feedback_amount 0.0 \
    --predictor.min_error_tol 0.05
