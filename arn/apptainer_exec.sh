pip install --no-deps -e ./arn/ &&
pip install --no-deps -e ./exputils/ &&
ls -lh ./ &&
docstr arn/arn/scripts/exp2/configs/container_kowl_tsf_gmmfinch_load_inc_paths.yaml \
    --feedback_amount 0.0 \
    --predictor.min_error_tol 0.05
