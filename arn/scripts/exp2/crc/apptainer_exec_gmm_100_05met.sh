pip install --no-deps -e ./dsp_arn/ &&
pip install --no-deps -e ./exputils/ &&
docstr dsp_arn/arn/scripts/exp2/configs/container_kowl_tsf_gmmfinch.yaml \
    --feedback_amount 1.0 \
    --predictor.min_error_tol 0.05
