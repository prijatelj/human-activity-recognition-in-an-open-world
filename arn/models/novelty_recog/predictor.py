"""The Open World Predictor with Novelty Recognition."""
from arn.models.owhar import OWHAPredictor


class GaussRecogPredictor(OWHAPredictor):
    """Open world predictor with novelty recognition through GaussRecog"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO change the uid prefix from 'owhar_' to 'gauss-recog-predictor_'

        # TODO handle docstrs appropriatley while reusing what you can from
        # parent.

        # TODO need to incorporate the recognition into the step process of:
        #   predict, fit, predict post fit

        # TODO need to isolate this predictor's recognized class labels from
        # those told to it by an oracle, if any.
