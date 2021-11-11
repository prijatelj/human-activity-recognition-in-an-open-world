"""The evaluation code of an OWHAR predictor given an OWHAR data at the
current increment.
"""

import exputils

class Evaluator(object):
    """The Evaluator consists measures and stores the results over time for
    every increment.

    Attributes
    ----------
    conf_mats :
        If ConfusionMatrix objects are being saved, then they go here. This is
        the confusion matrix for each increment(?). For every increment, the
        confusion matrix is important to save in order to obtain all possible
        measurements.

        TODO Top-K ConfusionMatrix objectst
    results :
    """
    def __init__(self):
        raise NotImplementedError()

    def measure_recog(self, labels, predictions):
        return
