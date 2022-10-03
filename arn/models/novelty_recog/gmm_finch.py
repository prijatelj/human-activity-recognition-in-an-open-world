"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""
import h5py
import numpy as np
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.models.novelty_recog.gmm import (
    GMM,
    GMMRecognizer,
    join_gmms,
    recognize_fit,
)

import logging
logger = logging.getLogger(__name__)


class GMMFINCH(GMMRecognizer):
    """Gaussian Mixture Model per class using FINCH to find the components.

    Attributes
    ----------
    gmms : list(GMM)
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns and unknown.
    label_enc : NominalDataEncoder
        The label encoder corresponding to the classes with GMMs. This follows
        the order of unknown (index 0), then knowns.
    """
    def __init__(self, *kwargs):
        raise NotImplementedError

        # TODO self.known_label_enc...
        # TODO self._label_enc...

    def add_new_knowns(self, new_knowns):
        raise NotImplementedError

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        raise NotImplementedError
        super().reset_recogs()
        # TODO Reset general recognizer to use just knowns

    def fit_knowns(self, features):
        raise NotImplementedError

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        raise NotImplementedError

        # NOTE Given multiple GMMs, find the thresholds per GMM (i don't think
        # this applies, just use detect() of the GMM.)

        # NOTE return the list of GMMs, each w/ thier own label_enc, gaussians,
        # and thresholds and methods for finding recog, detect w/in, and
        # returning the log prob of samples belonging to the entire GMM

    def recognize(self, features, detect=False):
        raise NotImplementedError

    def detect(self, features, known_only=True):
        raise NotImplementedError

    def save(self, h5, overwrite=False):
        raise NotImplementedError
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        # Save the attrs unique to this object
        h5.attrs['level'] = self.level

        # Save known_gmm, unknown_gmm, but NOT gmm, as it is joined by the 2.
        if self.known_gmm:
            self.known_gmm.save(h5.create_group('known_gmm'))

        super().save(h5)
        if close:
            h5.close()


    @staticmethod
    def load(h5):
        raise NotImplementedError
