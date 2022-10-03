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

    def fit_known(self, features):
        raise NotImplementedError

    def fit(self, features):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def save(self, h5, overwrite=False):
        raise NotImplementedError

    @staticmethod
    def load(h5):
        raise NotImplementedError
