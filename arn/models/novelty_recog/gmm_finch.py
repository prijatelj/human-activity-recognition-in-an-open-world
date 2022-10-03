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
    known_gmms : list(GMM) = None
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns.
    see GMMRecognizer.__init__
    """
    def __init__(self, *args, *kwargs):
        """
        Args
        ----
        see GMMRecognizer.__init__
        """
        # NOTE uses self.known_label_enc and self._label_enc
        super().__init__(*args, **kwargs)
        self.known_gmms = None

    def add_new_knowns(self, new_knowns):
        raise NotImplementedError('Anything different from parent?')

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        super().reset_recogs()

        # Reset general recognizer to use just knowns
        self._label_enc = deepcopy(self.known_label_enc)

    def fit_knowns(self, features, labels, val_dataset=None):
        raise NotImplementedError

        # TODO for each known class with labels, fit the GMM.

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        if not self.known_gmms:
            raise ValueError('Recognizer is not fit: self.known_gmms is None.')

        super().recognize_fit(features, n_expected_classes, **kwargs)

        # Update the general knowns + unknown recogs expanded
        self._label_enc = deepcopy(self.known_label_enc)

    def recognize(self, features, detect=False):
        # Loop through all known gmms + unknown_recogs getting log_probs.
        recogs = [gmm.log_prob(features) for gmm in self.known_gmms]

        if self.has_recogs:
            unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
            recogs = torch.cat([recogs, uknown_log_probs], dim=1)

        if detect:
            detect_unknowns = (recogs < self.thresholds).all(1)

            recogs = F.pad(F.softmax(recogs, dim=1), (1, 0), 'constant', 0)

            # Sets unknown to max prob value, scales the rest by 1 - max
            if detect_unknowns.any():
                recogs[detect_unknowns, 0] = \
                    recogs[detect_unknowns].max(1).values
                recogs[detect_unknowns, 1:] *= 1 \
                    - recogs[detect_unknowns, 0].reshape(-1, 1)
            return recogs
        return F.softmax(known_log_probs, dim=1)

    def detect(self, features, known_only=True):
        recogs = [gmm.log_prob(features) for gmm in self.known_gmms]

        if self.has_recogs:
            unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
            recogs = torch.cat([recogs, uknown_log_probs], dim=1)

        return (recogs < self.thresholds).all(1)

    def save(self, h5, overwrite=False):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        # Save known_gmms, but NOT gmm, as it is joined by the 2.
        knowns = h5.create_group('known_gmms')
        for gmm in self.known_gmms:
            gmm.save(knowns.create_group(gmm.unknown_key))

        super().save(h5)
        if close:
            h5.close()

    @staticmethod
    def load(h5):
        raise NotImplementedError
