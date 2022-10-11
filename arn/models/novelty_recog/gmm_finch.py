"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""
from copy import deepcopy

import h5py
import numpy as np
import torch
F = torch.nn.functional

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.models.novelty_recog.gaussian import min_max_threshold
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
    known_gmms : GMM = None
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns.
    threshold : float | torch.Tensor = None
        Global threshold for all distribs involved. If None, use the internal
        distribs thresholding for detection.
    likelihood_unknown : float = 0.0
        The likelihood used to specify how likely a sample is unknown to the
        the minimum maximum log prob sample. We recommend zero or negative
        values as it is added to the log_prob, and subtraction is then saying
        it is less likely, e.g., likelihood of -2 means the (currnetly static)
        prior belief is that the unknown samples will be half as likely as the
        least likely known class any sample was assigned to.
    see GMMRecognizer.__init__
    """
    def __init__(self, likelihood_unknown=0.0, *args, **kwargs):
        """
        Args
        ----
        likelihood_unknown : see self
        see GMMRecognizer.__init__
        """
        # NOTE uses self.known_label_enc and self._label_enc
        super().__init__(*args, **kwargs)
        self.known_gmms = None
        self.thresholds = None
        self.likelihood_unknown = likelihood_unknown

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        super().reset_recogs()

        # Reset general recognizer to use just knowns
        self._label_enc = deepcopy(self.known_label_enc)

        logger.info('%s.reset_recogs() finished', type(self))

    def fit_knowns(self, features, labels, val_dataset=None):
        # For each known class with labels, fit the GMM.
        knowns = iter(self.known_label_enc.items())
        next(knowns)

        if self.known_gmms is None:
            n_old_knowns = 0
            self.known_gmms = []
        else:
            n_old_knowns = len(self.known_gmms)

        for known, idx in knowns:
            class_mask = labels == idx
            if not class_mask.any():
                raise ValueError(
                    'Every known class must have at least one feature sample.'
                )
            prior_idx = idx - 1
            if prior_idx < n_old_knowns:
                self.known_gmms[prior_idx] = recognize_fit(
                    known,
                    features[class_mask],
                    self.known_gmms[prior_idx].counter,
                    self.threshold_func,
                    self.min_error_tol,
                    level=self.level,
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                self.known_gmms.append(recognize_fit(
                    known,
                    features[class_mask],
                    0,
                    self.threshold_func,
                    self.min_error_tol,
                    level=self.level,
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                ))

        logger.info(
            "%s's %s.fit_knowns() finished fitting the GMMs per class",
            self.uid,
            type(self).__name__,
        )

        # If threshold_func is min_max_threshold, then it is global to the
        # known gmms and thus needs set.
        if self.threshold_func == 'min_max_threshold':
            self.thresholds = min_max_threshold(
                self.known_gmms,
                features,
                self.likelihood_unknown,
            )

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        if not self.known_gmms:
            raise ValueError('Recognizer is not fit: self.known_gmms is None.')

        super().recognize_fit(features, n_expected_classes, **kwargs)

        # Update the general knowns + unknown recogs expanded
        self.update_label_enc(False)

    def recognize(self, features, detect=False):
        # Loop through all known gmms + unknown_recogs getting log_probs.
        if self.thresholds is not None:
            recogs = torch.stack(
                [gmm.log_prob(features) for gmm in self.known_gmms],
                dim=1
            )

            if self.has_recogs:
                unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
                recogs = torch.cat([recogs, unknown_log_probs], dim=1)

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
            return F.softmax(recogs, dim=1)

        raise NotImplementedError('Only global thresholding implemented atm.')

        # TODO Must use gmms' thresholds instead.
        recogs = [gmm.detect(features) for gmm in self.known_gmms]

        if self.has_recogs:
            unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
            recogs.append(unknown_log_probs)
        recogs = (torch.stack(recogs, dim=1) < self.thresholds).all(1)

    def detect(self, features, known_only=True):
        if self.thresholds is not None:
            recogs = torch.stack(
                [gmm.log_prob(features) for gmm in self.known_gmms],
                dim=1
            )

            if self.has_recogs and not known_only:
                unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
                recogs = torch.cat([recogs, unknown_log_probs], dim=1)

            return (recogs < self.thresholds).all(1)

        raise NotImplementedError('Only global thresholding implemented atm.')
        # TODO Use the gmms' thresholds

    def log_prob(self, features):
        return torch.stack(
            [gmm.log_prob(features) for gmm in self.known_gmms],
            dim=1,
        ).sum(1)

    def save(self, h5, overwrite=False):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        # Save known_gmms, but NOT gmm, as it is joined by the 2.
        knowns = h5.create_group('known_gmms')
        for gmm in self.known_gmms:
            gmm.save(knowns.create_group(gmm.label_enc.unknown_key))

        super().save(h5)
        if close:
            h5.close()

    @staticmethod
    def load(h5):
        raise NotImplementedError
