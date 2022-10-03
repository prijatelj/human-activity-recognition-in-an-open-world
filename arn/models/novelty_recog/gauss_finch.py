"""Recognizer with a Gaussian per known class, GMM for unknowns using FINCH to
find unknown class clustes.
"""
from copy import deepcopy

import numpy as np
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder

from arn.models.novelty_recog.gaussian import (
    GaussianRecognizer,
    cred_hyperellipse_thresh,
    closest_other_marignal_thresholds,
)
from arn.models.novelty_recog.gmm_finch import (
    GMM,
    join_gmms,
    recognize_fit,
)

import logging
logger = logging.getLogger(__name__)


class GaussFINCH(GaussianRecognizer):
    """A GaussianRecognizer with FINCH for recognize_fit. Gaussian per class,
    Gaussian Mixture Model for unknowns based on detection thresholds of known
    classes. Every recognize_fit() with new data the unknown GMM gets refit.

    Args
    ----
    level : int = -1
        The level of cluster partitions to use during recognition_fit. FINCH
        returns three levels of clustering. Defaults to the final level with
        maximum clusters.
    see GaussianRecognizer
    """
    def __init__(self, level=-1, *args, **kwargs):
        """
        Args
        ----
        level : see self
        see GaussianRecognizer.__init__
        """
        self.level = level
        super().__init__(*args, **kwargs)

        # known gmm, gaussian per class
        self.known_gmm = None

        # unknown gmm, as in recognize_fit.
        self.unknown_gmm = None

        # The combined gmms into one for predict()/recigonize(), detect()
        self.gmm = None

    @property
    def known_label_enc(self):
        if self.known_gmm is not None:
            return self.known_gmm.label_enc

    @property
    def recog_label_enc(self):
        if self.unknown_gmm is not None:
            return self.unknown_gmm.label_enc

    @property
    def label_enc(self):
        if self.gmm is not None:
            return self.gmm.label_enc

    def add_new_knowns(self, new_knowns):
        """Adds the given class labels as new knowns to the known label encoder

        Both known and unknown gmm label encoders have 'unknown' catchall!
        """
        if self.known_gmm is None:
            if isinstance(new_knowns, NominalDataEncoder):
                self.known_gmm = GMM(
                    new_knowns,
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                    threshold_func=self.threshold_func,
                    min_samples=0,
                    accepted_error=self.detect_error_tol,
                )
            else:
                self.known_gmm = GMM(
                    NominalDataEncoder(
                        new_knowns,
                        unknown_key='unknown',
                    ),
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                    threshold_func=self.threshold_func,
                    min_samples=0,
                    accepted_error=self.detect_error_tol,
                )
        else:
            self.known_gmm.label_enc.append(new_knowns)

    def fit_knowns(self, features, labels, val_dataset=None):
        self.known_gmm.fit(
            features,
            labels,
        )

    def recognize_fit(
        self,
        features,
        n_expected_classes=None,
        **kwargs,
    ):
        """Given unknown feature space points, find new class clusters using
        FINCH.

        Args
        ----
        features : np.ndarray | torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims)
            that are the features of points treated as outliers.
        **kwargs : dict
            Key word arguments for FINCH for detecting clusters.

        Side Effects
        ------------
        Any new unknown classes are added to the self.recog_label_enc, and
        self._gaussians
        """
        if not self.known_gmm:
            raise ValueError('Recognizer is not fit: self.known_gmm is None.')

        self.pre_recognize_fit()

        # Update the unknown classes GMM
        if self.unknown_gmm is None:
            counter = 0
        else:
            counter = self.unknown_gmm.counter

        self.unknown_gmm = recognize_fit(
            'unknown',
            features,
            counter,
            self.threshold_func,
            self.min_error_tol,
            level=self.level,
            cov_epsilon=self.cov_epsilon,
            device=self.device,
            **kwargs,
        )

        self.gmm = join_gmms(self.known_gmm, self.unknown_gmm)

    def recognize(self, features, detect=False):
        """Using the existing Gaussians per class-cluster, get log probs.
         Normalize the probability each feature belongs to a recognized class,
         st the recognized classes are mutually exclusive to one another.
        """
        return self.gmm.recognize(features, detect)

    def detect(self, features, knowns_only=True):
        """Given data samples, detect novel samples to the known classes.

        fit/recog: If given unlabeled data, decide if any of those samples form
        new clusters that consitute as a new class based on the initial
        hyperparameters.

        Recognized novel unknown classes are represented as 'unknown_#' where #
        is the newest class number.

        Criteria for a new class cluster:
        - min number of samples
        - which density param is acceptable?
            min density of any class XOR density level of nearest class.

        Args
        ----
        features : torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims).
        preds : torch.Tensor
            1 dimensional integer tensor of shape (samples,). Contains the
            index encoding of each label predicted per feature.
        n_expected_classes : int = None
            The number of expected classes, thus including the total known
            classes + unknown class + any recognized unknown classes.

        Returns
        -------
        torch.Tensor
            The predicted class label whose values are within label_enc.

        Notes
        -----
        NOTE IDEAL elsewhere: Detect is an inference time only task, and
        inherently binary classificicaiton of known vs unknown.
          0. new data is already assumed to have been used to fit ANN / recog
          1. call `class_log_probs = recognize(features)`
          2. reduce unknowns.
          3. return sum of probs of unknowns, that is detection given recog.
        NOTE in this method, there is disjoint between ANN and recog.
          with this approach, perhaps should compare just DPGMM on frepr to
          DPGMM on ANN of frepr.
        Given never letting go of all experienced data, the DPGMM on ANN
          has a better chance of success over the increments than on just the
          frepr as the frepr currently is frozen over the increments.
        """
        if not self.known_gmm:
            raise ValueError('Recognizer is not fit: self.known_gmm is None.')

        if knowns_only:
            return self.known_gmm.detect(features)
        return self.gmm.detect(features)

    def save(self, h5, overwrite=False):
        raise NotImplementedError
        # TODO save super().save(h5)
        # TODO Save the attrs unique to this object
        # TODO save known_gmm, unknown_gmm, NOT gmm, as it is joined by the 2.

    @staticmethod
    def load(h5):
        raise NotImplementedError
