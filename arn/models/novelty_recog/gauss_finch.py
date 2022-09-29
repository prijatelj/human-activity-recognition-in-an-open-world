"""Naive DPGMM version of GaussianRecognizer."""
from copy import deepcopy

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder
from vast.clusteringAlgos.FINCH.python.finch import FINCH

from arn.models.novelty_recog.gaussian import (
    GaussianRecognizer,
    cred_hyperellipse_thresh,
)
from arn.models.novelty_recog.gmm_finch import recognize_fit

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

        # TODO known gmm, gaussian per class
        self.known_gmm = None

        # TODO unknown gmm, as in recognize_fit.
        self.unknown_gmm = None

        # The combined gmms into one for predict()/recigonize(), detect()
        self.gmm = None

        # TODO recognize and predict using

    @property
    def label_enc(self):
        if self.gmm is not None:
            return self.gmm.label_enc

    def fit_knowns(self, features, val_dataset=None):
        # TODO update known label_enc using GMM.fit() or gmm_fit()
        if self.known_gmm is None:
            self.known_gmm = GMM(
                min_samples=0,
            )
        self.known_gmm.fit(features, known_labels)

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

        # Must update experience everytime and handle prior unknowns if any
        unks = ['unknown']
        if self.recog_label_enc:
            unks += list(self.recog_label_enc)
        unlabeled = self.experience[~self.experience['oracle']]
        unknowns = unlabeled['labels'].isin(unks)
        if unknowns.any():
            self.experience.loc[unknowns.index, 'labels'] = \
                self.known_label_enc.unknown_key

        # Update the unknown classes GMM
        if self.unknown_gmm is None:
            counter = 0
        else:
            counter = self.unknown_gmm.counter

        self.unknown_gmm = recognize_fit(
            'unknown',
            features,
            counter,
            allowed_error=self.min_error_tol,
            level=self.level,
            cov_epsilon=self.cov_epsilon,
            device=self.device,
            **kwargs,
        )

        self.gmm = join_gmms(self.known_gmm, self.unknown_gmm)
