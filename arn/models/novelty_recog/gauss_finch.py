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
    """A GaussianRecognizer with FINCH for recognize_fit.

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

    # TODO handle recog_label_enc mapping to the gmm.label_enc!
    #   When and where is recog_label_enc state changed?


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
        if not self._gaussians:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')

        # Must update experience everytime and handle prior unknowns if any
        unks = ['unknown']
        if self.recog_label_enc:
            unks += list(self.recog_label_enc)
        unlabeled = self.experience[~self.experience['oracle']]
        unknowns = unlabeled['labels'].isin(unks)
        if unknowns.any():
            self.experience.loc[unknowns.index, 'labels'] = \
                self.known_label_enc.unknown_key

        self.label_enc = deepcopy(self.known_label_enc)

        # Update the unknown classes GMM
        if self.gmm is None:
            counter = 0
        else:
            counter = self.gmm.counter

        gmm = recognize_fit(
            'unknown',
            features,
            counter,
            allowed_error=self.min_error_tol,
            level=self.level,
            cov_epsilon=self.cov_epsilon,
            device=self.device,
            **kwargs,
        )


        # rm old unknown classes, replacing with current ones as this is
        # always called to redo the unknown class-clusters on ALL currently
        # unlabeled data deemed unknown.
        #if self.recog_label_enc is None:
        # TODO Update the knowns GMM.
        self._gaussians = self._gaussians[:self.n_known_labels - 1]
        self._thresholds = self._thresholds[:self.n_known_labels - 1]


        # Update label_enc to include the recog_label_enc at the end.
        if self.recog_label_enc:
            self.label_enc.append(self.recog_label_enc)
