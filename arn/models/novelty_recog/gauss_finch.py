"""Recognizer with a Gaussian per known class, GMM for unknowns using FINCH to
find unknown class clustes.
"""
from copy import deepcopy

import numpy as np
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder
from vast.clusteringAlgos.FINCH.python.finch import FINCH

from arn.models.novelty_recog.gaussian import (
    GaussianRecognizer,
    cred_hyperellipse_thresh,
    closest_other_marignal_thresholds,
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

    def fit_knowns(self, features, labels, val_dataset=None):
        # TODO update known label_enc using GMM.fit() or gmm_fit()
        if self.known_gmm is None:
            self.known_gmm = GMM(
                NominalDataEncoder(np.unique(dataset.labels)), # TODO ???
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

    def recognize(self, features, detect=False):
        """Using the existing Gaussians per class-cluster, get log probs."""
        # Normalize the probability each feature belongs to a recognized class,
        # st the recognized classes are mutually exclusive to one another.
        recogs = torch.stack(
            [mvn.log_prob(features) for mvn in self._gaussians],
            dim=1,
        )
        if detect:
            thresholds = torch.Tensor(self._thresholds)
            detect_unknowns = (recogs < thresholds).all(1)

            recogs = F.pad(F.softmax(recogs, dim=1), (1, 0), 'constant', 0)

            # Sets unknown to max prob value, scales the rest by 1 - max
            if detect_unknowns.any():
                recogs[detect_unknowns, 0] = \
                    recogs[detect_unknowns].max(1).values
                recogs[detect_unknowns, 1:] *= 1 \
                    - recogs[detect_unknowns, 0].reshape(-1, 1)
            return recogs

        return F.softmax(recogs, dim=1)

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
        """
        if not self._gaussians:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')

        # NOTE IDEAL elsewhere: Detect is an inference time only task, and
        # inherently binary classificicaiton of known vs unknown.
        #   0. new data is already assumed to have been used to fit ANN / recog
        #   1. call `class_log_probs = recognize(features)`
        #   2. reduce unknowns.
        #   3. return sum of probs of unknowns, that is detection given recog.
        # NOTE in this method, there is disjoint between ANN and recog.
        #   with this approach, perhaps should compare just DPGMM on frepr to
        #   DPGMM on ANN of frepr.
        # Given never letting go of all experienced data, the DPGMM on ANN
        #   has a better chance of success over the increments than on just the
        #   frepr as the frepr currently is frozen over the increments.

        num_labels = self.n_known_labels if knowns_only else self.n_labels
        thresholds = torch.Tensor(self._thresholds[:num_labels])
        return (torch.stack(
            [mvn.log_prob(features) for mvn in self._gaussians[:num_labels]],
            dim=1,
        ) < thresholds).all(1)
