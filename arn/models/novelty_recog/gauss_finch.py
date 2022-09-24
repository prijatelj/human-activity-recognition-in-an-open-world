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
    get_log_prob_mvn_thresh,
)

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

        if isinstance(features, torch.Tensor):
            dtype = features.dtype
            features = features.detach().cpu().numpy()
        else:
            dtype = getattr(torch, str(features.dtype))

        default_kwargs = {'verbose': False}
        default_kwargs.update(kwargs)

        logger.info("Begin fit of GaussianRecog's FINCH.")
        logger.debug(
            "Fitting %s with %d samples.",
            type(self).__name__,
            len(features),
        )
        recog_labels, n_clusters, _ = FINCH(features, **default_kwargs)

        # TODO level is always the last, most fine grained w/ max clusters,
        # we probably do not want this and would instead like to find the max
        # likely clusters on the data from ranges of [2, max].
        #   A way to do this, fit MVN to clusters in every partition and calc
        #   the Bayes factor or likelihood ratios to find best fit.
        # Using the above method could incorporate the known MVNs as well
        # such that the resulting clusters are then fit on all points in the
        # space, not just the detected outliers/novel points.

        recog_labels = recog_labels[:, self.level]
        n_clusters = n_clusters[self.level]

        # TODO Max the likelihood of GMM to select the number of FINCH parts
        # TODO generalize this recognize_fit to apply to any set of pts and be
        # func called that returns all the class' GMM state

        # Get all MVNs found to serve as new class-clusters, unless deemed
        # unconfident its a new class based on critera (min samples and
        # density)
        logger.debug(
            "%s found %d new unknown classes.",
            type(self).__name__,
            n_clusters,
        )
        if n_clusters <= 1 or n_clusters < self.min_samples:
            # No recognized unknown classes.
            return

        # TODO rm old unknown classes, replacing with current ones as this is
        # always called to redo the unknown class-clusters on ALL currently
        # unlabeled data deemed unknown.
        #if self.recog_label_enc is None:
        self._gaussians = self._gaussians[:self.n_known_labels - 1]
        self._thresholds = self._thresholds[:self.n_known_labels - 1]
        self.recog_label_enc = NominalDataEncoder()

        # Numerical stability adjustment for the sample covariance's diagonal
        stability_adjust = self.cov_epsilon * torch.eye(
            features.shape[-1],
            device=self.device
        )


        for i in range(n_clusters):
            cluster_mask = recog_labels == i
            logger.debug(
                "%s's %d-th potential class has %d samples.",
                type(self).__name__,
                i,
                sum(cluster_mask),
            )
            if self.min_samples:
                if sum(cluster_mask) < self.min_samples:
                    continue

            class_features = torch.tensor(
                features[cluster_mask],
                dtype=self.dtype,
                device=self.device,
            )
            loc = class_features.mean(0)
            cov_mat = class_features.T.cov()
            try:
                mvn = MultivariateNormal(loc, cov_mat)
            except:
                mvn = MultivariateNormal(loc, cov_mat + stability_adjust)

            logger.debug(
                "%s's %d-th potential class has "
                '%f log_prob.mean.',
                type(self).__name__,
                i,
                mvn.log_prob(class_features).mean(),
            )

            self._gaussians.append(mvn)
            self._thresholds.append(get_log_prob_mvn_thresh(
                mvn,
                self.detect_error_tol,
            ))

            # Update the label encoder with new recognized class-clusters
            self.recog_label_enc.append(f'unknown_{self._recog_counter}')
            self._recog_counter += 1

        # Save the normalized belief of the unknowns
        #self._recog_weights = dpgmm.weights_[argsorted_weights]
        logger.debug(
            "%s found %d new classes.",
            type(self).__name__,
            self.n_recog_labels
        )

        # Update label_enc to include the recog_label_enc at the end.
        self.label_enc = deepcopy(self.known_label_enc)
        self.label_enc.append(self.recog_label_enc)
