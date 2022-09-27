"""Naive DPGMM version of GaussianRecognizer."""
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder

from arn.models.novelty_recog.gaussian import (
    GaussianRecognizer,
    cred_hyperellipse_thresh,
)

import logging
logger = logging.getLogger(__name__)


class NaiveDPGMM(GaussianRecognizer):
    """Naive Dirichlet Process Gaussian Mixture Model that assumes independence
    of detected novel points in the event space to points in the same space
    belonging to known classes. This results in not conditioning the DPGMM on
    known class points.

    Args
    ----
    see GaussianRecognizer
    """
    def __init__(self, *args, **kwargs):
        """
        Args
        ----
        see GaussianRecognizer.__init__
        """
        super().__init__(*args, **kwargs)

    def recognize_fit(
        self,
        features,
        n_expected_classes=None,
        **kwargs,
    ):
        """Given unknown feature space points, find new class clusters using
        a Dirichlet Process Gaussian Mixture Model.

        Args
        ----
        features : np.ndarray | torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims)
            that are the features of points treated as outliers.
        **kwargs : dict
            Key word arguments for the Dirichlet Process Gaussian Mixture Model
            as implemented in scikit-learn as the BayesianGaussianMixture.

        Side Effects
        ------------
        Any new unknown classes are added to the self.recog_label_enc, and
        self._gaussians

        Returns
        -------
        torch.Tensor
            Recognized label per feature space point.
        """
        if not self._gaussians:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')

        if isinstance(features, torch.Tensor):
            dtype = features.dtype
            #torch_features = features
            features = features.detach().cpu().numpy()
        else:
            dtype = getattr(torch, str(features.dtype))
            #torch_features = None

        default_kwargs = {
            'n_components': len(features),
            'tol': 1e-5,
            'max_iter': int(1e6),
            'n_init': 10,
            'weight_concentration_prior_type': 'dirichlet_process',
            'weight_concentration_prior': None if n_expected_classes is None
                else 1.0 / n_expected_classes,
            #'warm_start': True, would have to store DPGMM for this. Not now.
            'covariance_type': 'full',
        }
        default_kwargs.update(kwargs)

        logger.info("Begin fit of GaussianRecog's Dirichlet Process GMM.")
        logger.debug(
            "Fitting GaussianRecog's DPGMM with %d samples.",
            len(features),
        )
        dpgmm = BayesianGaussianMixture(**default_kwargs)
        recog_labels = dpgmm.fit_predict(features)

        # Select the number of new classes from DPGMM weights (Descending)
        argsorted_weights =  np.argsort(dpgmm.weights_)
        if n_expected_classes:
            argsorted_weights = argsorted_weights[::-1][:n_expected_classes]
        elif self.min_error_tol:
            # If the DPGMM weights are correct, then remove the components that
            # contribute the least to the correct number of unknown classes.
            argsorted_weights = argsorted_weights[
                dpgmm.weights_[argsorted_weights].cumsum() >= self.min_error_tol
            ][::-1]

        # Get all MVNs found to serve as new class-clusters, unless deemed
        # unconfident its a new class based on critera (min samples and
        # density)
        logger.debug(
            "%s found %d new unknown classes.",
            type(self).__name__,
            len(argsorted_weights),
        )
        if (
            len(argsorted_weights) <= 1
            or len(argsorted_weights) < self.min_samples
        ):
            # No recognized unknown classes.
            return
        if self.recog_label_enc is None:
            self.recog_label_enc = NominalDataEncoder()

        for i, arg in enumerate(argsorted_weights):
            cluster_mask = recog_labels == arg
            logger.debug(
                "%s's potential class has %d samples.",
                type(self).__name__,
                sum(cluster_mask),
            )
            if self.min_samples:
                if sum(cluster_mask) < self.min_samples:
                    continue

            mvn = MultivariateNormal(
                torch.tensor(
                    dpgmm.means_[arg],
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.tensor(
                    dpgmm.covariances_[arg],
                    dtype=self.dtype,
                    device=self.device,
                ),
            )

            #if self.min_density
            #    if mvn.log_prob(class_features).mean() \
            #        < torch.log(torch.tensor(self.min_density)):
            #        continue
            logger.debug(
                "%s's %d-th potential class has "
                '%f log_prob.mean.',
                type(self).__name__,
                i,
                mvn.log_prob(torch.tensor(features[cluster_mask])).mean(),
            )

            self._gaussians.append(mvn)
            self._thresholds.append(cred_hyperellipse_thresh(
                mvn,
                self.detect_error_tol,
            ))

            # Update the label encoder with new recognized class-clusters
            self.recog_label_enc.append(f'unknown_{self._recog_counter}')
            self._recog_counter += 1

        # Save the normalized belief of the unknowns
        self._recog_weights = dpgmm.weights_[argsorted_weights]

        # Update label_enc to include the recog_label_enc at the end.
        self.label_enc.append(self.recog_label_enc, ignore_dups=True)
