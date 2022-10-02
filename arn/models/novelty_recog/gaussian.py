"""Novelty Recognition abstract/generic class using a Gaussian distributions to
model the class' density in feature space.
"""
from copy import deepcopy
import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.models.novelty_recog.recognizer import OWHARecognizer, load_owhar

import logging
logger = logging.getLogger(__name__)


def cred_hyperellipse_thresh(mvn, min_error_tol):
    """Calculates the log prob threshold for the Multivariate Normal given the
    minimum error tolerance (alpha). Thresholding below the resulting log prob
    excludes all samples that fall outside of the confidence interval for this
    multivariate normal given alpha is the minimum error tolerable.

    Args
    ----
    mvn : torch.distributions.multivariate_normal.MultivariateNormal
        The multivariate normal distirbution whose confidence hyperellipse is
        to be found and used to determine the log prob threshold
    min_error_tol : float

    Returns
    -------
    float
        The log prob threshold for the given multivariate normal that denotes
        the samples when their log prob is less than this threshold and thus
        outside of the confidence inteval given alpha.

    Notes
    -----
    This relies on the symmetry of the normal distribution.
    """
    # Find a single dim's eigenvalue from the given covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eig(mvn.covariance_matrix)

    # Find the magnitude given the desired min_error_tol
    magnitude = torch.sqrt(
        eigenvalues[0] * chi2.ppf(1.0 - min_error_tol, mvn.loc.shape[-1])
    )

    # Find the corresponding magnitude for that dim of the covariance matrix.
    vector = (magnitude * eigenvectors[:, 0]).to(mvn.loc.dtype)

    return mvn.log_prob(mvn.loc + vector)


def closest_other_marignal_thresholds(mvns, min_thresholds):
    """Given a GMM, for each gaussian component, find the pairwise closest
    other gaussian and set the hyper ellipse to the maximum margin between the
    pair of gaussians. Use this hyper ellipse based on log_prob thresholding
    as the threshold for this

    Args
    ----
    mvns : list(torch.distributions.multivariate_normal.MultivariateNormal)
        List of MultivariateNormal objects.
    min_thresholds : list
        A List of the empirical minum thresholds per mvn.
    accepted_error : float = 1e-5
        The accepted error (alpha) used to find the credible hyper ellipse as
        fallback or reference per gaussian component if a pairwise closest
        other cannot be found.

    Returns
    -------
    list(float)
        The closest other marginal log_prob thershold per gaussian component.
    """
    raise NotImplementedError('TODO')
    return


class GaussianRecognizer(OWHARecognizer):
    """Gaussian clustering per class to perform novel class recognition.
    Recognition involves detecting novel/outlier samples to the distribution of
    classes within the feature representation space and then learning new
    clusters as new classes as enough samples are obtained.

    Note that a novel sample is synonymous to a sample that is an anomaly, an
    outlier, or out-of-distribution to the known classes.

    This fits a single Gaussian per class. Any unlabeled sample is then
    compared to its classifier predicted class cluster. If the probability of
    that sample belonging to a class cluster is below a threshold, then it is
    deemed an outlier ('unknown').

    If there are enough outliers ('unknowns') within a minimum radius, then a
    new class cluster is formed and that is marked as the next 'unknown_#'
    class.  This relates to the concept of a minimum density.

    # TODO
    min_volume : int
        The minimum volume or radius of a class cluster. This assumes all class
        Gaussians are equal covariance across all dimensions. We want to
        control the minium volume separate from min samples, as current
        implementation checks samples first then checks the min mean log_prob
        as a check for min density.

    Attributes
    ----------
    min_density : float = None
        The minimum normalized density (probability) for a cluster of unknowns
        to be considered a new class cluster. This is a value within [0, 1].
        Density is calculated as the mean of probabilities of points within a
        space.  Raises an error during fitting if the gaussians per known class
        are less than min_density. This is related to the cross entropy and
        serves as a class-cluster distribution's goodness-of-fit test of the
        found Gaussian to the data.

        If None, the default, then the minimum density is set on the fitting of
        known classes using the minimum of
        `self._gaussian[i].log_prob(samples).mean()` across all known classes.
    min_error_tol : float = 5e-3
        The minimum amount of error that is tolerable if the number of new
        unknown classes predicted is less than the correct amount of
        classes. This value is compared to the cumulative sum of the least
        likely number of components' probabilities in ascending order to
        determine the number of new unknown classes by removing the least
        likely components first. If bool(min_error_tol) == False then all
        components are used.

        This is also used to determine the threshold for each class-cluster's
        MultivariateNormal if a sample is deemed an outlier, if
        detect_error_tol is None.
    detect_error_tol : float = None
        If this is not None, then it is used to determine outliers for each
        class-cluter's multivariate normal. This is the alpha for determining
        the confidence hyperellipse per multivariate normal.
    cov_epsilon : float = 1e-12
        The torch.tensor.cov() is numerically unstable and so a small value
        will need adde to the diagonal of the resulting covariance matrix to
        avoid being treated as not a positive semi-definite matrix.
    see OWHARecognizer
    """
    def __init__(
        self,
        min_error_tol=5e-3,
        detect_error_tol=None,
        min_density=None,
        cov_epsilon=1e-12,
        **kwargs,
    ):
        """Initialize the recognizer.

        Args
        ----
        min_error_tol : see self
        detect_error_tol : see self
        min_density : see self
        cov_epsilon : see self
        see OWHARecognizer.__init__
        """
        super().__init__(**kwargs)
        if min_error_tol < 0 or min_error_tol > 1:
            raise ValueError(
                'min_error_tol must be within inclusive range [0, 1].'
            )
        self.min_error_tol = min_error_tol

        if detect_error_tol is None:
            self.detect_error_tol = self.min_error_tol
        elif detect_error_tol < 0 or detect_error_tol > 1:
            raise ValueError(
                'detect_error_tol must be within inclusive range [0, 1].'
            )
        else:
            self.detect_error_tol = detect_error_tol

        if min_density:
            raise NotImplementedError('min_density when not None.')
            if min_density < 0 or min_density > 1:
                raise ValueError(
                    'min_density must be within inclusive range [0, 1].'
                )
        self.min_density = min_density

        self.cov_epsilon = cov_epsilon
        self.min_cov_mag = 1.0

    def fit_knowns(self, features, labels, val_dataset=None):
        # Fit the Gaussians and thresholds per class-cluster. in F.Repr. space

        # NOTE decide here if this is for fitting on frepr or the ANN.:
        #   Staying with fitting in frepr for now.

        raise NotImplementedError(
            'This no longer funtions due to refactor. The inheritting class '
            'should override this method.'
        )

        if self.device is None:
            device = features.device
            self.device = device
        else:
            device = self.device
            features.to(self.device)
            labels.to(self.device)

        # Numerical stability adjustment for the sample covariance's diagonal
        stability_adjust = self.cov_epsilon * torch.eye(
            features.shape[-1],
            device=device
        )

        # Fit a Gaussian per known class, starts at 1 to ignore unknown class.
        # TODO This could made efficient by skipping knowns w/o any data changes
        self._gaussians = []
        thresholds = []
        for label in list(self.known_label_enc.inv)[
            self.known_label_enc.unknown_idx + 1:
        ]:
            mask = labels == torch.tensor(label)
            if not mask.any():
                logger.warning(
                    '%s has known class %s (idx %d) with no samples in fit()',
                    type(self).__name__,
                    self.known_label_enc.inv[label],
                    label,
                )
                continue

            class_features = features[mask].to(device, self.dtype)

            if self.min_samples:
                if class_features.shape[0] < self.min_samples:
                    #raise ValueError(
                    logger.warning(
                        "Label %d: %s features' samples < min_samples.",
                        label,
                        self.known_label_enc.inv[label]
                    )
            if class_features.shape[0] == 1:
                loc = class_features.squeeze()
                cov_mat = torch.eye(features.shape[-1], device=device) \
                    * self.min_cov_mag
            else:
                loc = class_features.mean(0)
                cov_mat = class_features.T.cov()
            try:
                mvn = MultivariateNormal(loc, cov_mat)
            except:
                mvn = MultivariateNormal(loc, cov_mat + stability_adjust)

            if self.min_density:
                if (
                    mvn.log_prob(class_features).mean()
                        < torch.log(
                            torch.tensor(self.min_density, device=device)
                        )
                ):
                    raise ValueError(
                        f"Label {label} features' normalized density "
                        '< min_density.'
                    )
            self._gaussians.append(mvn)
            min_log_prob = mvn.log_prob(class_features).min().detach()
            err_lprob = cred_hyperellipse_thresh(mvn, self.detect_error_tol)
            if err_lprob <= min_log_prob:
                thresholds.append(err_lprob)
            else:
                thresholds.append(min_log_prob)
                logger.warning(
                    'The min_log_prob for the knowns of class %d is %f which '
                    'is less than the detect error tol log prob of %f',
                    label,
                    min_log_prob,
                    err_lprob,
                )
                # NOTE it is non-trivial to change detect_err_tol given new min
            logger.debug(
                'Label %d: "%s" num samples = %d',
                label,
                self.known_label_enc.inv[label],
                len(class_features),
            )
            logger.debug(
                'Label %d: "%s" mean log_prob = %f',
                label,
                self.known_label_enc.inv[label],
                float(mvn.log_prob(class_features).mean().detach()),
            )
            logger.debug(
                'Label %d: "%s" min log_prob = %f',
                label,
                self.known_label_enc.inv[label],
                min_log_prob,
            )
            logger.debug(
                'Label %d: "%s" log_prob threshold = %f',
                label,
                self.known_label_enc.inv[label],
                float(thresholds[-1].detach()),
                #    if isinstance(thresholds[-1], torch.Tensor)
                #    else float(thresholds[-1]),
            )
        self._thresholds = thresholds

        # NOTE This is a hack for single sample cases.
        self.min_cov_mag = np.stack([
            mvn.covariance_matrix.sum() / mvn.loc.shape[-1]
            for mvn in self._gaussians
        ]).min()

    def fit(self, dataset, val_dataset=None):
        """Fits a Gaussian to each class within feature space and a finds a
        threshold over the logarithmic probability to each class' Gaussian that
        includes all knowns within and excludes all unknowns.

        Args
        ----
        features : torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims).
        labels : torch.Tensor
            1 dimensional integer tensor of shape (samples,). Contains the
            index encoding of each label per features.
        """
        dset_feedback_mask, features, labels = self.pre_fit(dataset)
        self.fit_knowns(features, labels, val_dataset)
        self.post_fit(dset_feedback_mask, features)

        # NOTE Should fit on the soft labels (output of recognize) for
        # unlabeled data. That way some semblence of info from the recog goes
        # into the predictor.
        #   Can consider fitting on known labels with a weighting that strongly
        #   prioritizes the ground truth, but sets the other class values to be
        #   proportional, albeit scaled down, to the recognize output.

        # Fit the FineTune ANN if it exists now that the labels are determined.
        super().fit_knowns(dataset, val_dataset)

    def save(self, h5, overwrite=False):
        """Save as an HDF5 file."""
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        state = dict(
            # GaussianRecognizer
            min_error_tol=self.min_error_tol,
            detect_error_tol=self.detect_error_tol,
            min_samples=self.min_samples,
            min_density=self.min_density,
            cov_epsilon=self.cov_epsilon,
            dtype=str(self.dtype)[6:],
            device=self.device.type,
        )
        for key, val in state.items():
            if val is None:
                continue
            h5.attrs[key] = val

        super().save(h5)
        if close:
            h5.close()

    @staticmethod
    def load(h5):
        return load_owhar(h5, GaussianRecognizer)
