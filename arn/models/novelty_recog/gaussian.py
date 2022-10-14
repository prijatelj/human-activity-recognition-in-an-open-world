"""Novelty Recognition abstract/generic class using a Gaussian distributions to
model the class' density in feature space.
"""
from abc import abstractmethod
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


def min_max_threshold(
    distribs,
    samples: torch.Tensor,
    likelihood: float = 0.0,
    batch_size=None,
):
    """For all the mvns over the data, find the sample with the minimum of the
    maximum log_probs. This is now the threshold to be used overall

    Args
    ----
    distribs :
    samples : torch.Tensor
    likelihood : float = 0
        The likelihood used to specify how likely a sample is unknown to the
        the minimum maximum log prob sample of a distribution. We recommend
        zero or negative values as it is added to the log_prob, and subtraction
        is then saying it is less likely, e.g., likelihood of -1 means the
        (currnetly static) prior belief is that the unknown samples will be
        half as likely as the least likely known class any sample was assigned
        to, assuming base 2. A likelihood of -2 would be 1/4 as likely.
    batch_size : int = 8192
        The memory required to calculate the gmm.log_prob per distribution is
        `memory_required = dims * components * samples * 4 Bytes`. 2**13 = 8192
        samples which when dims are 768 and max components is 4000, <51GB of
        RAM is required.
    """
    if batch_size is None:
        batch_size = len(samples) #8192
    logger.debug(
        'min_max_threshold: '
        'type(distribs) = %s; '
        'samples.shape = %s; '
        'samples.dtype = %s; '
        'likelihood = %s; ',
        type(distribs),
        samples.shape,
        samples.dtype,
        likelihood,
    )
    if (
        isinstance(distribs, list)
        and all([hasattr(d, 'log_prob') for d in distribs])
    ):
        logger.debug('list: len(distribs) = %s', len(distribs))
        #log_probs = torch.stack([d.log_prob(samples) for d in distribs], dim=1)
        #log_probs = []
        min_maxes = torch.tensor(torch.inf)
        for idx in range(batch_size, len(samples) + batch_size, batch_size):
            batch = samples[idx - batch_size:idx]
            log_probs = torch.tensor(
                [[-torch.inf]] * len(batch),
                dtype=samples.dtype,
            )
            for i, d in enumerate(distribs):
                log_probs = torch.max(
                    log_probs,
                    d.log_prob(batch).reshape(-1, 1),
                )
            min_maxes = torch.min(min_maxes, log_probs.min())
        #logger.debug(
        #    'the %d-th distrib, w/ unknown_key = %s; %d components',
        #    i,
        #    'No label_enc attr' if not hasattr(d, 'label_enc')
        #        else d.label_enc.unknown_key,
        #    1 if not hasattr(d, 'gmm') else
        #        d.gmm.component_distribution.batch_shape[0],
        #)
        #log_probs = torch.stack(log_probs, dim=1)
        #logger.debug('list: log_probs.shape = %s', log_probs.shape)
        #log_probs = log_probs.max(1).values
        #logger.debug('log_probs.max(1).values.shape = %s', log_probs.shape)
    elif hasattr(distribs, 'log_prob'):
        log_probs = distribs.log_prob(samples)
        #logger.debug('has log_prob(): log_probs.shape = %s', log_probs.shape)
        min_maxes = log_probs.min()
    else:
        raise ValueError(
            'Expected either an object or list of objects with .log_prob().'
        )

    logger.debug('log_probs.min() = %s', min_maxes)
    if likelihood:
        return min_maxes + likelihood
    return min_maxes


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
    cov_epsilon : float = None
        The torch.tensor.cov() is numerically unstable and so a small value
        will need adde to the diagonal of the resulting covariance matrix to
        avoid being treated as not a positive semi-definite matrix.
    threshold_func : str = 'cred_hyperellipse_thresh'
        The function to use for finding the thresold for known to unknown.
    threshold_global : bool = False
        If True, applies the threshold of the model globally to the log_probs
        of all the known distributions. Otherwise, assesses each known with
        it own local threshold. For local detection, overall detection of an
        unknown occurs when all known distribs detect a sample as unknown.
    _thresholds : float = None
    detect_likelihood : float = 0.0
        see min_max_threshold.likelihood
    batch_size : int = None
    see OWHARecognizer
    """
    def __init__(
        self,
        min_error_tol=5e-3,
        detect_error_tol=None,
        min_density=None,
        cov_epsilon=None,
        threshold_func='cred_hyperellipse_thresh',
        threshold_global=False,
        detect_likelihood : float = 0.0,
        batch_size : int = None,
        **kwargs,
    ):
        """Initialize the recognizer.

        Args
        ----
        min_error_tol : see self
        detect_error_tol : see self
        min_density : see self
        cov_epsilon : see self
        threshold_func : see self
        threshold_global : see self
        detect_likelihood : see self
        batch_size : see self
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
        if cov_epsilon is None:
            self.cov_epsilon = torch.finfo(self.dtype).eps * 10
        else:
            self.cov_epsilon = cov_epsilon

        self.min_cov_mag = 1.0
        self.threshold_func = threshold_func
        self.threshold_global = threshold_global
        self._thresholds = None
        self._detect_likelihood = detect_likelihood
        self.batch_size = int(batch_size) if batch_size else None

    @property
    def thresholds(self):
        """Global threshold for all distribs involved. If None, use the
        internal distribs thresholding for detection.
        """
        return self._thresholds

    def set_detect_likelihood(self, value):
        self._detect_likelihood = value

    @abstractmethod
    def fit_knowns(self, features, labels, val_dataset=None):
        raise NotImplementedError('Inheriting class overrides this.')

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
        self._increment += 1
        logger.info(
            "Begin call to %s's %s.pre_fit()",
            self.uid,
            type(self).__name__,
        )
        dset_feedback_mask, features, labels = self.pre_fit(dataset)
        logger.info(
            "Begin call to %s's %s.fit_knowns()",
            self.uid,
            type(self).__name__,
        )
        self.fit_knowns(features, labels, val_dataset)
        logger.info(
            "Begin call to %s's %s.post_fit()",
            self.uid,
            type(self).__name__,
        )
        self.post_fit(dset_feedback_mask, features)

        if (
            self.increment == 1
            and self.threshold_func == 'min_max_threshold'
            and self.threshold_global
        ):
            # TODO use initial increment val_dataset to  fit the
            # detect_likelihood as the difference between the self.threhsolds
            # and the self.min_error_tol quantile of the sample's max log_prob
            # values.  Able to be done w/o lookig at labels as all are known.
            val_features = []
            for feature_tensor, label in val_dataset:
                val_features.append(feature_tensor)
            del feature_tensor, label
            log_probs = self.recognize(
                torch.stack(val_features)
            )
            max_log_probs = (
                log_probs[log_probs < self.thresholds]
            ).max(1).values
            if max_log_probs.any():
                if self.min_error_tol:
                    min_max = torch.quantiles(
                        max_log_probs,
                        self.min_error_tol,
                    )
                else:
                    min_max = max_log_probs.min()
                if min_max < self.thresholds:
                    self.set_detect_likelihood(min_max - self.thresholds)

        # NOTE Should fit on the soft labels (output of recognize) for
        # unlabeled data. That way some semblence of info from the recog goes
        # into the predictor.
        #   Can consider fitting on known labels with a weighting that strongly
        #   prioritizes the ground truth, but sets the other class values to be
        #   proportional, albeit scaled down, to the recognize output.

        # Fit the FineTune ANN if it exists now that the labels are determined.
        logger.info(
            "Begin call to %s's super(%s).fit()",
            self.uid,
            type(self).__name__,
        )
        super().fit_knowns(dataset, val_dataset)
        logger.info(
            "End call to %s's %s.fit()",
            self.uid,
            type(self).__name__,
        )

    def save(self, h5, overwrite=False):
        """Save as an HDF5 file."""
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        state = dict(
            # GaussianRecognizer
            min_error_tol=self.min_error_tol,
            detect_error_tol=self.detect_error_tol,
            min_density=self.min_density,
            cov_epsilon=self.cov_epsilon,
            threshold_func=self.threshold_func
                if isinstance(self.threshold_func, str)
                else str(self.threshold_func),
            threshold_global=self.threshold_global,
            detect_likelihood=self.detect_likelihood,
            batch_size=self.batch_size,
        )
        for key, val in state.items():
            if val is None:
                continue
            h5.attrs[key] = val

        if self._thresholds:
            h5['_thresholds'] = self._thresholds.detach().cpu().numpy()

        super().save(h5)
        if close:
            h5.close()

    @staticmethod
    def load(h5):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(h5, 'r')

        loaded = load_owhar(h5, GaussianRecognizer)
        if '_thresholds' in h5:
            loaded._thresholds = torch.tensor(h5['_thresholds'])

        if close:
            h5.close()
        return loaded
