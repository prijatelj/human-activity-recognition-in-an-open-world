"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""
from copy import deepcopy

import numpy as np
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder
from vast.clusteringAlgos.FINCH.python.finch import FINCH

from arn.models.novelty_recog.gaussian import (
    OWHARecognizer,
    GaussianRecognizer, # TODO Ensure the parent handles all experience updates
    cred_hyperellipse_thresh,
)

import logging
logger = logging.getLogger(__name__)


#@ray
def fit_multivariate_normal(
    features,
    stability_adjust=None,
    cov_epsilon=1e-12,
    device='cpu',
):
    """Construct a MultivariateNormal for a GMM."""
    if stability_adjust is None:
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )
    loc = features.mean(0)
    cov_mat = features.T.cov()
    try:
        mvn = MultivariateNormal(loc, cov_mat)
    except:
        mvn = MultivariateNormal(loc, cov_mat + stability_adjust)
    return mvn


def closest_other_marignal_thresholds(mvns):
    """Given a GMM, for each gaussian component, find the pairwise closest
    other gaussian and set the hyper ellipse to the maximum margin between the
    pair of gaussians. Use this hyper ellipse based on log_prob thresholding
    as the threshold for this

    Args
    ----
    mvns : list(torch.distributions.multivariate_normal.MultivariateNormal)
        List of MultivariateNormal objects.
    accepted_error : float = 1e-5
        The accepted error (alpha) used to find the credible hyper ellipse as
        fallback or reference per gaussian component if a pairwise closest
        other cannot be found.

    Returns
    -------
    list(float)
        The closest other marginal log_prob thershold per gaussian component.
    """
    raise NotImplementedError
    return


#@ray
def fit_gmm(
    class_name,
    features,
    recog_labels,
    n_clusters,
    counter=0,
    stability_adjust=None,
    cov_epsilon=1e-12,
    device='cpu',
    dtype='float32',
    threshold_method='cred_hyperellipse_thresh',
    min_samples=2,
    accepted_error=1e-5,
):
    """Construct a Gaussian Mixture Model as a component of larger mixture.

    Args
    ----
    features : torch.Tensor

    Returns
    -------
    GMM
        A Gaussian Mixture Model object as fit to the features.
    """
    raise NotImplementedError

    label_enc = NominalDataEncoder([], unknown_key=class_name)

    if isinstance(threshold_method, str):
        threshold_method = getattr(__name__, threshold_method)

    mvns = []
    thresholds = []
    for i in range(n_clusters):
        cluster_mask = recog_labels == i
        logger.debug(
            "`fit_gmm()`'s %d-th potential class has %d samples.",
            i,
            sum(cluster_mask),
        )
        if min_samples:
            if sum(cluster_mask) < min_samples:
                continue

        mvn = fit_multivariate_normal(
            torch.tensor(features[cluster_mask], dtype=dtype, device=device),
        )
        mvns.append(mvn)
        if threshold_method == 'cred_hyperellipse_thresh':
            thresholds.append(cred_hyperellipse_thresh(mvn, accepted_error))

        # Update the label encoder with new recognized class-clusters
        label_enc.append(f'{class_name}_{counter}')
        counter += 1

    if threshold_method == 'closest_other_marignal_thresholds':
        thresholds = closest_other_marignal_thresholds(mvns)
    return GMM(label_enc, mvns, thresholds)


def recognize_fit(
    class_name,
    features,
    counter=0,
    threshold_method='cred_hyperellipse_thresh',
    allowed_error=1e-5,
    max_likely_gmm=False,
    level=-1,
    device='cpu',
    **kwargs,
):
    """For a single class' features, fit a Gaussian Mixture Model to it using
    the clusters found by FINCH.

    Args
    ----
    class_name : str
        The name for this class. Used to construct the labels in the resulting
        label encoder as the prefix to all the labels:
            `f'{class_name}_{component_idx}'`
    features: np.ndarray | torch.Tensor
        2 dimensional float tensor of shape (samples, feature_repr_dims)
        that are the features deemed to be corresponding to the class.
    threshold_method : str 'credible_ellipse'
        The method used to find the thresholds per component, defaulting to
        use the credible ellipse per gaussian given the accepted_error (alpha).
    accepted_error : float = 1e-5
        The accepted error (alpha) to be used to find the corresponding 1-alpha
        credible ellipse per gaussian component.
    **kwargs : dict
        Key word arguments for FINCH for detecting clusters.

    Returns
    -------
    GMM
        The label encoder for this class' components, a list of the components'
        MultivariateNormal distributions and a list of the components'
        thresholds.
    """
    raise NotImplementedError

    #label_enc = NominalDataEncoder([class_name], unknown_key=class_name)

    if isinstance(features, torch.Tensor):
        dtype = features.dtype
        features = features.detach().cpu().numpy()
    else:
        dtype = getattr(torch, str(features.dtype))

    default_kwargs = {'verbose': False}
    default_kwargs.update(kwargs)

    logger.info(
        'Begin fit of FINCH for class %s, samples %d, counter %d, '
        'threshold_method %s, allowed_error %f, max_likely_gmm %s, level %d',
        class_name,
        len(features),
        counter,
        threshold_method,
        allowed_error,
        max_likely_gmm,
        level,
    )
    recog_labels, n_clusters, _ = FINCH(features, **default_kwargs)

    if max_likely_gmm:
        raise NotImplementedError('max_likely_gmm')
        # TODO MLE GMM: from lowest level to highest, fit GMM to class clusters
        #   Keep looking or most likely until next in level sequence is less
        #   likely by log_prob on the data.
    else:
        # Fit a GMM to the given class clusters using level
        gmm = fit_gmm(
            class_name,
            features,
            recog_labels[:, level],
            n_clusters[level],
            device=device,
            threshold_method=threshold_method,
        )
    return gmm


class GMM(object):
    """Gaussian Mixture Model consisting of MVN components and thresholds.

    Attributes
    ----------
    mvns : list(GMM)
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns and unknown.
    thresholds : list(float)
    label_enc : NominalDataEncoder
    """
    def __init__(self, class_name, *kwargs):
        self.label_enc = NominalDataEncoder([], unknown_key='class_name')
        raise NotImplementedError

    @property
    def class_name(self):
        return self.label_enc.unknown_key

    def log_prob(self, features):
        """The logarithmic probability of the features belonging to this GMM"""
        raise NotImplementedError

    def recognize(self, features):
        """The logarithmic probabilities of the features per MVN component."""
        raise NotImplementedError


#class GMMFINCH(GaussianRecognizer):
class GMMFINCH(OWHARecognizer):
    """Gaussian Mixture Model per class using FINCH to find the components.

    Attributes
    ----------
    gmms : list(GMM)
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns and unknown.
    label_enc : NominalDataEncoder
    """
    def __init__(self, *kwargs):
        raise NotImplementedError

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        raise NotImplementedError

        # NOTE Given multiple GMMs, find the thresholds per GMM (i don't think
        # this applies, just use detect() of the GMM.)

        # NOTE return the list of GMMs, each w/ thier own label_enc, gaussians,
        # and thresholds and methods for finding recog, detect w/in, and
        # returning the log prob of samples belonging to the entire GMM

    def recognize(self, features, detect=False):
        raise NotImplementedError

    def detect(self, features, known_only=True):
        raise NotImplementedError

    def fit_known(self, features):
        raise NotImplementedError

    def fit(self, features):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError
