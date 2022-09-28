"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""
from copy import deepcopy

import h5py
import numpy as np
import torch
F = torch.nn.functional
from torch.distributions import(
    MultivariateNormal,
    MixtureSameFamily,
    Categorical,
)

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath
from vast.clusteringAlgos.FINCH.python.finch import FINCH

from arn.models.novelty_recog.gaussian import (
    OWHARecognizer,
    GaussianRecognizer, # TODO Ensure the parent handles all experience updates
    cred_hyperellipse_thresh,
)

import logging
logger = logging.getLogger(__name__)


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
    raise NotImplementedError('TODO')
    return


#@ray
def fit_multivariate_normal(
    features,
    stability_adjust=None,
    cov_epsilon=1e-12,
    device='cpu',
):
    """Construct a MultivariateNormal for a GMM. Handles stability errors in
    the covariance matrix to pass the PositiveDefinite check.
    """
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
    label_enc = NominalDataEncoder([], unknown_key=class_name)

    if isinstance(threshold_method, str):
        threshold_method = getattr(__name__, threshold_method)

    if stability_adjust is None:
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )

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
            features[cluster_mask].to(dtype=dtype, device=device),
            stability_adjust,
            device=device,
        )
        mvns.append(mvn)
        if threshold_method == 'cred_hyperellipse_thresh':
            thresholds.append(cred_hyperellipse_thresh(mvn, accepted_error))

        # Update the label encoder with new recognized class-clusters
        label_enc.append(f'{class_name}_{counter}')
        counter += 1

    if threshold_method == 'closest_other_marignal_thresholds':
        thresholds = closest_other_marignal_thresholds(mvns)

    return GMM(label_enc, mvns, thresholds=thresholds)


def recognize_fit(
    class_name,
    features,
    counter=0,
    threshold_method='cred_hyperellipse_thresh',
    allowed_error=1e-5,
    max_likely_gmm=False,
    level=-1,
    stability_adjust=None,
    cov_epsilon=1e-12,
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

    if stability_adjust is None:
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )

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
            dtype=dtype,
            threshold_method=threshold_method,
            stability_adjust=stability_adjust,
            cov_epsilon=cov_epsilon,
        )
    return gmm


class GMM(object):
    """Gaussian Mixture Model consisting of MVN components and thresholds.

    Attributes
    ----------
    label_enc : NominalDataEncoder
    gmm : MixtureSameFamily
        The list of MultivariateNormals per class (component) is at
    thresholds : list(float)
    """
    def __init__(
        self,
        class_name,
        locs=None,
        covariance_matrices=None,
        thresholds=None,
        mix=None,
        #counter=0,
    ):
        self.set_label_enc(class_name)
        self.set_gmm(locs, covariance_matrices, mix)
        self.set_thresholds(thresholds)
        #self.counter = counter

    @property
    def class_name(self):
        return self.label_enc.unknown_key

    def set_label_enc(self, label_enc):
        if isinstance(label_enc, NominalDataEncoder):
            assert label_enc.unknown_key is not None
            self.label_enc = label_enc
        else:
            self.label_enc = NominalDataEncoder([], unknown_key=label_enc)

    def set_gmm(self, locs=None, covariance_matrices=None, mix=None):
        """Sets the gmm with given locs, covariance matrices, and mix."""
        if locs is None:
            if covariance_matrices is not None:
                raise ValueError(
                    'No `locs` given when `covariance_matrices` given.'
                )
            self.gmm = None
            return

        if mix is None:
            mix = Categorical(torch.tensor(
                [1 / locs.shape[0]] * locs.shape[0],
                dtype=locs.dtype,
            ))

        if isinstance(locs, MultivariateNormal):
            self.gmm = MixtureSameFamily(mix, locs)
            return
        elif (
            isinstance(locs, list)
            and all([isinstance(x, MultivariateNormal) for x in locs])
        ):
            if covariance_matrices is not None:
                raise ValueError(
                    '`locs` is list of MultivariateNormal objects which '
                    'typically have their loc and covariance_matrix '
                    'parameters extracted, but `covariance_matrices` is not '
                    'None!'
                )
            mvns = locs
            locs = []
            covariance_matrices = []
            for mvn in locs:
                locs.append(mvn.loc)
                covariance_matrices.append(mvn.covariance_matrix)
            locs = torch.stack(locs)
            covariance_matrices = torch.stack(covariance_matrices)

        self.gmm = MixtureSameFamily(
            mix,
            MultivariateNormal(locs, covariance_matrices),
        )

    def set_thresholds(self, thresholds):
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        if not isinstance(thresholds, torch.Tensor):
            thresholds = torch.tensor(thresholds)
        thresholds = thresholds.reshape(-1)
        if (
            thresholds.shape[0]
            != self.gmm.component_distribution.batch_shape
         ):
            raise ValueError('Thresholds flattened dims != number of mvns')
        self.thresholds = thresholds

    def log_prob(self, features):
        """The logarithmic probability of the features belonging to this GMM"""
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        return self.gmm.log_prob(features)

    def comp_log_prob(self, features):
        """The log_prob of each component per sample."""
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        return self.gmm.component_distribution.log_prob(
            features.unsqueeze(1).expand(
                -1,
                self.gmm.component_distribution.batch_shape,
                -1,
            )
        )

    def recognize(self, features, detect=True):
        """The logarithmic probabilities of the features per MVN component.

        Args
        ----
        features : torch.Tensor
        detect : bool = True
            If True, includes the detected

        Returns
        -------
        torch.Tensor
            The softmax normalized probability vectors of the component log
            probs per sample (row). If detect is True, then the first index
            (self.label_enc.unknown_idx) is included amongst the components by
            detection through the thresholding over the invidual components'
            log probs.
        """
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        raise NotImplementedError

        recogs = self.comp_log_prob(features)

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

    def detect(self, features):
        """Detects samples belong to the general class, or a single component.
        """
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        return (self.comp_log_prob(features) < self.thresholds).all(1)

    def save(self, h5, overwrite=False):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        for key in ['label_enc', 'locs', 'covariance_matrices', 'thresholds']:
            if key in  h5:
                raise ValueError(f'{key} already in given h5 dataset!')

        h5['label_enc'] = np.array(self.label_enc).astype(np.string_)
        h5['locs'] = self.gmm.component_distribution.loc.detach().cpu().numpy()
        h5['covariance_matrices'] = (
            self.gmm.component_distribution.covariance_matrix
        ).detach().cpu().numpy()
        h5['thresholds'] = self.thresholds.detach().cpu().numpy()
        h5['mix'] = self.gmm.mixture_distribution.probs.detach().cpu().numpy()

        if close:
            h5.close()

    @staticmethod
    def load(h5):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(h5, 'r')
        loaded = GMM(
            NominalDataEncoder(
                np.array(h5['label_enc'], dtype=str),
                unknown_idx=0,
            ),
            torch.tensor(h5['locs']),
            torch.tensor(h5['covariance_matrices']),
            torch.tensor(h5['thresholds']),
            np.array(h5['mixes']),
        )
        if close:
            h5.close()
        return loaded


#class GMMFINCH(GaussianRecognizer):
class GMMFINCH(OWHARecognizer):
    """Gaussian Mixture Model per class using FINCH to find the components.

    Attributes
    ----------
    gmms : list(GMM)
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns and unknown.
    label_enc : NominalDataEncoder
        The label encoder corresponding to the classes with GMMs. This follows
        the order of unknown (index 0), then knowns.
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

    def save(self, h5):
        raise NotImplementedError

    @staticmethod
    def load(h5):
        raise NotImplementedError
