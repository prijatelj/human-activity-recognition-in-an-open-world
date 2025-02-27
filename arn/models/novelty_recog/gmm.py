"""Gaussian Mixture Model utility functions and class with unsupervised
finding of the gaussians by using the partitions resulting from FINCH.

Notes
-----
I took a more "functional" approach to this, which you'll see results in a lot
of arguments being passed around. I did this to experiment and learn the
balance I want when writing objects and functions. I think its good to specify
all the calls to `self` attributes in a function/method, but I would definitely
will use those calls more when writing something like this where the functions
are heavily tied to the object. More DRY please and the reason for methods or
for structs/grouped together state by semantics.  The important thing I was
experimenting with is specificaiton in practice of those methods as functions
in order to answer the question: What is the complete set of inputs and
outputs?
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

from arn.torch_utils import torch_dtype
from arn.models.novelty_recog.recognizer import join_label_encs, h5_io
from arn.models.novelty_recog.gaussian import (
    GaussianRecognizer,
    cred_hyperellipse_thresh,
    min_max_threshold,
)


import logging
logger = logging.getLogger(__name__)


def join_gmms(left, right, use_right_key=True):
    """Joins the right GMM into the left to form a new single GMM.

    Args
    ----
    left : GMM
    right : GMM
    use_right_key : bool = True
        The order of labels will always follow left then right, but the
        unknown_key and other attributes used are specified by this flag,
        defaulting to the right GMM's attributs unless use_right_key is False.
    """
    if right.gmm is None:
        return left
    src = right if use_right_key else left
    label_enc = join_label_encs(left.label_enc, right.label_enc, use_right_key)
    mix = len(label_enc) - 1
    mix = torch.tensor([1 / mix] * mix, dtype=src.dtype)
    # NOTE creates uniform mixture, not combining them and normalizing
    #torch.cat([left.mix, right.mix]),
    return GMM(
        label_enc,
        torch.cat([
            left.gmm.component_distribution.loc,
            right.gmm.component_distribution.loc,
        ]),
        torch.cat([
            left.gmm.component_distribution.covariance_matrix,
            right.gmm.component_distribution.covariance_matrix,
        ]),
        torch.cat([left.thresholds, right.thresholds]),
        mix,
        src.counter,
        src.cov_epsilon,
        src.device,
        src.dtype,
        src.threshold_func,
        src.min_samples,
        src.accepted_error,
        src.detect_likelihood,
        src.batch_size,
    )


# NOTE perhaps @ray?
def fit_multivariate_normal(
    features,
    stability_adjust=None,
    cov_epsilon=None,
    device='cpu',
    cov_mat=None
):
    """Construct a MultivariateNormal for a GMM. Handles stability errors in
    the covariance matrix to pass the PositiveDefinite check.
    """
    if stability_adjust is None:
        if cov_epsilon is None:
            cov_epsilon = torch.finfo(features.dtype).eps * 10
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )
    loc = features.mean(0)
    if cov_mat is None:
        cov_mat = features.T.cov()
    try:
        mvn = MultivariateNormal(loc, cov_mat)
    except:
        mvn = MultivariateNormal(loc, cov_mat + stability_adjust)
    return mvn


#@ray
def fit_gmm(
    label_enc,
    features,
    recog_labels,
    n_clusters=None,
    counter=0,
    stability_adjust=None,
    cov_epsilon=None,
    device='cpu',
    dtype='float32',
    threshold_func='cred_hyperellipse_thresh',
    min_samples=2,
    accepted_error=1e-5,
    detect_likelihood=0.0,
    batch_size=None,
    return_kwargs=True,
):
    """Construct a Gaussian Mixture Model as a component of larger mixture.
    Given this algorithm implies accepting that the points are within each
    designated cluster, if the sample with the minimum log prob is less than
    the found thresholding log_prob given threshold_func, then the minium
    log prob will be used instead.

    Args
    ----
    label_enc : str | NominalDataEncoder
    features : torch.Tensor
    recog_labels : list(int)
        The list of integers as found by FINCH to indicate the cluster each
        sample belongs to, otherwise the label_enc index encoding without any
        zeros as the catch-all class belongs to the zeroth index and does not
        recieve a MultivariateNormal distribution.
    n_clusters : int = None

    Returns
    -------
    GMM
        A Gaussian Mixture Model object as fit to the features.
    """
    update_label_enc = not isinstance(label_enc, NominalDataEncoder)
    if update_label_enc:
        if n_clusters is None:
            raise ValueError(
                '`n_clusters` is required when label_enc is not a '
                'NominalDataEncoder'
            )
        label_enc = NominalDataEncoder([label_enc], unknown_key=label_enc)
    else:
        if n_clusters is not None:
            raise ValueError(
                '`n_clusters` given when label_enc is a NominalDataEncoder!'
                'label_enc.inv is used inplace of range(n_clusters)'
            )
        n_clusters = iter(label_enc.inv)
        if label_enc.unknown_idx is not None:
            assert label_enc.unknown_idx == 0
            next(n_clusters)

    #if isinstance(threshold_func, str):
    #    threshold_func = globals()[threshold_func]

    if cov_epsilon is None:
        cov_epsilon = torch.finfo(features.dtype).eps * 10
    if stability_adjust is None:
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )

    mvns = []
    if threshold_func is None:
        thresholds = None
    else:
        thresholds = []
    for i in range(n_clusters) if update_label_enc else n_clusters:
        cluster_mask = recog_labels == i
        logger.debug(
            "`fit_gmm()`'s %d-th potential class has %d samples.",
            i,
            sum(cluster_mask),
        )
        #if min_samples:
        if sum(cluster_mask) < min_samples:
            #continue
            cov_mat = stability_adjust
        elif min_samples <= 1 and sum(cluster_mask) == 1:
            # NOTE this is probably an issue, was setting to overall min
            # magnitude before, but this works for now.
            cov_mat = stability_adjust
        else:
            cov_mat = None

        cluster_features = features[cluster_mask].to(device, dtype)
        mvn = fit_multivariate_normal(
            cluster_features,
            stability_adjust,
            device=device,
            cov_mat=cov_mat,
        )
        mvns.append(mvn)
        min_log_prob = mvn.log_prob(cluster_features).min().detach()
        if threshold_func == 'cred_hyperellipse_thresh':
            err_lprob = cred_hyperellipse_thresh(mvn, accepted_error)
            if err_lprob <= min_log_prob:
                thresholds.append(err_lprob)
            else:
                thresholds.append(min_log_prob)
        elif isinstance(thresholds, list):
            thresholds.append(min_log_prob)

        # Update the label encoder with new recognized class-clusters
        if update_label_enc:
            label_enc.append(f'{label_enc.unknown_key}_{counter}')
            counter += 1

    if threshold_func == 'min_max_threshold':
        thresholds = min_max_threshold(
            mvns,
            features,
            detect_likelihood,
            batch_size,
        )

    if return_kwargs:
        return {'label_enc': label_enc, 'locs': mvns, 'thresholds': thresholds}
    return GMM(
        label_enc,
        mvns,
        thresholds=thresholds,
        counter=counter,
        cov_epsilon=cov_epsilon,
        device=device,
        dtype=dtype,
        threshold_func=threshold_func,
        min_samples=min_samples,
        accepted_error=accepted_error,
        detect_likelihood=detect_likelihood,
        batch_size=batch_size,
    )


def recognize_fit(
    class_name,
    features,
    counter=0,
    max_likely_gmm=False,
    level=-1,
    stability_adjust=None,
    cov_epsilon=None,
    device='cpu',
    dtype=None,
    min_samples=2,
    threshold_func='cred_hyperellipse_thresh',
    accepted_error=1e-5, # TODO change name to accepted_error?
    detect_likelihood=0.0,
    batch_size=None,
    return_kwargs=False,
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
    threshold_func : str 'credible_ellipse'
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
        if dtype is None:
            dtype = features.dtype
        finch_features = features.detach().cpu().numpy()
    else:
        if dtype is None:
            dtype = getattr(torch, str(features.dtype))
        finch_features = features
        features = torch.tensor(features, dtype=dtype)

    default_kwargs = {'verbose': False}
    default_kwargs.update(kwargs)

    logger.info(
        'Begin fit of FINCH for class %s, samples %d, counter %d, '
        'threshold_func %s, accepted_error %f, max_likely_gmm %s, level %d '
        'detect_likelihood %f, batch_size %s',
        class_name,
        len(features),
        counter,
        threshold_func,
        accepted_error,
        max_likely_gmm,
        level,
        detect_likelihood,
        batch_size,
    )
    recog_labels, n_clusters, _ = FINCH(finch_features, **default_kwargs)
    del finch_features

    if cov_epsilon is None:
        cov_epsilon = torch.finfo(features.dtype).eps * 10
    if stability_adjust is None:
        # Numerical stability adjustment for the sample covariance diagonal
        stability_adjust = cov_epsilon * torch.eye(
            features.shape[-1],
            device=device,
        )
    """
    if len(n_clusters) == 1 and n_clusters[-1] == 1:
        logger.debug(
            'Resulting GMM for %s kept 0 / 1 potential clusters',
            class_name,
        )
        gmm = GMM(
            class_name,
            cov_epsilon=cov_epsilon,
            device=device,
            dtype=dtype,
            threshold_func=threshold_func,
            # min samples, accepted error
        )
    #"""
    if max_likely_gmm:
        raise NotImplementedError('max_likely_gmm')
        # TODO MLE GMM: from lowest level to highest, fit GMM to class clusters
        #   Keep looking or most likely until next in level sequence is less
        #   likely by log_prob on the data.
    else:
        logger.debug(
            'FINCH for %s with level %d found %d potential clusters',
            class_name,
            level,
            n_clusters[level],
        )

        # Fit a GMM to the given class clusters using level
        gmm = fit_gmm(
            class_name,
            features,
            recog_labels[:, level],
            n_clusters[level],
            stability_adjust=stability_adjust,
            device=device,
            dtype=dtype,
            threshold_func=threshold_func,
            min_samples=min_samples,
            accepted_error=accepted_error,
            detect_likelihood=detect_likelihood,
            batch_size=batch_size,
            return_kwargs=return_kwargs,
        )
        # NOTE kept has len(label_enc)-1 to exclude the 1st catch-all class
        logger.debug(
            'Resulting GMM for %s kept %d / %d potential clusters',
            class_name,
            len(gmm['label_enc'] if return_kwargs else gmm.label_enc) - 1,
            n_clusters[level],
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
    counter : int = 0
    cov_epsilon: float = None
    device: str = 'cpu'
    dtype: str = 'float32'
    threshold_func: str = 'cred_hyperellipse_thresh'
    min_samples: int = 2
    accepted_error: float = 1e-5
    detect_likelihood: float = 0.0
        Applied to min_max_threshold(likelihood=detect_likelihood)
    batch_size : int = None
        The default batch_size to use for log_prob calls. Defaults to no
        batching.
    """
    def __init__(
        self,
        label_enc,
        locs=None,
        covariance_matrices=None,
        thresholds=None,
        mix=None,
        counter: int= 0,
        cov_epsilon: float = None,
        device: str = 'cpu',
        dtype: str = 'float32',
        threshold_func: str = 'cred_hyperellipse_thresh',
        min_samples: int = 2,
        accepted_error: float = 1e-5,
        detect_likelihood: float = 0.0,
        batch_size : int = None,
    ):
        self.counter = counter
        self.set_label_enc(label_enc)
        self.dtype = torch_dtype(dtype)
        if cov_epsilon is None:
            self.cov_epsilon = torch.finfo(self.dtype).eps * 10
        else:
            self.cov_epsilon = cov_epsilon

        self.device = torch.device(device)
        self.threshold_func = threshold_func
        self.min_samples = min_samples
        self.accepted_error = accepted_error
        self.detect_likelihood = detect_likelihood
        self.batch_size = batch_size

        self.set_gmm(locs, covariance_matrices, mix)
        self.set_thresholds(thresholds)

    @property
    def class_name(self):
        return self.label_enc.unknown_key

    def set_label_enc(self, label_enc):
        if isinstance(label_enc, NominalDataEncoder):
            self.label_enc = label_enc
            if label_enc.unknown_idx is None:
                self.counter += len(label_enc) - 1
            else:
                assert label_enc.unknown_idx == 0
                self.counter += len(label_enc) - 1
        else:
            self.label_enc = NominalDataEncoder(
                [label_enc],
                unknown_key=label_enc,
            )

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
                [1 / len(locs)] * len(locs),
                dtype=self.dtype,
            ))
        elif isinstance(mix, torch.Tensor):
            mix = Categorical(mix)

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
            for mvn in mvns:
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
            if thresholds is None:
                self.thresholds = None
                return
            else:
                raise ValueError('`self.gmm` is None, must set gmm!')
        if not isinstance(thresholds, torch.Tensor):
            thresholds = torch.tensor(thresholds)
        thresholds = thresholds.reshape(-1)
        if (
            thresholds.shape != self.gmm.component_distribution.batch_shape
            and self.threshold_func != 'min_max_threshold'
         ):
            raise ValueError('Thresholds flattened dims != number of mvns')
        elif (
            (len(thresholds.shape) != 1 or thresholds.shape[0] != 1)
            and self.threshold_func != 'min_max_threshold'
        ):
            raise ValueError(
                'Thresholds flattened dims != 1 when '
                'thresholf_func="min_max_threshold"'
            )
        self.thresholds = thresholds

    def log_prob(self, features, batch_size=None):
        """The logarithmic probability of the features belonging to this GMM.
        Notes
        -----
        Batching to avoid gmm log_prob requesting too much memory (OOM).
        """
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        if batch_size is None and self.batch_size is not None:
            batch_size = self.batch_size
        else:
            return self.gmm.log_prob(features)

        log_probs = torch.empty(
            [features.shape[0]],
            dtype=self.dtype,
            device=self.device,
        )
        for idx in range(batch_size, len(features) + batch_size, batch_size):
            log_probs[idx - batch_size:idx] = \
                self.gmm.log_prob(features[idx - batch_size:idx])

        return log_probs

    def comp_log_prob(self, features):
        """The log_prob of each component per sample."""
        if self.gmm is None:
            raise ValueError('`self.gmm` is None, must set gmm!')
        return self.gmm.component_distribution.log_prob(
            features.unsqueeze(1).expand(
                -1,
                self.gmm.component_distribution.batch_shape[0],
                -1,
            )
        )

    def fit(self, *args, use_label_enc=False, **kwargs):
        """Fit the existing GMM with the given features and labels.
        Args
        ----
        use_label_enc : bool = False
            If use_label_enc is True, then the current GMM's label enc is used
            as the known labels that all have their own respective
            gaussian. Otherwise, creates a new label encoder with the same
            unknown_key based on the given labels.
        """
        params = fit_gmm(
            self.label_enc if use_label_enc else self.label_enc.unknown_key,
            *args,
            counter=self.counter,
            return_kwargs=True,
            cov_epsilon=self.cov_epsilon,
            device=self.device,
            dtype=self.dtype,
            threshold_func=self.threshold_func,
            min_samples=self.min_samples,
            accepted_error=self.accepted_error,
            detect_likelihood=self.detect_likelihood,
            batch_size=self.batch_size,
            **kwargs
        )
        self.set_label_enc(params['label_enc'])
        self.set_gmm(params['locs'])
        self.set_thresholds(params['thresholds'])

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

    @h5_io('w')
    def save(self, h5):
        for key in ['label_enc', 'locs', 'covariance_matrices', 'thresholds']:
            if key in  h5:
                raise ValueError(f'{key} already in given h5 dataset!')

        #h5['label_enc'] = np.array(self.label_enc).astype(np.string_)
        self.label_enc.save(h5.create_group('label_enc'))
        h5['locs'] = self.gmm.component_distribution.loc.detach().cpu().numpy()
        h5['covariance_matrices'] = (
            self.gmm.component_distribution.covariance_matrix
        ).detach().cpu().numpy()
        h5['thresholds'] = self.thresholds.detach().cpu().numpy()
        h5['mix'] = self.gmm.mixture_distribution.probs.detach().cpu().numpy()

        state = dict(
            counter=self.counter,
            cov_epsilon=self.cov_epsilon,
            device=self.device.type,
            dtype=str(self.dtype)[6:],
            threshold_func=self.threshold_func,
            min_samples=self.min_samples,
            accepted_error=self.accepted_error,
            detect_likelihood=self.detect_likelihood,
            batch_size=self.batch_size,
        )
        for key, val in state.items():
            if val is None:
                continue
            h5.attrs[key] = val

    @h5_io
    @staticmethod
    def load(h5):
        if hasattr(h5['label_enc'], 'keys'):
            nde = NominalDataEncoder.load_h5(h5['label_enc'])
        else:
            # Use hotfix for older model states that only save byte string of
            # labels.
            nde = load_h5_gmm_hotfix(h5['label_enc'])

        loaded = GMM(
            nde,
            # NOTE tmp hotfix for models saved before keys dtypes was saved
            torch.tensor(h5['locs']),
            torch.tensor(h5['covariance_matrices']),
            torch.tensor(h5['thresholds']),
            torch.tensor(np.array(h5['mix'])),
            # Everything else!
            **dict(h5.attrs.items())
        )

        return loaded


def load_h5_gmm_hotfix(h5):
    """Issue: old versions only saved the keys as byte strings.

    _argsorted_keys not existing in older versions may cause an error in
    future, but i currently do not think so with exputils v0.1.7
    """
    labels = np.array(h5, dtype=str),
    return NominalDataEncoder(
        unknown_key=str(labels[0]),
        unknown_idx=0,
    )


class GMMRecognizer(GaussianRecognizer):
    """GMM Recognizer generic methods. This is not intended to be initialized
    itself, but rather inheritted as it defines the GMM part for the unknowns.

    Attributes
    ----------
    level : int = -1
        The level of cluster partitions to use during recognition_fit. FINCH
        returns three levels of clustering. Defaults to the final level with
        maximum clusters.
    unknown_likelihood : float = None
        Relative log likelihood to the min_max recognized unknown gaussians to
        determine is a sample is the catch-all unknown. Defaults to same as
        self.detect_likelihood.
    unknown_gmm : GMM = None
    see GaussianRecognizer
    """
    def __init__(self, level=-1, unknown_likelihood=None, *args, **kwargs):
        """Initialize the recognizer

        Args
        ----
        level : see self
        unknown_likelihood : see self
        see GaussianRecognizer.__init__
        """
        super().__init__(*args, **kwargs)

        # All GMMRecognizers will be using FINCH, so put specific params here.
        self.level = level

        # unknown gmm, as in recognize_fit.
        self.unknown_gmm = None
        if unknown_likelihood is None:
            self.unknown_likelihood = self.detect_likelihood
        else:
            self.unknown_likelihood = unknown_likelihood

    def set_detect_likelihood(self, value):
        if self.unknown_likelihood == self._detect_likelihood:
            self.unknown_likelihood = value
        self._detect_likelihood = value

    @property
    def n_recog_labels(self):
        """The number of labels in recog_label_enc."""
        return 0 if not self.has_recogs else len(self.recog_label_enc) - 1

    @property
    def recog_label_enc(self):
        if self.unknown_gmm is not None:
            return self.unknown_gmm.label_enc

    @property
    def has_recogs(self):
        """Checks if there are any recognized labels."""
        return bool(self.recog_label_enc) and len(self.recog_label_enc) > 1

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        self.unknown_gmm = None

    def recognize_fit(
        self,
        features,
        n_expected_classes=None,
        **kwargs,
    ):
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
            level=self.level,
            cov_epsilon=self.cov_epsilon,
            device=self.device,
            dtype=self.dtype,
            threshold_func=self.threshold_func,
            min_samples=self.min_samples,
            accepted_error=self.min_error_tol,
            detect_likelihood=self.unknown_likelihood,
            batch_size=self.batch_size,
            **kwargs,
        )

    @h5_io('w')
    def save(self, h5):
        # Save the attrs unique to this object
        h5.attrs['level'] = self.level

        # Save unknown_gmm
        if self.unknown_gmm:
            self.unknown_gmm.save(h5.create_group('unknown_gmm'))

        super().save(h5)

    @h5_io
    @staticmethod
    def load(h5, class_type=None):
        # Call parent load() to load the init object w/ changes wrt ancestors.
        if class_type is None:
            loaded = GaussianRecognizer.load(h5, GMMRecognizer)
        else:
            loaded = GaussianRecognizer.load(h5, class_type)

        if 'unknown_gmm' in h5:
            loaded.unknown_gmm = GMM.load(h5['unknown_gmm'])
            loaded.unknown_likelihood = loaded.unknown_gmm.detect_likelihood
        else:
            loaded.unknown_gmm = None
            loaded.unknown_likelihood = loaded.detect_likelihood

        return loaded

    def load_state(self, h5, return_tmp=False, **kwargs):
        print('GMMRecognizer type(self) = ', type(self))
        tmp = super().load_state(h5, True, **kwargs)

        print("@" * 30)
        print("gmm.py line 869 load_state")
        print("self.label_enc", self.label_enc, type(self.label_enc))
        print("self.known_label_enc", self.known_label_enc, type(self.known_label_enc))
        print("self.recog_label_enc", self.recog_label_enc, type(self.recog_label_enc))

        print("self._label_enc", self._label_enc, type(self._label_enc))
        print("self._known_label_enc", self._known_label_enc, type(self._known_label_enc))
        print("self._recog_label_enc", self._recog_label_enc, type(self._recog_label_enc))

        print("tmp.label_enc", tmp.label_enc, type(tmp.label_enc))
        print("tmp.known_label_enc", tmp.known_label_enc, type(tmp.known_label_enc))
        print("tmp.recog_label_enc", tmp.recog_label_enc, type(tmp.recog_label_enc))

        print("tmp._label_enc", tmp._label_enc, type(tmp._label_enc))
        print("tmp._known_label_enc", tmp._known_label_enc, type(tmp._known_label_enc))
        print("tmp._recog_label_enc", tmp._recog_label_enc, type(tmp._recog_label_enc))

        self.level = tmp.level
        self.unknown_gmm = tmp.unknown_gmm
        self.unknown_likelihood = tmp.unknown_likelihood

        if return_tmp:
            return tmp
