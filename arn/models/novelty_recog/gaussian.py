"""Novelty Recognition baseline using a Gaussian per class in feature space."""
from copy import deepcopy

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import BayesianGaussianMixture
import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

from exputils.data.labels import NominalDataEncoder

from arn.models.novelty_recog.predictor import OWHARecognizer
from arn.torch_utils import torch_dtype

import logging
logger = logging.getLogger(__name__)


def get_log_prob_mvn_thresh(mvn, min_error_tol):
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
    min_samples : int = None
        The minimum number of samples within a class cluster. Will raise an
        error if there are not enough samples within a given known classes
        based on labels in fit().

        Minimum number of samples for a cluster of outlier points to be
        considered a new class cluster.
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
    dtype : str = 'float64'
        The dtype to use for the MultivariateNormal calculations based on the
        class features. Sets each class_features per known class to this dtype
        prior to finding the torch.tensor.mean() or torch.tensor.cov().
    device : str = None
        The device on which the internal tensors are stored and calculations
        are performed. When None, default, it is inferred upon fitting.
    cov_epsilon : float = 1e-12
        The torch.tensor.cov() is numerically unstable and so a small value
        will need adde to the diagonal of the resulting covariance matrix to
        avoid being treated as not a positive semi-definite matrix.
    _gaussians : list = None
        List of `torch.distributions.multivariate_normal.MultivariateNormal`
        per known class, based on this recognizer's predictor's label encoder.
    _thresholds : list = None
        A list of floats that are the thresholds over the log probabilities of
        the Gaussians of each class.
    _recog_weights : np.array = None
        The recognized class weights as determined by the component weights
        from the Dirichlet Process Gaussian Mixture Model.
    see OWHARecognizer
    """
    def __init__(
        self,
        min_error_tol=5e-3,
        detect_error_tol=None,
        min_samples=None,
        min_density=None,
        dtype='float64',
        device=None,
        cov_epsilon=1e-12,
        **kwargs,
    ):
        """Initialize the recognizer.

        Args
        ----
        min_error_tol : see self
        detect_error_tol : see self
        min_samples : see self
        min_density : see self
        dtype : see self
        device : see self
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

        if min_samples:
            raise NotImplementedError('min_samples when not None.')
            if min_samples < 0 or min_samples > 1:
                raise ValueError(
                    'min_samples must be within inclusive range [0, 1].'
                )
        self.min_samples = min_samples

        if min_density:
            raise NotImplementedError('min_density when not None.')
            if min_density < 0 or min_density > 1:
                raise ValueError(
                    'min_density must be within inclusive range [0, 1].'
                )
        self.min_density = min_density

        self.dtype = torch_dtype(dtype)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = device

        self._recog_weights = None
        self._gaussians = None
        self._thresholds = None
        self.cov_epsilon = cov_epsilon

    def fit(self, dataset, val_dataset=None, task_id=None):
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
        super().fit(dataset, val_dataset, task_id)

        features = []
        labels = []
        for feature_tensor, label in dataset:
            features.append(feature_tensor)
            labels.append(label)
        del feature_tensor
        features = torch.stack(features)
        labels = torch.stack(labels).argmax(1)

        # TODO determine how to handle fit, fit_knowns, and fit_recog

        if self.device is None:
            device = features.device
            self.device = device
        else:
            device = self.device

        # Numerical stability adjustment for the sample covariance's diagonal
        stability_adjust = self.cov_epsilon * torch.eye(
            features.shape[-1],
            device=device
        )

        # Fit a Gaussian per known class, starts at 1 to ignore unknown class.
        self._gaussians = []
        thresholds = []
        for label in list(self.label_enc.inv)[self.label_enc.unknown_idx + 1:]:
            mask = labels == torch.tensor(label)
            if not mask.any():
                continue

            class_features = features[mask].to(device, self.dtype)

            if self.min_samples:
                if class_features.shape[0] < self.min_samples:
                    raise ValueError(
                        f"Label {label} features' samples < min_samples."
                    )

            try:
                mvn = MultivariateNormal(
                        class_features.mean(0),
                        class_features.T.cov(),
                    )
            except:
                mvn = MultivariateNormal(
                        class_features.mean(0),
                        class_features.T.cov() + stability_adjust,
                    )

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
            thresholds.append(get_log_prob_mvn_thresh(mvn, self.detect_error_tol))

            # TODO need to return to min threshold per class to correct if
            # known. Ideal would be to find the new alpha to then update all
            # other classes, but this is not a trivial inverse calculation.

            logger.debug(
                'Label %d: num samples = %d',
                label,
                len(class_features),
            )
            logger.debug(
                'Label %d: mean log_prob = %f',
                label,
                float(mvn.log_prob(class_features).mean().detach()),
            )
            logger.debug(
                'Label %d: min log_prob = %f',
                label,
                float(mvn.log_prob(class_features).min().detach()),
            )
            logger.debug(
                'Label %d: log_prob threshold = %f',
                label,
                float(thresholds[-1].detach()),
            )

        self._thresholds = thresholds

    def detect(self, features, preds, n_expected_classes=None):
        """Given data samples, detect novel samples.

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
        if self._gaussians is None:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')

        # Predict a class label per feature point
        new_preds = preds.argmax(1) #.to(self.device)
        for i, label in enumerate(
            list(self.label_enc.inv)[self.label_enc.unknown_idx + 1:]
        ):
            mask = new_preds == torch.tensor(label)
            logger.debug(
                'Label %d: Total given predicted = %d',
                label,
                mask.sum(),
            )
            if not mask.any():
                continue
            detected = (
                self._gaussians[i].log_prob(features[mask].to(self.dtype))
                    < self._thresholds[i]
            )
            logger.debug(
                'Label %d: with thresh %f total detected outliers: %d / %d',
                label,
                self._thresholds[i],
                detected.sum(),
                mask.sum(),
            )

            new_preds[torch.nonzero(mask)[detected]] = \
                self.label_enc.unknown_idx

        # TODO need to detect on unknown general
        #mask = new_preds == torch.tensor()
        #logger.debug(
        #    'Label %d: Total given predicted = %d',
        #    label,
        #    mask.sum(),
        #)
        #if mask.any():
        #    # TODO compare to all recognized not w/in predictor finetune

        # TODO need to detect on recognized classes! Unless predictor preds them
        #for i, label in enumerate(self.recog_label_enc):

        # Shouldn't this just be recognize and then threshold on the closeness
        # to some node of uncertainty?

        return new_preds

    def recognize(
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
        if self._gaussians is None:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')

        if isinstance(features, torch.Tensor):
            dtype = features.dtype
            torch_features = features
            features = features.detach().cpu().numpy()
        else:
            dtype = getattr(torch, str(features.dtype))
            torch_features = None

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
        dpgmm = BayesianGaussianMixture(**default_kwargs)
        dpgmm.fit(features)

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
        logger.info(
            "GaussianRecog's DPGMM found %d new unknown classes.",
            len(argsorted_weights),
        )
        if len(argsorted_weights) <= 1:
            # No recognized unknown classes.
            return
        if self.recog_label_enc is None:
            self.recog_label_enc = NominalDataEncoder()

        for arg in argsorted_weights:
            #if self.min_samples >= 2:
            #    # TODO Get class_features once!
            #    if class_features.shape[0] < self.min_samples:
            #        continue

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
            #    # TODO Get class_features once!
            #    if mvn.log_prob(class_features).mean() \
            #        < torch.log(torch.tensor(self.min_density)):
            #        continue

            self._gaussians.append(mvn)
            #if isinstance(self._thresholds, list):
            #    self._thresholds.append(mvn.log_prob(class_features).min())
            #elif mvn.log_prob(class_features).min() < self._thresholds:
            #    logger.warning(
            #        'mvn.log_prob(class_features).min() for label arg %d is '
            #        'less than the min threshold given %f',
            #        arg,
            #        self._thresholds,
            #    )

            # Update the label encoder with new recognized class-clusters
            self.recog_label_enc.append(f'unknown_{len(self.recog_label_enc)}')

        # Save the normalized belief of the new unknowns
        if self._recog_weights is None:
            self._recog_weights = argsorted_weights
        else:
            self._recog_weights = np.concatenate([
                self._recog_weights,
                argsorted_weights,
            ])

        # Update label_enc to include the recog_label_enc at the end.
        self.label_enc.append(self.recog_label_enc, ignore_dups=True)

        # Predict the prob vector for recognized classes per sample
        if torch_features is not None:
            features = torch_features

        # Normalize the probability each feature belongs to a recognized class,
        # st the recognized classes are mutually exclusive to one another.
        return F.softmax(torch.stack(
            [mvn.log_prob(features) for mvn in self._gaussians],
            dim=1,
        ), dim=1)
