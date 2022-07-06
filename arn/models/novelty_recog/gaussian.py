"""Novelty Recognition baseline using a Gaussian per class in feature space."""
from copy import deepcopy

import torch

import logging
logger = logging.getLogger(__name__)


class GaussianRecog(object):
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
    min_samples : int
        The minimum number of samples within a class cluster. Will raise an
        error if there are not enough samples within a given known classes
        based on labels in fit().

        Minimum number of samples for a cluster of outlier points to be
        considered a new class cluster.
    min_density : float
        The minimum normalized density (probability) for a cluster of unknowns
        to be considered a new class cluster. This is a value within [0, 1].
        Density is calculated as the mean of probabilities of points within a
        space.  Raises an error during fitting if the gaussians per known class
        are less than min_density.
    _start_idx : int = 1
        The index of the label encoder where the known labels begin.  Assumes
        that all indices greather than or equal to this value is a known label.
    _gaussians : list = None
        List of `torch.distributions.multivariate_normal.MultivariateNormal`
        per known class, based on this recognizer's predictor's label encoder.
    _thresholds : list = None
        A list of floats that are the thresholds over the log probabilities of
        the Gaussians of each class.
    _novel_class_count : int = 0
        Internal state of the novel unknown classes recognized by this
        recognizer.  Used to maintain different class labeles of format
        'unknown_#'.
    """
    def __init__(self, min_samples, min_density):
        """Initialize the recognizer.

        Args
        ----
        min_samples : see self
        min_density : see self
        """
        self.min_samples = min_samples
        if min_density < 0 or min_density > 1:
            raise ValueError(
                'min_density must be within inclusive range [0, 1].'
            )
        self.min_density = min_density

        self._start_idx = 1
        self._gaussians = None
        self._thresholds = None
        self._novel_class_count = 0

    @property
    def novel_class_count(self):
        return self._novel_class_count

    def fit(
        self,
        features,
        labels,
        label_enc,
        start_idx=1,
        val_features=None,
        val_labels=None,
    ):
        """Fits a Gaussian to each class within feature space and a finds a
        threshold over the logarithmi probability to each class' Gaussian that
        includes all knowns within and excludes all unknowns.

        Args
        ----
        features : torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims).
        labels : torch.Tensor
            1 dimensional integer tensor of shape (samples,). Contains the
            index encoding of each label per features.
        label_enc : exputils.data.labels.NominalDataEncoder
            The label encoder for the labels. Unknown is asusmed to be the
            first index and encoded as integer 0.
        start_idx : int = 1
            The index of the label encoder where the known labels begin.
            Assumes that all indices greather than or equal to this value is a
            known label.
        val_features : = None
            If provided, same as features. Used to improve the threshold per
            class.
        val_labels :  = None
            If provided, same as features. Used to improve the threshold per
            class.
        """
        # Fit a Gaussian per known class, starts at 1 to ignore unknown class.
        self._start_idx = start_idx
        self._gaussians = []
        self._thresholds = []
        for label in list(label_enc.inv)[start_idx:]:
            class_features = features[labels == torch.tensor(label)]

            if class_features.shape[0] < self.min_samples:
                raise ValueError(
                    f"Label {label} features' samples < min_samples."
                )

            mvn = torch.distributions.multivariate_normal.MultivariateNormal(
                    class_features.mean(0),
                    class_features.T.cov(),
                )

            if (
                mvn.log_prob(class_features).mean()
                < torch.log(torch.tensor(self.min_density))
            ):
                raise ValueError(
                    f"Label {label} features' normalized density "
                    '< min_density.'
                )
            self._gaussians.append(mvn)
            self._thresholds.append(mvn.log_prob(class_features).min())

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
                float(self._thresholds[-1].detach()),
            )

        # TODO use validation data to update the threshold further!

    def recognize(self, features, preds, label_enc):
        """Given unknown feature space points, find new class clusters given
        this recognizer's class cluster criteria. Are there enough samples
        within a specified hypersphere's volume (based on min_radius)?

        Args
        ----
        features : torch.Tensor
            2 dimensional float tensor of shape (samples, feature_repr_dims).
        preds : torch.Tensor
            1 dimensional integer tensor of shape (samples,). Contains the
            index encoding of each label predicted per feature.
        label_enc : exputils.data.labels.NominalDataEncoder
            The label encoder for the labels. Unknown is asusmed to be the
            first index and encoded as integer 0.

        Returns
        -------
        (torch.Tensor, exputils.data.labels.NominalDataEncoder)
            Recognized label per feature space point and the label_encoder copy
            with the appened new class clusters.
        """
        if self._gaussians is None:
            raise ValueError('Recognizer is not fit: self._gaussians is None.')
        label_enc = deepcopy(label_enc)

        # Defaults recognition to unknown assuming only given unknowns
        recogs = torch.full(
            preds.shape,
            label_enc.unknown_idx,
            dtype=preds.dtype,
        )

        # TODO Perform clustering checking if any clusters that fit criteria:
        #   1. samples >= self.min_samples
        #   2. mvn.log_prob(samples).mean() >= torch.log(self.min_density)
        #   May have to use logic from DBSCAN or similar density approaches for
        #   optimal finding of clusters.

        # TODO once a new class cluster is recognized, add its identifier to
        # the label encoder and add the new label to the sample's corresponding
        # indices in recogs
        new_recog = f'unknown_{self._novel_class_count}'
        label_enc.append(new_recog)

        # TODO increment self._novel_class_count by the number of new class
        # clusters recognized = f'unknown_{self._novel_class_count}'
        self._novel_class_count += 1

        return recogs, label_enc

    def predict(
        self,
        features,
        preds,
        label_enc,
        n_expected_classes=None,
        start_idx=None,
    ):
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
        label_enc : exputils.data.labels.NominalDataEncoder
            The label encoder for the labels. Unknown is asusmed to be the
            first index and encoded as integer 0.
        start_idx : int = None
            Overrides use of self._start_idx as set from fitting. The index of
            the label encoder where the known labels begin.  Assumes that all
            indices greather than or equal to this value is a known label.
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
        if start_idx is None:
            start_idx = self._start_idx

        # Predict a class label per feature point
        new_preds = preds.detach().clone()
        for i, label in enumerate(list(label_enc.inv)[start_idx:]):
            new_preds[
                self._gaussians[i].log_prob(
                    features[new_preds == torch.tensor(label)]
                ) < torch.tensor(self._thresholds[i])
            ] = label_enc.unknown_idx
        return new_preds

        # TODO outside of this, update pred label enc, and handle environment
        # label_enc != pred label_enc

        # TODO should mark in dataframe which samples are labeled by recognizer
        # rather than oracle label: new col in experience dataframe.
