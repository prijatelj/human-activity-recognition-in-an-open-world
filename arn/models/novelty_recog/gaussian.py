"""Novelty Recognition baseline using a Gaussian per class in feature space."""
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

from arn.models.owhar import OWHAPredictor
from arn.torch_utils import torch_dtype

import logging
logger = logging.getLogger(__name__)


class OWHARecognizer(OWHAPredictor):
    """Abstract predictor with recognition functionality.

    Attributes
    ----------
    experience : pandas.DataFrame = None
        An ordered pandas DataFrame of with the index corresponding to the
        sample unique identifiers (integer based on experiment) and a column
        for the task label. Another column for the belief as a probability
        (certainty) in that label.

        `experience.columns == ['uid', 'sample_path', 'labels', 'oracle']`,
        where 'uid' is the Unique Identifier of the sample as the integer given
        to it by the experiment upon recieving the sample. 'labels' is the
        predictor's knowledge of the discrete label given to that sample,
        whether provided by the oracle or inferred. 'oracle' Boolean column
        indicates if it is given feedback that means this is the accepted known
        label for that sample (True), or if it is from the predictor's
        inference (False).

        'sample_path' is the filepath to load the sample.

        'label_certainty' is True
        if the label is provided from the oracle, otherwise is a float value
        between [0, 1] indicating the recognition probability for that class
        versus all other known classes.
    recog_label_enc : NominalDataEncoder = None
    label_enc : NominalDataEncoder
    store_all : bool = True
        When True, the default, store all feature points encountered in order
        they were encountered, regardless of their data split being train,
        validate, or test. If False (TODO), only store the training points.
    feedback_request_method : str = 'uncertain_first'
        The method used to request feedback. Defaults to 'uncertain_first',
        allows also 'random'.

        Note, current docstr==0.0.3rc1  splits words on space within str.
    """
    def __init__(self, **kwargs):
        """Initialize the OWHARecognizer.

        Args
        ----
        see OWHAPredictor.__init__
        feedback_request_method : see self
        """
        self.feedback_request_method = kwargs.pop(
            'feedback_request_method',
            'uncertain_first',
        )
        super().__init__(**kwargs)
        self.experience = pd.DataFrame(
            [],
            columns=['uid', 'sample_path', 'labels', 'oracle'],
        ).convert_dtypes([int, str, str, bool])
        self.recog_label_enc = None
        self.label_enc = None

    @property
    def n_recog_labels(self):
        return 0 if self.recog_label_enc is None else len(self.recog_label_enc)

    @property
    def n_known_labels(self):
        all_labels = 0 if self.label_enc is None else len(self.label_enc)
        return  all_labels - self.n_recog_labels

    def fit(self, dataset, val_dataset=None):
        """Inheritting classes must handle experience maintence."""
        # NOTE This is an unideal hotfix, the predictor should not effect
        # evaluator data, but in this case we need to use recogs as labels, so
        # those recogs need added if missing to the provided experience
        # (dataset).

        self._increment += 1

        # Ensure the new recogs are used in fitting.
        #   Predictor experience is the unlabeld recogs, temporairly append it
        #   and swap the label encoders.
        original_data_len = len(dataset.data)
        original_label_enc = dataset.label_enc

        # Only add experience not in dataset.data
        if len(self.experience) > 0:
            dataset.data = dataset.data.append(
                self.experience[
                    ~self.experience['oracle'].convert_dtypes(bool)
                ]
            )
            dataset.label_enc = self.label_enc

        # NOTE need todo val_datset management... if ever using it
        super().fit(dataset, val_dataset=None)

        # Now rm the experience from the dataframe and restore the label enc
        if len(self.experience) > 0:
            dataset.data = dataset.data.iloc[:original_data_len]
            dataset.label_enc = original_label_enc

        # NOTE if there are any unlabeled samples, then perform recognition.
        #   Currently not included in project's experiments.

        # Add new experience data
        if len(self.experience) <= 0:
            unseen_mask = [True] * len(dataset.data)
        else:
            unseen_mask = ~dataset.data['sample_index'].isin(
                self.experience['uid']
            ).values

        # TODO NOTE! you may run into the issue where val / test indices as
        # uid's are encountered and are not unique to the train sample indices.
        # Need to test this!

        if any(unseen_mask):
            self.experience = self.experience.append(
                pd.DataFrame(
                    np.stack(
                        [
                            dataset.data['sample_index'][unseen_mask],
                            dataset.data[unseen_mask]['sample_path'],
                            dataset.data[unseen_mask][dataset.label_col],
                            [True] * len(unseen_mask),
                        ],
                        axis=1,
                    ),
                    columns=self.experience.columns,
                ).convert_dtypes([int, str, str, bool])
            )

    def feedback_request(self, features, available_uids, amount=1.0):
        """The predictor's method of requesting feedback.
        Args
        ----
        features : torch.Tensor
        available_uids : list
            List of uids, integers unique to the smaples for this Kinetics
            OWL experiment.
        amount : float = 1.0
            The decimal amount of feedback to be expected to be given of the
            new data for this increment.
        """
        if self.label_enc is None:
            raise ValueError('label enc is None. This predictor is not fit.')
        if available_uids is None:
            raise ValueError('Must be given available_uids')
            available_uids = self.experience[
                self.experience['oracle']
            ]['uid'].values

        if self.feedback_request_method == 'random':
            return super().feedback_request(available_uids)

        # Recognize w/ all knowns and unknown hierarchical
        recogs = self.recognize(features)

        if self.feedback_request_method == 'uncertain_first':
            # Get latest uncertainty scores for recog samples, sort by
            # descending uncertainty overall, regardless of most likely class
            maxes = recogs.max(1)
            maxes_ascend = torch.sort(maxes.values)
            return available_uids[maxes_ascend.indices.detach().cpu().numpy()]

        if self.feedback_request_method == 'uncertain_known_certain_unknown':
            # TODO prioritize most uncertain knowns and most certain knowns
            raise NotImplementedError()
            maxes = recogs.max(1)
            # TODO Determine if argmax is known or unknown
            # TODO if recognized as known, most uncertain first
            # TODO if recognized as unknown, most certain first
            return

        if self.feedback_request_method == 'certain_unknown_uncertain_known':
            raise NotImplementedError()

        raise ValueError(
            'Unexpected feedback request method: '
            f'{self.feedback_request_method}'
        )

    def predict(self, dataset):
        if self.label_enc is None:
            raise ValueError('label enc is None. This predictor is not fit.')
        # TODO Consider fitting DPGMM on extracts in future, instead of frepr
        #preds, features = super().predict_extract(dataset)


        # 1. preds, features = predict_extract(dataset) (or not if frepr)
        # 2. recognize_fit(dataset) if any _new_ experienced data
        # 3. recogs = recognize(dataset)
        # 4. optionally, weight average preds and recogs by some hyperparam
        #   This weighting could be found during fit which maximizes mutual
        #   info between the preds and ground truth. Perhaps set a min s.t.
        #   recogs is not entirely ignored? Hmm...
        #preds = super().predict(dataset)

        # If dataset contains uids unseen, add to predictor experience
        #   TODO this breaks when predictor experience has past seen samples
        #   now with oracle feedback removed.
        if len(self.experience) <= 0:
            unseen_mask = [True] * len(dataset.data)
        else:
            unseen_mask = ~dataset.data['sample_index'].isin(
                self.experience['uid']
            ).values

        # TODO NOTE! you may run into the issue where val / test indices as
        # uid's are encountered and are not unique to the train sample indices.
        # Need to test this!

        # NOTE DPGMM fitting and clustering takes too long! Use detect to cull.
        if any(unseen_mask):
            features = torch.stack([
                dataset[i] for i, val in enumerate(unseen_mask) if val
            ]).to(self.device)
            detects = self.detect(features)
            if detects.any() and detects.sum() >= self.min_samples:
                self.recognize_fit(features[detects])

        preds = F.pad(
            self.recognize(torch.stack(list(dataset)).to(self.device)),
            (1, 0),
            'constant',
            0,
        )

        if any(unseen_mask):
            self.experience = self.experience.append(
                pd.DataFrame(
                    np.stack(
                        [
                            dataset.data['sample_index'][unseen_mask],
                            dataset.data[unseen_mask]['sample_path'],
                            self.label_enc.decode(
                                preds[unseen_mask].argmax(
                                    1,
                                ).detach().cpu().numpy()
                            ),
                            [False] * len(unseen_mask),
                        ],
                        axis=1,
                    ),
                    columns=self.experience.columns,
                ).convert_dtypes([int, str, str, bool])
            )

        # TODO also, i suppose the predictor should not hold onto val/test
        # samples.... though it can. and currently as written does.
        #   Fix is to ... uh. just pass a flag if eval XOR fit

        return preds


        detected_novelty = self.detect(features, preds=preds) \
            == self.label_enc.unknown_idx

        # If new recogs were added before a call to fit, which adds them into
        # the predictor's output space, then append zeros to that space.
        n_recogs_in_preds = preds.shape[-1] - self.n_known_labels
        n_recogs = self.n_recog_labels
        if n_recogs_in_preds < n_recogs:
            preds = F.pad(
                preds,
                (0, n_recogs - n_recogs_in_preds),
                'constant',
                0,
            )

        total_detected = detected_novelty.sum()
        logger.debug('Detected %d novel/unknown samples.', total_detected)

        if detected_novelty.any() and total_detected > self.min_samples:
            recogs = self.recognize(features[detected_novelty])

            # Add new recognized class-clusters to output vector dim
            if recogs is not None:
                # Add general unknown class placeholder as zero
                recogs = F.pad(recogs, (1, 0), 'constant', 0)

                # Increase all of preds' vector sizes, filling with zeros
                preds = F.pad(
                    preds,
                    (0, recogs.shape[-1] - preds.shape[-1]),
                    'constant',
                    0,
                )

                # Fill in the detected novelty with their respective prob vecs
                # and weigh recogs by the prob of unknown, and set unknown to
                # zero
                preds[detected_novelty] = recogs.to(preds.device, preds.dtype)

        return preds

    @abstractmethod
    def recognize_fit(self, features):
        raise NotImplementedError('Inheriting class overrides this.')

    @abstractmethod
    def recognize(self, features):
        raise NotImplementedError('Inheriting class overrides this.')

    @abstractmethod
    def detect(self, features, **kwargs):
        """The inference task of binary classification if the sample belongs to
        a known class or unknown class. This may be a post process step to
        predict().
        """
        raise NotImplementedError('Inheriting class overrides this.')


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
    min_samples : int = 2
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
        min_samples=2,
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
            if min_samples < 0:
                raise ValueError(
                    'min_samples must be greater than 0.'
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

        # Combined known and unknown recognized variables
        self._gaussians = []
        self._thresholds = []

        # Unknown recognized variables
        self._recog_counter = 0
        self._recog_weights = None

        self.cov_epsilon = cov_epsilon
        self.min_cov_mag = 1.0

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
        # Mask experience which oracle feedback was given (assumes experiment
        # is maintaining the predictor's experience)
        if len(self.experience) > 0:
            # Given dataset, rm any samples in
            mask = self.experience['uid'].isin(dataset.data['sample_index'])
            self.experience.loc[mask, 'oracle'] = True
            exp_unique_labels = set(self.experience[mask]['labels'].unique())

            self.experience.loc[mask, 'labels'] = \
                dataset.data[dataset.label_col].loc[
                self.experience['uid'][mask]
            ].convert_dtypes(str)

            # NOTE For each uniquely removed labeled recog sample, check if
            # that recognized cluster still has enough samples, if yes keep
            # else set to all unknown. Rm recog labels set to unknowns.
            if self.recog_label_enc:
                # The the recog labels changed:
                changed_recog_labels = set(self.recog_label_enc) #\
                    #& exp_unique_labels
                for exp_label in changed_recog_labels:
                    mask = self.experience['labels'] == exp_label
                    # TODO sum(mask) only errored out at end, so it was NEVER
                    # called in the prior 19 steps for sims???? That has to be
                    # part of the recog label amount discrepency error.
                    if np.sum(mask) < self.min_samples:
                        self.experience.loc[mask, 'labels'] = \
                            self.label_enc.unknown_key
                        idx = self.recog_label_enc.pop(exp_label)
                        if isinstance(self._recog_weights, list):
                            del self._recog_weights[idx]

        # Update the predictor's label encoder with new knowns
        self.label_enc = deepcopy(dataset.label_enc)
        if self.recog_label_enc:
            # Keeping the recognized labels at the end.
            self.label_enc.append(self.recog_label_enc)

        # NOTE decide here if this is for fitting on frepr or the ANN.:
        #   Staying with fitting in frepr for now.

        # Fit the Gaussians and thresholds per class-cluster. in F.Repr. space
        features = []
        labels = []
        for feature_tensor, label in dataset:
            features.append(feature_tensor)
            labels.append(label)
        del feature_tensor
        features = torch.stack(features)
        labels = torch.stack(labels).argmax(1)

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
        # TODO this causses an issue by forgetting the recognized unknowns
        # TODO This could made efficient by skipping knowns w/o any data changes
        self._gaussians = []
        thresholds = []
        for label in list(self.label_enc.inv)[self.label_enc.unknown_idx + 1:]:
            mask = labels == torch.tensor(label)
            if not mask.any():
                logger.warning(
                    '%s has known class %s (idx %d) with no samples in fit()',
                    type(self).__name__,
                    self.label_enc.inv[label],
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
                        self.label_enc.inv[label]
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
            err_lprob = get_log_prob_mvn_thresh(mvn, self.detect_error_tol)
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
                self.label_enc.inv[label],
                len(class_features),
            )
            logger.debug(
                'Label %d: "%s" mean log_prob = %f',
                label,
                self.label_enc.inv[label],
                float(mvn.log_prob(class_features).mean().detach()),
            )
            logger.debug(
                'Label %d: "%s" min log_prob = %f',
                label,
                self.label_enc.inv[label],
                min_log_prob,
            )
            logger.debug(
                'Label %d: "%s" log_prob threshold = %f',
                label,
                self.label_enc.inv[label],
                float(thresholds[-1].detach()),
                #    if isinstance(thresholds[-1], torch.Tensor)
                #    else float(thresholds[-1]),
            )
        self._thresholds = thresholds

        self.min_cov_mag = np.stack([
            mvn.covariance_matrix.sum() / mvn.loc.shape[-1]
            for mvn in self._gaussians
        ]).min()

        # Update the recog distirbs given changes to knowns for the recogs
        if len(self.experience) > 0 and self.recog_label_enc:
            logger.debug(
                'Checking if to recognize in fit: '
                'len(self.experience) = %d; len(self.recog_label_enc) = %d',
                len(self.experience),
                len(self.recog_label_enc),
            )

            # Update all unknowns, refitting the DPGMM entirely.
            # Must check if any points deemed outside of knowns given threshs
            detects = self.detect(features)
            if detects.any() and detects.sum() >= self.min_samples:
                self.recognize_fit(features[detects])
        else:
            logger.debug('No recognize fit in fit()')


        # If self.save_dir, save the state of this recognizer
        if self.save_dir:
            # TODO consider adding number of predict calls per fit increment.
            self.save(os.path.join(
                self.save_dir,
                f'recog_chkpt-{type(self).__name__}',
                f'{self.uid}-{self.increment}.h5',
            ))

        # TODO Should fit on the soft labels (output of recognize) for
        # unlabeled data. That way some semblence of info from the recog goes
        # into the predictor.
        #   Can consider fitting on known labels with a weighting that strongly
        #   prioritizes the ground truth, but sets the other class values to be
        #   proportional, albeit scaled down, to the recognize output.

        # Fit the FineTune ANN if it exists now that the labels are determined.
        super().fit(dataset, val_dataset)

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        raise NotImplementedError('Inheritting class implements.')

    # Consider: def recognize_fit_mvns(self, argsorted_weights, recog_labels):
    #   For generic fitting of mvns from feature data w/ cluster mask

    def recognize(self, features):
        """Using the existing Gaussians per class-cluster, get log probs."""
        # Normalize the probability each feature belongs to a recognized class,
        # st the recognized classes are mutually exclusive to one another.
        return F.softmax(torch.stack(
            [mvn.log_prob(features) for mvn in self._gaussians],
            dim=1,
        ), dim=1)

    def detect(self, features, n_expected_classes=None):
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



        # NOTE Checks if the feature points are deemed to belong to ANY known
        detects = torch.tensor([True] * len(features)).to(features.device)
        for i in range(self.n_known_labels - 1):
            detects &= self._gaussians[i].log_prob(features) \
                < self._thresholds[i]

        return detects

    def save(self, h5, save_fine_tune=False, overwrite=False):
        """Save as an HDF5 file."""
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        state = dict(
            # OWHAPredictor
            fine_tune=self.fine_tune if save_fine_tune else None,
            uid=self.uid,
            skip_fit=self.skip_fit,
            save_dir=self.save_dir,
            increment=self.increment,
            # GaussianRecognizer
            min_error_tol=self.min_error_tol,
            detect_error_tol=self.detect_error_tol,
            min_samples=self.min_samples,
            min_density=self.min_density,
            cov_epsilon=self.cov_epsilon,
            dtype=str(self.dtype)[6:],
            device=self.device.type,
            _recog_counter=self._recog_counter,
        )
        for key, val in state.items():
            if val is None:
                continue
            h5.attrs[key] = val

        h5['label_enc'] = np.array(self.label_enc).astype(np.string_)
        h5['recog_label_enc'] = np.array(
            self.recog_label_enc
        ).astype(np.string_)
        if self._recog_weights is not None:
            h5['_recog_weights'] = self._recog_weights

        if len(self.experience) > 0:
            h5_exp = h5.create_group('experience')
            h5_exp['uid'] = self.experience['uid'].values.astype(int)
            h5_exp['sample_path'] = \
                self.experience['sample_path'].astype(np.string_)
            h5_exp['labels'] = self.experience['labels'].astype(np.string_)
            h5_exp['oracle'] = self.experience['oracle'].values.astype(bool)

        locs = []
        cov_mats = []
        thresholds = []
        for i, gauss in enumerate(self._gaussians):
            locs.append(gauss.loc)
            cov_mats.append(gauss.covariance_matrix)
            thresholds.append(float(self._thresholds[i]))
        locs = torch.stack(locs).detach().cpu()
        cov_mats = torch.stack(cov_mats).detach().cpu()

        h5['_thresholds'] = thresholds

        h5_gg = h5.create_group('_gaussians')
        h5_gg['locs'] = locs
        h5_gg['cov_mats'] = cov_mats

        if close:
            h5.close()

    @staticmethod
    def load(h5):
        """Load the HDF5 file."""
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(h5, 'r')

        attrs = dict(h5.attrs.items())
        increment = attrs.pop('increment', None)
        if increment:
            attrs['start_increment'] = increment
        _recog_counter = attrs.pop('_recog_counter', None)
        loaded = GaussianRecognizer(fine_tune=None, **attrs)
        loaded._recog_counter = _recog_counter

        loaded.label_enc = NominalDataEncoder(
            np.array(h5['label_enc'], dtype=str),
            unknown_idx=0,
        )

        loaded.recog_label_enc = np.array(
            h5['recog_label_enc'],
            dtype=str,
        ).reshape(-1)
        if (
            len(loaded.recog_label_enc) == 1
            and loaded.recog_label_enc == 'None'
        ):
            loaded.recog_label_enc = None
        else:
            loaded.recog_label_enc = NominalDataEncoder(
                loaded.recog_label_enc
            )

        loaded._recog_weights = h5.get('_recog_weights', None)
        if loaded._recog_weights:
            loaded._recog_weights = np.array(loaded._recog_weights)

        loaded._thresholds = [
            torch.tensor(t).to(loaded.device, loaded.dtype)
            for t in h5['_thresholds']
        ]

        if h5['experience']:
            loaded.experience = pd.DataFrame(
                np.stack(
                    [
                        np.array(h5['experience']['uid']).astype(int),
                        np.array(h5['experience']['sample_path'], dtype=str),
                        np.array(h5['experience']['labels'], dtype=str),
                        np.array(h5['experience']['oracle'], dtype=bool),
                    ],
                    axis=1,
                ),
                columns=['uid', 'sample_path', 'labels', 'oracle'],
            ).convert_dtypes([int, str, str, bool])

        loaded._gaussians = [
            MultivariateNormal(torch.tensor(
                    h5['_gaussians']['locs'][i]
                ).to(loaded.device, loaded.dtype),
                torch.tensor(cov_mat).to(loaded.device, loaded.dtype),
            )
            for i, cov_mat in enumerate(h5['_gaussians']['cov_mats'])
        ]

        if close:
            h5.close()
        return loaded
