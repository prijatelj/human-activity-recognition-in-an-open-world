"""Abstract Open World Human Activity Recognizer with some generics or default
properties and methods.
"""
from abc import abstractmethod
from copy import deepcopy
import os

import h5py
import numpy as np
import pandas as pd
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.models.predictor import OWHAPredictor
from arn.torch_utils import torch_dtype

import logging
logger = logging.getLogger(__name__)


def join_label_encs(left, right, use_right_key=True):
    label_enc = deepcopy(left)
    if use_right_key:
        if right.unknown_key is None:
            raise ValueError(
                'right unknown key is None when use_right_key is True'
            )
        key = right.unknown_key
    else:
        if left.unknown_key is None:
            raise ValueError(
                'left unknown key is None when use_right_key is False'
            )
        key = left.unknown_key
    if right.unknown_key is None:
        right = iter(right)
    else:
        right = iter(right)
        next(right)
    label_enc.append(list(right))
    label_enc.inv[0] = key
    return label_enc


def load_owhar(h5, class_type=None):
    """Load the class instance from the HDF5 file."""
    if class_type is None:
        class_type = OWHARecognizer
    close = isinstance(h5, str)
    if close:
        h5 = h5py.File(h5, 'r')

    attrs = dict(h5.attrs.items())
    increment = attrs.pop('increment', None)
    if increment:
        attrs['start_increment'] = increment

    # NOTE Does not load the fine_tune objects yet.
    loaded = class_type(fine_tune=None, **attrs)

    if h5['experience']:
        loaded.experience = pd.DataFrame(
            {
                'uid': pd.Series(
                    np.array(h5['experience']['uid']).astype(int),
                    dtype=int,
                ),
                'sample_path': pd.Series(
                    np.array(h5['experience']['sample_path'], dtype=str),
                    dtype=str,
                ),
                'labels': pd.Series(
                    np.array(h5['experience']['labels'], dtype=str),
                    dtype=str,
                ),
                'oracle': pd.Series(
                    np.array(h5['experience']['oracle'], dtype=bool),
                    dtype=bool,
                ),
            },
            index=loaded.experience['uid'],
        )

    if '_known_label_enc' in h5:
        loaded._known_label_enc = NominalDataEncoder.load_h5(
            h5['_known_label_enc']
        )
    if '_recog_label_enc' in h5:
        loaded._recog_label_enc = NominalDataEncoder.load_h5(
            h5['_recog_label_enc']
        )

    if close:
        h5.close()
    return loaded


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
    _known_label_enc : NominalDataEncoder = None
    _recog_label_enc : NominalDataEncoder = None
    _label_enc : NominalDataEncoder = None
    feedback_request_method : str = 'uncertain_first'
        The method used to request feedback. Defaults to 'uncertain_first',
        allows also 'random'.

        Note, current docstr==0.0.3rc1  splits words on space within str.
    min_samples : int = 2
        The minimum number of samples within a class cluster. Will raise an
        error if there are not enough samples within a given known classes
        based on labels in fit().

        Minimum number of samples for a cluster of outlier points to be
        considered a new class cluster.
    dtype : str = 'float64'
        The dtype to use for the MultivariateNormal calculations based on the
        class features. Sets each class_features per known class to this dtype
        prior to finding the torch.tensor.mean() or torch.tensor.cov().
    device : str = None
        The device on which the internal tensors are stored and calculations
        are performed. When None, default, it is inferred upon fitting.
    """
    def __init__(
        self,
        feedback_request_method: str = 'uncertain_first',
        min_samples: int = 2,
        dtype: str = 'float64',
        device: str = None,
        **kwargs,
    ):
        """Initialize the OWHARecognizer.

        Args
        ----
        see OWHAPredictor.__init__
        feedback_request_method : see self
        min_samples: see self
        dtype : see self
        device : see self
        """
        self.feedback_request_method = feedback_request_method
        if min_samples:
            if min_samples < 0:
                raise ValueError(
                    'min_samples must be greater than 0.'
                )
        self.min_samples = min_samples

        self.dtype = torch_dtype(dtype)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = device

        super().__init__(**kwargs)
        self.experience = pd.DataFrame({
            'uid': pd.Series(dtype=int),
            'sample_path': pd.Series(dtype=str),
            'labels': pd.Series(dtype=str),
            'oracle': pd.Series(dtype=bool),
        })
        self._recog_label_enc = None
        self._known_label_enc = None
        self._label_enc = None

    @property
    def n_recog_labels(self):
        """The number of labels in recog_label_enc."""
        return 0 if not self.has_recogs else len(self.recog_label_enc)

    @property
    def n_known_labels(self):
        """The number of labels in known_label_enc."""
        return 0 if self.known_label_enc is None else len(self.known_label_enc)

    @property
    def n_labels(self):
        """The number of labels in label_enc."""
        return 0 if self.label_enc is None else len(self.label_enc)

    @property
    def known_label_enc(self):
        """Interface and forces to set label enc w/o assignment `=`."""
        return self._known_label_enc

    @property
    def recog_label_enc(self):
        """Interface and forces to set label enc w/o assignment `=`."""
        return self._recog_label_enc

    @property
    def label_enc(self):
        """Interface and forces to set label enc w/o assignment `=`."""
        return self._label_enc

    @property
    def has_recogs(self):
        """Checks if there are any recognized labels."""
        return bool(self.recog_label_enc)

    def add_new_knowns(self, new_knowns):
        """Adds the given class labels as new knowns to the known label encoder
        """
        if self.known_label_enc is None:
            self._known_label_enc = NominalDataEncoder(
                new_knowns,
                unknown_key='unknown',
            )
        else:
            self._known_label_enc.append(new_knowns)

        self.update_label_enc(False)

    def update_label_enc(self, use_right_key=False):
        # Update the label encoder given new knowns
        if self.has_recogs:
            self._label_enc = join_label_encs(
                self._known_label_enc,
                self.recog_label_enc,
                use_right_key=use_right_key,
            )
        else:
            self._label_enc = deepcopy(self._known_label_enc)

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        self._recog_label_enc = None
        self._label_enc = deepcopy(self.known_label_enc)

    def get_unseen_mask(self, dataset_df):
        """Compares given DataFrame to experience to find unseen samples.
        Args
        ----
        dataset_df : pd.DataFrame

        Returns
        -------
        np.array
            Numpy array of bools along the rows of the given dataframe to mark
            samples as True when they were not experienced in the current
            experience, othewise False for they were experienced.
        """
        if len(self.experience) <= 0:
            unseen_mask = np.array([True] * len(dataset_df))
        else:
            unseen_mask = ~dataset_df['sample_index'].isin(
                self.experience['uid']
            ).values
        return unseen_mask

    def set_concat_exp(self, dataset_df, labels, oracle_feedback):
        """Updates the experience dataframe with the given data."""
        self.experience = self.experience.append(
            pd.DataFrame(
                {
                    'uid': dataset_df['sample_index'].astype(int),
                    'sample_path': dataset_df['sample_path'].astype(str),
                    'labels': pd.Series(labels, dtype=str),
                    'oracle': pd.Series(oracle_feedback, dtype=bool),
                },
                index=dataset_df['sample_index'],
            )
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

    def pre_predict(self, dataset, experience=None):
        """Update the experience and predictor state prior to predictions."""
        # If dataset contains uids unseen, add to predictor experience
        unseen_mask = self.get_unseen_mask(dataset.data)

        # NOTE DPGMM fitting and clustering takes too long! Use detect to cull.
        if any(unseen_mask):
            # If any unseen, perform detection. If detets, then recognize_fit()
            features = torch.stack([
                dataset[i] for i, val in enumerate(unseen_mask) if val
            ]).to(self.device)

            detects = self.detect(features)
            update_detects = detects.any() and detects.sum() >=self.min_samples
            if update_detects:
                # Only recognize_fit on those detected as unknown.
                features = features[detects]
                if experience:
                    # Get the experience features of orcale == False AND
                    # unknown predictions to be added to the detected here as
                    # all unknowns/unlabeled should be fit together
                    unlabeled = self.experience[~self.experience['oracle']]
                    unks = ['unknown']
                    if self.has_recogs:
                        unks += list(self.recog_label_enc)
                    unknowns = unlabeled[unlabeled['labels'].isin(unks)]

                    # Get the features for unknowns w/in given experience
                    unk_mask = experience.train.data['sample_index'].isin(
                        unknowns['uid'],
                    )

                    if np.any(unk_mask):
                        # Relies on sample_idx being the experience index.
                        assert all(
                            experience.train.data[unk_mask]['sample_index']
                            == experience.train.data[unk_mask].index
                        )

                        exp_features = torch.stack([
                            experience.train[i] for i
                            in np.arange(len(experience.train))[unk_mask]
                        ]).to(self.device)

                        # Append the experienced unknowns to detected unknowns
                        features = torch.cat([features, exp_features])
                        logger.debug(
                            'features shape post cat of experience: %s',
                            features.shape,
                        )
                self.recognize_fit(features)
        else:
            update_detects = False

        return unseen_mask, update_detects

    def post_predict(
        self,
        dataset,
        experience,
        preds,
        unseen_mask,
        update_detects,
    ):
        if update_detects:
            # Update internal experience with the change in unknowns clusters
            # from call to self.recognize_fit().
            # Align the features to their corresponding experience uid

            # if _not_ oracle in self.experience, then update label.
            not_oracle_mask = ~self.experience['oracle']
            seen_no_oracle = self.experience[not_oracle_mask]

            # Get dataset indices to use, get the pre-calced preds from above
            dset_idx = np.arange(len(dataset))[
                dataset.data['sample_index'].isin(seen_no_oracle['uid'])
            ]
            if len(dset_idx) > 0: # seen and not oracle: so update from dset
                dset_uids = dataset.data.iloc[dset_idx]['sample_index']
                exp_mask = self.experience[not_oracle_mask].isin(dset_uids)
                exp_loc = self.experience[exp_mask].index
                self.experience.loc[exp_loc, 'labels'] = self.label_enc.decode(
                    preds[dset_idx].argmax(1).detach().cpu().numpy()
                )
                # rm dset_uids from not_oracle_mask and seen_no_oracle
                not_oracle_mask &= ~exp_mask
                seen_no_oracle = self.experience[not_oracle_mask]

            # Update the labels of seen and not oracle and not in dset
            if any(not_oracle_mask):
                repred_mask = experience.train.data['sample_index'].isin(
                    seen_no_oracle['uid']
                )
                if repred_mask.any():
                    exp_repred = self.label_enc.decode(
                        self.recognize(
                            torch.stack([
                                experience.train[i] for i in
                                np.arange(len(experience.train))[
                                    repred_mask.values
                                ]
                            ]),
                            detect=True,
                        ).argmax(1).detach().cpu().numpy()
                    )

                    self.experience.loc[
                        repred_mask[repred_mask].index,
                        'labels',
                    ] = exp_repred

        if any(unseen_mask):
            # Add any unseen features in dataset to experience w/ predictions
            self.set_concat_exp(
                dataset.data[unseen_mask],
                self.label_enc.decode(
                    preds[unseen_mask].argmax(
                        1,
                    ).detach().cpu().numpy()
                ),
                [False] * sum(unseen_mask),
            )

        # TODO also, i suppose the predictor should not hold onto val/test
        # samples.... though it can. and currently as written does.
        #   Fix is to ... uh. just pass a flag if eval XOR fit
        # NOTE If experience is never given w/ all train, val, and test, then
        #   it only ever uses train, unless otherwise specified.  For example,
        #   in fit() it only uses the train data. Keeping that trend w/in
        #   predict as well for now.

    def predict(self, dataset, experience=None):
        if self.label_enc is None:
            raise ValueError('label enc is None. This predictor is not fit.')
        # NOTE Consider fitting DPGMM on extracts in future, instead of frepr
        #preds, features = super().predict_extract(dataset)

        # 1. preds, features = predict_extract(dataset) (or not if frepr)
        # 2. recognize_fit(dataset) if any _new_ experienced data
        # 3. recogs = recognize(dataset)
        # 4. optionally, weight average preds and recogs by some hyperparam
        #   This weighting could be found during fit which maximizes mutual
        #   info between the preds and ground truth. Perhaps set a min s.t.
        #   recogs is not entirely ignored? Hmm...
        #preds = super().predict(dataset)

        logger.info(
            "Begin call to %s's %s.pre_predict()",
            self.uid,
            type(self).__name__,
        )
        unseen_mask, update_detects = self.pre_predict(dataset, experience)
        logger.info(
            "Begin call to %s's %s.recognize()",
            self.uid,
            type(self).__name__,
        )
        preds = self.recognize(
            torch.stack(list(dataset)).to(self.device),
            detect=True,
        )
        logger.info(
            "Begin call to %s's %s.oost_predict()",
            self.uid,
            type(self).__name__,
        )
        self.post_predict(
            dataset,
            experience,
            preds,
            unseen_mask,
            update_detects,
        )
        logger.info(
            "End call to %s's %s.predict()",
            self.uid,
            type(self).__name__,
        )
        return preds

    def pre_fit(self, dataset):
        """Manage experience with new data and find masks for new and old data
        """
        # Mask experience which oracle feedback was given (assumes experiment
        # is maintaining the predictor's experience)
        # Given dataset, mark any samples in experience as oracle & label
        dset_feedback_mask = ~pd.isna(dataset.labels)

        if len(self.experience) > 0:
            exp_feedback_mask = self.experience['uid'].isin(
                dataset.data[dset_feedback_mask]['sample_index']
            )
            exp_no_feedback_mask = \
                self.experience[~exp_feedback_mask]['uid'].isin(
                    dataset.data[~dset_feedback_mask]['sample_index']
                )
            exp_mask = exp_feedback_mask.copy()
            exp_mask[~exp_feedback_mask] |= exp_no_feedback_mask
            del exp_no_feedback_mask

            # Update to True only for those whose label_col is not None
            self.experience.loc[exp_feedback_mask, 'oracle'] = True

            # Assign the feedback labels from dataset to experience.
            self.experience.loc[exp_feedback_mask, 'labels'] = \
                dataset.labels[dset_feedback_mask].loc[
                    self.experience[exp_feedback_mask]['uid']
                ].astype(str)

            # NOTE For each uniquely removed labeled recog sample, check if
            # that recognized cluster still has enough samples, if yes keep
            # else set to all unknown. Rm recog labels set to unknowns.
            if self.has_recogs:
                assert self.recog_label_enc.unknown_idx == 0
                distinct_recogs = iter(self.recog_label_enc)
                next(distinct_recogs)
                for exp_label in distinct_recogs:
                    mask = self.experience['labels'] == exp_label
                    if np.sum(mask) < self.min_samples:
                        self.experience.loc[mask, 'labels'] = \
                            self.label_enc.unknown_key
                        idx = self.recog_label_enc.pop(exp_label)

        # Update the predictor's label encoder with new knowns
        unique_dset_feedback = dataset.labels[dset_feedback_mask].unique()
        new_knowns = []
        for new_known in np.array(dataset.label_enc):
            if (
                new_known in unique_dset_feedback
                and (
                    self.known_label_enc is None
                    or new_known not in self.known_label_enc
                )
            ):
                new_knowns.append(new_known)

        # TODO Ensure OrderedConfusionMAtrices and CMs handle label alignment
        # Ensure that all decoding of predictors' encodings uses predictor's
        # label_enc, and the data uses the datasets label_enc, as written right
        # now, they will have label misalignment with partial feedback.

        self.add_new_knowns(new_knowns)

        # Add new experience data
        unseen_mask = self.get_unseen_mask(dataset.data)
        if any(unseen_mask):
            # Add any unseen features in dataset to experience w/ feedback
            self.set_concat_exp(
                dataset.data[unseen_mask],
                dataset.labels[unseen_mask],
                dset_feedback_mask[unseen_mask],
            )

        features = []
        #labels = []
        # NOTE This doesn't check for oracle feedback or not, won't handle None
        #   within this code itself.
        for feature_tensor, label in dataset:
            features.append(feature_tensor)
            #labels.append(label)
        del feature_tensor, label
        features = torch.stack(features)
        labels = torch.tensor(
            self.known_label_enc.encode(dataset.labels.astype(str))
        )

        return dset_feedback_mask, features, labels

    def post_fit(self, dset_feedback_mask, features):
        """Any experience or recognizer state management necessary after
        fitting the knowns.
        """
        dset_no_feedback_mask = ~dset_feedback_mask

        # Update the recog distirbs given changes to knowns for the recogs
        if len(self.experience) > 0 and np.any(dset_no_feedback_mask):
            logger.debug(
                'Checking if to recognize in fit: '
                'len(self.experience) = %d; len(self.recog_label_enc) = %d',
                len(self.experience),
                self.n_recog_labels,
            )

            # Need to get the given data mapped to internal experience w/o
            # oracle feedback. This gets recognized over and labels saved to
            # exp.
            recogs = self.recognize(
                features[dset_no_feedback_mask.values],
                detect=True,
            ).argmax(1).detach().cpu().numpy()

            # NOTE this was using knonw_label_enc.decode(recogs), but errored
            # due to recognize using all  knowns and recognized unknowns. If
            # this is to actually use only knowns in recognize, then it needs
            # changed to do so.
            self.experience.loc[
                dset_no_feedback_mask[dset_no_feedback_mask].index,
                'labels',
            ] = self.label_enc.decode(recogs)

            detects = recogs == self.known_label_enc.unknown_idx
            detects_mask = dset_no_feedback_mask.copy()
            detects_mask[detects_mask] = detects

            # Update all unknowns, refitting the DPGMM entirely.
            # Must check if any points deemed outside of knowns given threshs
            if detects.any() and detects.sum() >= self.min_samples:
                self.recognize_fit(features[detects_mask.values])

                # Update experience with detects as unknown otherwise w/
                # new recognized labels. Update to experience occurs in
                # parent.fit(), but detects will NEVER be set, only recognized
                # unknowns.  Depending on desired functionality this may be
                # fine, but if you expect to be recording outliers as unknown
                # then this needs changed in parent fit or somehow informed by
                # detects here.

                # The non-oracle experience should be recalculated given
                # changes to both known and unknown.

                if self.has_recogs:
                    # TODO experience does not have index == uid
                    self.experience.loc[
                        dset_no_feedback_mask[dset_no_feedback_mask].index,
                        'labels',
                    ] = self.label_enc.decode(self.recognize(
                        features[dset_no_feedback_mask.values],
                        detect=True,
                    ).argmax(1).detach().cpu().numpy())
                else:
                    logger.debug('No recognize fit in fit(). prior label enc.')
                    # TODO HANLDE ALL OF THIS given children... and GMMs
                    self.reset_recogs()
            else:
                logger.debug('No recognize fit in fit(). Not enough detects')
                self.reset_recogs()
        else:
            logger.debug('No recognize fit in fit(). No experience w/o oracle')
            self.reset_recogs()

        # If self.save_dir, save the state of this recognizer
        if self.save_dir:
            # TODO consider adding number of predict calls per fit increment.
            self.save(os.path.join(
                self.save_dir,
                f'recog_chkpt-{type(self).__name__}',
                f'{self.uid}-{self.increment}.h5',
            ))

    @abstractmethod
    def fit_knowns(self, dataset, val_dataset=None):
        """Inheritting classes must handle experience maintence."""
        if self.skip_fit >= 0 and self._increment >= self.skip_fit:
            return
        # NOTE This is an unideal hotfix, the predictor should not effect
        # evaluator data, but in this case we need to use recogs as labels, so
        # those recogs need added if missing to the provided experience
        # (dataset).

        # NOTE this fit() assumes that dataset is ALL prior experience.
        #self._increment += 1

        # Ensure the new recogs are used in fitting.
        #   Predictor experience includes the unlabeled recogs, temporairly
        #   append it and swap the label encoders.
        original_data_len = len(dataset.data)
        original_label_enc = dataset.label_enc

        # NOTE need to check what is unseen before adding missing experience!
        #   If dataset includes unseen, experience is assumed updated by
        #   calling function / method.

        # Only add experience not in dataset.data
        if len(self.experience) > 0:
            # TODO probably perform check that the UID is not in exp[uid]
            dataset.data = dataset.data.append(
                self.experience[
                    ~self.experience['oracle'].astype(bool))
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
        #   And the gaussian recog is over frepr for now.

    def fit(self, dataset, val_dataset=None):
        """The interface for partial supervised fitting the recognizer.  This
        fit method exemplifies the calling of pre_fit, fit_knowns, and post_fit
        to show the intended pattern of functionality: do anything prior to
        fitting the knowns such as updating this predictor's experience, then
        fit the knowns, and then anything afterwards, such as updating
        experience now that the knowns have been fit.

        These methods are separated for the convenience of reusing the methods
        when writing custom fit_knowns() or overriding any other of these
        methods in children classes.
        """
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
        self.fit_knowns(dataset, val_dataset=val_dataset)
        logger.info(
            "Begin call to %s's %s.post_fit()",
            self.uid,
            type(self).__name__,
        )
        self.post_fit(dset_feedback_mask, features)
        logger.info(
            "End call to %s's %s.fit()",
            self.uid,
            type(self).__name__,
        )
        self._increment += 1

    def pre_recognize_fit(self):
        """Must update experience everytime and handle prior unknowns if any"""
        unks = ['unknown']
        if self.recog_label_enc:
            unks += list(self.recog_label_enc)
        unlabeled = self.experience[~self.experience['oracle']]
        unknowns = unlabeled['labels'].isin(unks)
        if unknowns.any():
            self.experience.loc[unknowns.index, 'labels'] = \
                self.known_label_enc.unknown_key

    @abstractmethod
    def recognize_fit(self, features):
        raise NotImplementedError('Inheriting class overrides this.')

    @abstractmethod
    def recognize(self, features, *args, **kwargs):
        raise NotImplementedError('Inheriting class overrides this.')

    @abstractmethod
    def detect(self, features, **kwargs):
        """The inference task of binary classification if the sample belongs to
        a known class or unknown class. This may be a post process step to
        predict().
        """
        raise NotImplementedError('Inheriting class overrides this.')

    def save(self, h5, overwrite=False, save_fine_tune=False):
        """Save as an HDF5 file."""
        if save_fine_tune:
            raise NotImplementedError(
                f'{type(self).__name__}.save(save_fine_tune=True)'
            )
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
            min_samples=self.min_samples,
            dtype=str(self.dtype)[6:],
            device=self.device.type,
            feedback_request_method=self.feedback_request_method,
        )
        for key, val in state.items():
            if val is None:
                continue
            h5.attrs[key] = val

        if len(self.experience) > 0:
            h5_exp = h5.create_group('experience')
            h5_exp['uid'] = self.experience['uid'].values.astype(int)
            h5_exp['sample_path'] = \
                self.experience['sample_path'].astype(np.string_)
            h5_exp['labels'] = self.experience['labels'].astype(np.string_)
            h5_exp['oracle'] = self.experience['oracle'].values.astype(bool)

        if self._known_label_enc is not None:
            self._known_label_enc.save(h5.create_group('_known_label_enc'))
        if self._recog_label_enc is not None:
            self._recog_label_enc.save(h5.create_group('_recog_label_enc'))

        if close:
            h5.close()

    @staticmethod
    def load(h5):
        return load_owhar(h5, OWHARecognizer)

    def load_state(self, h5, return_tmp=False, overwrite_uid=False):
        """Update state inplace by extracting it from the loaded predictor."""
        # TODO this won't work with inheritance calls. Need to chain
        tmp = type(self).load(h5)

        #self.fine_tune = tmp.fine_tune
        if overwrite_uid:
            self.uid = tmp.uid
        self.skip_fit = tmp.skip_fit
        self.save_dir = tmp.save_dir
        self._increment = tmp._increment
        self.min_samples = tmp.min_samples
        self.dtype = tmp.dtype
        self.device = tmp.device
        self.feedback_request_method = tmp.feedback_request_method

        self.experience = tmp.experience

        self._known_label_enc = tmp._known_label_enc
        self._recog_label_enc = tmp._recog_label_enc
        if self._known_label_enc is None:
            self._label_enc = None
        else:
            self.update_label_enc()

        if return_tmp:
            return tmp
