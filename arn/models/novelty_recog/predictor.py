"""The generic/abstract classes of the open world predictor with Novelty
Recognition.
"""
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
F = torch.nn.functional

from arn.models.predictor import OWHAPredictor

import logging
logger = logging.getLogger(__name__)


class Predictor(object):
    """The base predictor class that contains and manages the predictor parts.

    Attributes
    ----------
    _uid : str = None
        An optional str unique identifier for this predictor. When not
        given, the uid property of this class' object is the trainer
        tensorboard log_dir version number.
    skip_fit : int = -1
        If >= 0, skips fitting the model starting on the increment that matches
        skip_fit's value. Thus, skip_fit == 0 skips all fitting, skip_fit == 1
        skips all fitting after the first increment/starting data, 0th index.
        skip_fit serves as an exclusive upperbound to the range of incremental
        fitting allowed for this predictor. Values less than zero does not
        result in any skipping of calls to fit().
    save_dir : str = None
        If given, the filepath to a directory to save all model checkpoints
        after fitting on an increment.
    """
    def __init__(
        self,
        uid=None,
        skip_fit=-1,
        save_dir=None,
        start_increment=0,
    ):
        """Initializes the OWHAR.

        Args
        ----
        uid : str = None
            An optional str unique identifier for this predictor. When not
            given, the uid property of this class' object is the trainer
            tensorboard log_dir version number. If 'datetime', saves datetime
            of init.
        skip_fit : see self
        save_dir : str = None
        start_increment : int = 0
        """
        self.skip_fit = skip_fit
        if save_dir:
            self.save_dir = create_filepath(save_dir)
        if uid:
            self._uid = uid
        else:
            self._uid = (
                f'{type(self).__name__}-'
                f"{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"
            )

    @property
    def increment(self):
        """Increments correspond to how many times the predictor was fit."""
        return self._increment

    @property
    def uid(self):
        """Returns a string Unique Identity for this predictor."""
        return self._uid

    def fit(self, dataset, val_dataset=None):
        """Incrementally fit the OWHAPredictor's parts. Update classes in
        classifier to match the training dataset. This assumes the training
        dataset contains all prior classes. This deep copy is convenient for
        ensuring the class indices are always aligned.
        """
        if self.skip_fit >= 0 and self._increment >= self.skip_fit:
            return
        self._increment += 1

    @abstractmethod
    def predict(self, dataset):
        """Predictor performs the prediction (classification) tasks given
        dataset.

        Args
        ----
        dataset : torch.Dataset
        task_id : str = None
            The str identifier of the task to perform with the given inputs.
            This assumes the proper dataset input format is given for each task
            or that every task has the same input format.When task_id is None,
            default, it performs all tasks sequentially.
        feedback_budget : int | float = None
            TODO implement a feedback budget that allows the predictor to
            request feedback for only so many samples, so the selection of
            which samples to request feedback for matters.
        """
        raise NotImplementedError()


class Recognizer(Predictor):
    """Abstract predictor with recognition functionality.

    Attributes
    ----------
    experience : pandas.DataFrame = None
        An ordered pandas DataFrame of with the index corresponding to the
        sample unique identifiers (integer based on experiment) and a column
        for the task label. Another column for the belief as a probability
        (certainty) in that label.

        `experience.columns == ['uid', 'sample_path', 'labels']`, where
        'uid' is the Unique Identifier of the sample as the integer given to it
        by the experiment upon recieving the sample. 'labels' is the
        predictor's knowledge of the discrete label given to that sample,
        whether provided by the oracle or inferred.

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
    """
    def __init__(self, **kwargs):
        """Initialize the OWHARecognizer.

        Args
        ----
        see Predictor.__init__
        """
        super().__init__(**kwargs)
        self.experience = pd.DataFrame(
            [],
            columns=['uid', 'sample_path', 'labels'],
        )
        self.recog_label_enc = None
        self.label_enc = None

    @property
    def n_recog_labels(self):
        return 0 if self.recog_label_enc is None else len(self.recog_label_enc)

    @property
    def n_known_labels(self):
        all_labels = 0 if self.label_enc is None else len(self.label_enc)
        return  all_labels - self.n_recog_labels

    @property
    def

    def fit(self, dataset, val_dataset=None):
        super().fit(dataset, val_dataset=None)

        # NOTE This is an unideal hotfix, the predictor should not effect
        # evaluator data, but in this case we need to use recogs as labels, so
        # those recogs need added if missing to the provided experience
        # (dataset).

        # Remove experience which oracle feedback was given (assumes experiment
        # is maintaining the predictor's experience)
        if len(self.experience) > 0:
            mask = self.experience['uid'].isin(dataset.data.index)
            unique_labels = self.experience['labels'].unique()
            self.experience = self.experience[~mask]

        # Update the predictor's label encoder with new knowns
        self.label_enc = deepcopy(dataset.label_enc)
        if self.recog_label_enc:
            # TODO rm recog classes that have been determined as a new class
            self.label_enc.append(self.recog_label_enc)

    def predict(self, dataset):
        if self.label_enc is None:
            raise ValueError('label enc is None. This predictor is not fit.')
        if isinstance(dataset, (torch.DataSet, torch.DataLoader)):
            features = torch.stack(list(dataset))

        detected_novelty = self.detect(features, preds) \
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

        if detected_novelty.any() and total_detected > 1:
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
                preds[detected_novelty]  = recogs.to(preds.device, preds.dtype)

        # If dataset contains uids unseen, add to predictor experience
        if len(self.experience) > 0:
            mask = [True] * len(dataset.data)
        else:
            mask = ~dataset.data.index.isin(self.experience['uid'])

        # TODO NOTE! you may run into the issue where val / test indices as
        # uid's are encountered and are not unique to the train sample indices.
        # Need to test this!

        if any(mask):
            self.experience = self.experience.append(pd.DataFrame(
                np.stack(
                    [
                        dataset.data.index[mask],
                        dataset.data[mask]['sample_path'],
                        self.label_enc.decode(
                            preds[mask].argmax(1).detach().numpy()
                        ),
                    ],
                    axis=1,
                ),
                columns=['uid', 'sample_path', 'labels'],
            ))

        # TODO also, i suppose the predictor should not hold onto val/test
        # samples.... though it can. and currently as written does.
        #   Fix is to ... uh. just pass a flag if eval XOR fit

        return preds

    @abstractmethod
    def detect(self, features):
        """The inference task of binary classification if the sample belongs to
        a known class or unknown class. This may be a post process step to
        predict().
        """
        raise NotImplementedError('Inheriting class overrides this.')


# TODO feedback_request: Enable different priority schemes.
#   1. some number of unknown recognized samples w/ highest certainty
#   2. prioritize recognized unknown samples w/ lowest certainty
#   3. prioritize general unknown samples w/ lowest certainty
