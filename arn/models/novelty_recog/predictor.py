"""The Open World Predictor with Novelty Recognition."""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
F = torch.nn.functional

from arn.models.owhar import OWHAPredictor

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
    """
    def __init__(self, **kwargs):
        """Initialize the OWHARecognizer.

        Args
        ----
        see OWHAPredictor.__init__
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

    def detect(self, features, preds, n_expected_classes=None):
        raise NotImplementedError('Inheriting class overrides this.')

    def recognize(self, features, n_expected_classes=None, **kwargs):
        raise NotImplementedError('Inheriting class overrides this.')

    def fit(self, dataset, val_dataset=None, task_id=None):
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

            # TODO Replace any sample's label still in experience with the feedback

            # If there are multiple feedbak classes assigned to an unknown
            # class, then need to figure out some logic to assign those
            # classes. Perhaps, simply make them unknown to be handled in recog

        # Update the predictor's label encoder with new knowns
        self.label_enc = deepcopy(dataset.label_enc)
        if self.recog_label_enc:
            # TODO rm recog classes that have been determined as a new class
            self.label_enc.append(self.recog_label_enc)

        # Ensure the new recogs are used in fitting.
        #   Predictor experience is the unlabeld recogs, temporairly append it
        #   and swap the label encoders.
        original_data_len = len(dataset.data)
        original_label_enc = dataset.label_enc

        dataset.data = dataset.data.append(self.experience)
        dataset.label_enc = self.label_enc

        # NOTE need todo val_datset management... if ever using it
        super().fit(dataset, val_dataset=None, task_id=None)

        # Now rm the experience from the dataframe and restore the label enc
        dataset.data = dataset.data.iloc[:original_data_len]
        dataset.label_enc = original_label_enc

        # NOTE if there are any unlabeled samples, then perform recognition.
        #   Currently not included in project's experiments.

    def predict(self, dataset):
        if self.label_enc is None:
            raise ValueError('label enc is None. This predictor is not fit.')
        preds = super().predict(dataset)
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

    # TODO feedback_request: Enable different priority schemes.
    #   1. some number of unknown recognized samples w/ highest certainty
    #   2. prioritize recognized unknown samples w/ lowest certainty
    #   3. prioritize general unknown samples w/ lowest certainty
