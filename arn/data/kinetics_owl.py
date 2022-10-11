"""Kinetics Open World Learning pipeline object.

Environment: incremental learning experiment over Kinetics data
Predictor: arn.predictor.OWHARPredictor
Actuators: Feedback request system
    - No feedback
    - Oracle feedback
    - Oracle feedback budgeted amount per increment
    - Oracle feedback budgeted amount overall
    - Feedback Translation?
"""
from copy import deepcopy
from dataclasses import dataclass, InitVar
import os
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from arn.data.kinetics_unified import (
    KineticsUnified,
    KineticsUnifiedFeatures,
    load_file_list,
    get_filename,
)
from arn.models.novelty_recog.recognizer import OWHARecognizer

from exputils.data.labels import NominalDataEncoder
from exputils.data.confusion_matrix import ConfusionMatrix
from exputils.data.ordered_confusion_matrix import OrderedConfusionMatrices
from exputils.io import create_filepath

import logging
logger = logging.getLogger(__name__)


def log_all_ocm_measures(ocm, known_label_enc=None):
    logger.info(
        'OrderedConfusionMatrices: measures over a total of %d classes that '
        'are unique across predictions and eval labels. The first measures '
        'have no reductions.',
        len(ocm.label_enc),
    )

    log_ocm_measures(ocm, 'No Reduction')
    cm = ocm.get_conf_mat()
    log_cm_measures(cm, 'No Reduction')

    logger.info('knowns total = %d', len(known_label_enc))
    logger.debug('knowns = %s', list(known_label_enc))

    # Reduce unknowns
    unknowns = set(cm.label_enc) - set(known_label_enc)
    unknowns.add('unknown')

    logger.info('unknowns total = %d', len(unknowns))
    logger.debug('unknowns = %s', unknowns)

    classif_cm = cm.reduce(unknowns, 'unknown', reduced_idx=0)

    log_cm_measures(classif_cm, 'Classification Task')

    # Reduce knowns
    recog_cm = cm.reduce(unknowns, 'known', reduced_idx=-1, inverse=True)

    log_cm_measures(recog_cm, 'Novelty Recognition')

    # Reduce both knowns and unknowns
    detect_cm = classif_cm.reduce(
        ['unknown'],
        'known',
        reduced_idx=-1,
        inverse=True,
    )
    log_cm_measures(detect_cm, 'Novelty Detection')

    logger.info(
        'Detection Confusion Matrix = \n%s\n%s',
        list(detect_cm.label_enc),
        detect_cm.mat,
    )


def log_ocm_measures(ocm, prefix):
    logger.info(
        '%s: OrderedConfusionMatrices: top-5 Accuracy: %.4f',
        prefix,
        ocm.accuracy(5),
    )
    logger.debug(
        'OrderedConfusionMatrices: top-1 Accuracy: %.4f',
        ocm.accuracy(),
    )


def log_cm_measures(cm, prefix):
    logger.info(
        '%s: OrderedConfusionMatrices (CM): top-1 Accuracy: %.4f',
        prefix,
        cm.accuracy(),
    )
    logger.info(
        '%s: OrderedConfusionMatrices (CM): top-1 MCC: %.4f',
        prefix,
        cm.mcc(),
    )
    logger.info(
        '%s: OrderedConfusionMatrices (CM): top-1 Mutual Information: %.4f',
        prefix,
        cm.mutual_information(),
    )
    logger.info(
        '%s: OrderedConfusionMatrices (CM): top-1 '
        'Arithmetic Normalized Mutual Information: %.4f',
        prefix,
        cm.mutual_information('arithmetic'),
    )


def get_known_and_unknown_dfs(
    n_increments,
    dataframe,
    known_label_enc,
    seed=None,
    np_gen=None,
    label_col='labels',
):
    """Get the list of incremental known classes, stratified shuffled and the
    DataFrame of unknown labels.

    Returns
    -------
    tuple
        Tuple of a list of dataframes of known class samples, and the dataframe
        of unknown class samples to be handled externally.
    """
    if seed is None:
        np_gen = None
    elif np_gen is None:
        np_gen = np.random.default_rng(seed)
    known_df = dataframe

    # Separate samples with known labels from those with unknown.
    known_mask = known_df[label_col].isin(known_label_enc)
    unknown_df = known_df[~known_mask]
    known_df = known_df[known_mask]

    logger.debug(
        'sum(known_mask) = %d; len(known_df) = %d',
        sum(known_mask),
        len(known_df),
    )
    logger.debug(
        'sum(~known_mask) = %d; len(unknown_df) = %d',
        sum(~known_mask),
        len(unknown_df),
    )
    logger.debug('len(known_label_enc) = %d', len(known_label_enc))

    # If unique counts <= splits, then include those samples in first inc
    unique_known_labels, counts = np.unique(
        known_df[label_col],
        return_counts=True,
    )

    mask_knowns_le_incs = np.array([False] * len(known_df))
    for i, count in enumerate(counts):
        if count > n_increments:
            continue
        mask_knowns_le_incs |= known_df[label_col] == unique_known_labels[i]

    # Create the placeholder for known increments w/ an empty known DataFrame.
    known_incs = [known_df[[False] * len(known_df)]] * n_increments
    if mask_knowns_le_incs.sum() == len(known_df):
        logger.warning(
            'There are not enough known class\' samples to spread across '
            'increments.'
        )
        logger.debug(
            'The unique known classes and their counts %s\n%s',
            unique_known_labels,
            counts,
        )
        known_incs[0] = known_df
        return known_incs, unknown_df

    known_incs[0] = known_df[mask_knowns_le_incs]
    known_df = known_df[~mask_knowns_le_incs]

    # Otherwise stratify split the samples across the future increments
    # they persist across the rest of the future split.
    skf = StratifiedKFold(
        n_increments,
        random_state=seed,
        shuffle=np_gen is not None,
    )

    # Stratified folds for known class samples, ensuring presence and balance
    # across incs
    for i, (_, test) in enumerate(
        skf.split(known_df['sample_index'], known_df[label_col])
    ):
        known_incs[i] = known_incs[i].append(known_df.iloc[test])

    return known_incs, unknown_df


def get_increments(
    n_increments,
    src_datasplit,
    known_label_enc,
    seed=None,
    label_col='labels',
    intro_freq_first=False,
    deepcopy_label_enc=False,
):
    """Given a source DataSplit object, returns a list of incremental DataSplit
    objects. This stratified shuffle sthe known classes in known_label_enc,
    create a near uniform balance of unique unknown/novel classes that are
    introduced across the increments. The unique unknown classes' samples are
    then spread across the remaining increments.

    If there are unique unknowns in the validate or test splits outide of train,
    then those are kept within those splits, spread across the increments using
    stratified splitting.

    If there are less samples of a class than the number of remaining
    increments, then those samples are added into the first increment they
    occur in, rather than being spread across the future remaining increments.

    Args
    ----
    n_increments : int
        The number of increments to create
    src_datasplit : arn.data.kinetics_owl.DataSplits
        KineticsUnified
        The source KineticsUnified Dataset for splitting into the incremental
        datasets.
    known_label_enc : exputils.data.labels.NominalDataEncoder
    seed : int = None
    label_col : str = 'labels'
    intro_freq_first : bool = False
        If True, introduces the unknown class with the most frequent samples
        first and procedes to do so for the remaining unknown classes in
        descending order. This will weakly avoid too few samples of a class in
        an increment.
    deepcopy_label_enc : bool = False
        Optional because out of this function, if called in a loop, allows
        progressive updating of the original obejct.

    Returns
    -------
    list
        List of KineticsUnified Datasets that form the ordered increments.
    """
    if seed is None:
        np_gen = None
    else:
        np_gen = np.random.default_rng(seed)

    if deepcopy_label_enc:
        known_label_enc = deepcopy(known_label_enc)

    # NOTE assumes train is always present in given DataSplits.
    tmp_dataset = deepcopy(src_datasplit.train)
    #del src_datasplit
    tmp_dataset.data = None
    tmp_dataset.label_enc = None

    # Get known labels and unknown dataframe from each split.
    knowns_splits = []
    unknowns_splits = []
    split_names = ['train', 'validate', 'test']
    for dset_name in split_names:
        tmp_dset = getattr(src_datasplit, dset_name)
        if tmp_dset is None:
            knowns_splits.append(None)
            unknowns_splits.append(None)
        else:
            knowns, unknowns = get_known_and_unknown_dfs(
                n_increments,
                tmp_dset.data,
                known_label_enc,
                seed,
                np_gen,
                label_col,
            )
            knowns_splits.append(knowns)
            unknowns_splits.append(unknowns)

    label_enc = known_label_enc
    del known_label_enc

    # NOTE randomization criteron that forces certain classes to be separate,
    # e.g. classes farther from each other in some class hierarchy should be
    # separated across the increments such that there is sufficiently novel
    # classes in each increment.

    # NOTE assumes train contains all new unknown classes we care to introduce
    # over new increments.
    unknown_df = unknowns_splits[0]

    # Randomize unique, unknown classes across n increments. Last increment w/
    # remainder.
    unique, unique_inv, unique_counts = np.unique(
        unknown_df[label_col],
        return_inverse=True,
        return_counts=True,
    )
    n_unique = len(unique)
    logger.debug('unknowns: n_unique = %d', n_unique)

    if intro_freq_first:
        # Descending order of unknown classes introduced over incs.
        unique_perm = np.argsort(unique_counts)
    elif np_gen is not None:
        # Uniform random shuffle
        unique_perm = np.arange(n_unique)
        np_gen.shuffle(unique_perm)
    else:
        unique_perm = None

    if unique_perm is not None:
        nde = NominalDataEncoder(unique_perm)
        unique = unique[unique_perm]
        unique_inv = nde.encode(unique_inv)

    classes_per_inc = np.ceil(n_unique / n_increments)
    unique_slices = np.cumsum(
        [0] + [classes_per_inc] * n_increments,
        dtype=int,
    )

    # Stratified folds for val & test only unknown class samples, ensuring
    # presence and balance across incs. Appends to known_splits to be included.
    not_split_only_unks = set(unique) | set(label_enc)
    split_only_unks = []
    for k, unk_split_df in enumerate(unknowns_splits[1:], start=1):
        if unk_split_df is None:
            split_only_unks.append(None)
            continue
        only_unks = set(unk_split_df[label_col].unique()) - not_split_only_unks

        # Save sorted only_unks for val's and test's label enc over incs.
        split_only_unks.append(sorted(only_unks))

        if not only_unks:
            logger.debug(
                'There are no %s only unknown classes.',
                'validate' if k == 1 else 'test',
            )
            continue
        logger.debug(
            'There are %d %s only unknown classes.',
            len(only_unks),
            'validate' if k == 1 else 'test',
        )

        split_skf = StratifiedKFold(
            n_increments,
            random_state=seed,
            shuffle=np_gen is not None,
        )

        # Prune all but samples w/ labels in only_unks
        unk_split_df = unk_split_df.copy(deep=True)
        unk_split_df = unk_split_df[unk_split_df[label_col].isin(only_unks)]

        # Loop thru split's unks over incs & append to known splits' df
        for i, (train, test) in enumerate(split_skf.split(
            unk_split_df['sample_index'],
            unk_split_df[label_col],
        )):
            knowns_splits[k][i] = knowns_splits[k][i].append(
                unk_split_df.iloc[test]
            )

    # Loop through incs and create the incremental KineticsUnified datasets
    increments = []
    persistent_unknowns = []
    for i in range(n_increments):
        # Ordered update of the label encoder as labels are encountered
        inc_uniques = unique[unique_slices[i]:unique_slices[i+1]]
        label_enc.append(unique[unique_slices[i]:unique_slices[i+1]])

        remainder = n_increments - i
        if remainder > 1:
            persist_unk_skf = StratifiedKFold(
                remainder,
                random_state=seed,
                shuffle=np_gen is not None,
            )
            persistent_unks = [None] * 3
        else:
            persist_unk_skf = None
            persistent_unks = None

        # Create each k-th data split: train, val, test
        inc_datasets = []
        for k, unknown_df in enumerate(unknowns_splits):
            if unknown_df is None:
                inc_datasets.append(None)
                continue
            inc_dataset = deepcopy(tmp_dataset)
            inc_dataset.label_enc = deepcopy(label_enc)

            if k > 0 and split_only_unks[k - 1]:
                # Append split only unks to the label encoder.
                inc_dataset.label_enc.append(split_only_unks[k - 1])

            # Get all of the unknowns introduced at this increment
            unks = unknown_df[unknown_df[label_col].isin(inc_uniques)]
            logger.debug(
                'increment %d: split %s: total new unknown samples to be '
                'spread across remaining increments = %d',
                i,
                split_names[k],
                len(unks),
            )

            # Stratified shuffle split across this and remaining incs
            if persistent_unks is not None:
                # Check unks' samples for a class < n splits.
                unique_unks, unks_counts = np.unique(
                    unks[label_col],
                    return_counts=True,
                )
                unks_lt_rem_mask = unks[label_col].isin(
                    unique_unks[unks_counts < remainder]
                )
                unks_lt_rem_count = unks_lt_rem_mask.sum()

                if unks_lt_rem_count < len(unks):
                    if unks_lt_rem_count != 0:
                        # Separate classes w/ too few samples from others
                        unks_lt_rem = unks[unks_lt_rem_mask]
                        unks = unks[~unks_lt_rem_mask]

                    unknown_incs = [
                        unks.iloc[test] for train, test in
                        persist_unk_skf.split(
                            unks['sample_index'], unks[label_col]
                        )
                    ]
                    logger.debug(
                        'len(unknown_incs) at i=%d, k=%s: %d; remainder = %d',
                        i,
                        split_names[k],
                        len(unknown_incs),
                        remainder,
                    )

                    persistent_unk_df = unknown_incs.pop(0)

                    if unks_lt_rem_count != 0:
                        # Save those with too few samples to this inc only
                        persistent_unk_df = persistent_unk_df.append(
                            unks_lt_rem
                        )
                else:
                    # Simply make the entire dataframe for this increment only
                    persistent_unk_df = unks
                    # Creating unknown_incs as None to avoid accidental repeats
                    # when not last increment (remainder > 1).
                    unknown_incs = None
            else:
                # Handle remainder == 1 case, no stratified splitting
                persistent_unk_df = unks

            # Combine the persistent unknowns into one DataFrame for this inc
            for j in range(i):
                if persistent_unknowns[j][k]: # not None or not an empty stack
                    persistent_unk_df = persistent_unk_df.append(
                        persistent_unknowns[j][k].pop(0)
                    )

            # Save this increment's stack of persistent unknowns across incs
            if persistent_unks is not None:
                persistent_unks[k] = unknown_incs

            logger.debug(
                'increment %d: split %s: unknown samples = %d',
                i,
                split_names[k],
                len(persistent_unk_df),
            )
            logger.debug(
                'increment %d: split %s: known samples = %d',
                i,
                split_names[k],
                len(knowns_splits[k][i]),
            )
            inc_dataset.data = knowns_splits[k][i].append(persistent_unk_df)
            inc_datasets.append(inc_dataset)

        persistent_unknowns.append(persistent_unks)

        increments.append(DataSplits(
            train=inc_datasets[0],
            validate=inc_datasets[1],
            test=inc_datasets[2],
        ))

    return increments


class EvalDataSplitConfig(NamedTuple):
    """Configure the data split's saving of predictions or evaluation measures.
    Defaults to saving nothing.

    Attributes
    ----------
    pred_dir: str = None
        The directory where the predictions will be saved.
        Defaults to None and when None no predictions are saved.
    eval_dir: str  = None
        The directory where the evaluation measures will be saved.
        Defaults to None and when None no evaluations are saved.
        NOTE should probably just use tensorboard for this? except for
        confusion matrices.
    file_prefix: str = ''
        A prefix to be added to prior to the filename, but AFTER any given
        `prefix` in eval().
    save_preds_with_labels: bool = True
        If True, saves the predictions with the labels. Otherwise only saves
        the predictions.
    """
    # NOTE should make this optionally save to a database, like PostgreSQL.
    pred_dir: str = None
    eval_dir: str = None
    file_prefix: str = ''
    save_preds_with_labels: bool = True

    def __bool__(self):
        return self.pred_dir is not None or self.eval_dir is not None

    def eval(
        self,
        data_split,
        preds,
        measures,
        prefix=None,
        predictor=None,
    ):
        """Evaluated the predictions to the given datasplit using the measures.

        Args
        ----
        data_split : DataSplit
        preds : torch.Tensor
        measures : list
            List of measures to use as either a string identifier or callable
            object expecting two parameters of (reference labels, predictions)
        prefix : str = None
            The prefix to concatenate at the beginning of `self.file_prefix`
            to create the resulting filepath for storing the measures.
        """
        prefix = os.path.join(prefix, self.file_prefix)
        labels = None

        if predictor is None:
            n_classes = len(data_split.label_enc)
            if preds.shape[1] < n_classes:
                # TODO the mismatch in test to the predictor.label_enc will
                # result in issues!

                # Relies on data split label enc including all prior known
                # classes the predictor has seen.
                preds = np.hstack((
                    preds,
                    np.zeros([preds.shape[0], n_classes - preds.shape[1]]),
                ))
            label_enc = data_split.label_enc
        else:
            label_enc = deepcopy(predictor.label_enc)

        if isinstance(preds, torch.Tensor):
            if preds.device.type == 'cuda':
                preds = preds.cpu().numpy()
            else:
                preds = preds.numpy()

        if predictor is None:
            n_classes = len(data_split.label_enc)
            if preds.shape[-1] < n_classes:
                missing_labels = list(data_split.label_enc.keys())[
                    :n_classes - preds.shape[-1]
                ]
                logger.warning(
                    'The predictions are missing labels (%d) within the data '
                    'set used to evaluate it. The predictions are being '
                    'padded with zeros for the sake of evaluation only.',
                    len(missing_labels),
                )
                logger.debug(
                    'Last %d labels in data_split.label_enc %s. May exist in '
                    'predictor label encoder, but unable to tell when not '
                    'given predictor.',
                    len(missing_labels),
                    missing_labels,
                )
                # NOTE Relies on data split label enc including all prior known
                # classes the predictor has seen.
                preds = np.hstack((
                    preds,
                    np.zeros([preds.shape[0], len(missing_labels)]),
                ))
            label_enc = data_split.label_enc
        elif isinstance(predictor, OWHARecognizer):
            label_enc = deepcopy(predictor.label_enc)
            # TODO If these are changing cuz new unknown_# from recognize_fit()
            # then the state needs updated correctly after recognize_fit()! The
            # unknown_# would be misaligned i believe.
            n_recogs_in_pred = preds.shape[-1] - predictor.n_known_labels
            n_recogs = predictor.n_recog_labels
            if n_recogs_in_pred < n_recogs:
                logger.warning(
                    'n_recogs_in_pred < n_recogs: %d < %d. '
                    'preds.shape[-1] = %d ; predictor.n_known_labels = %d '
                    'predictor.n_labels = %d '
                    'When the predictor '
                    'is given, then the label encoder used is the  '
                    "predictor's label encoder and this was not caught by the "
                    'predictor itself. Beware this may indicate an issue in '
                    'class alignment of preds.',
                    n_recogs_in_pred,
                    n_recogs,
                    preds.shape[-1],
                    predictor.n_known_labels,
                    predictor.n_labels,
                )
                # Update end of preds
                pad_widths = [(0, 0)] * len(preds.shape)
                pad_widths[-1] = (0, n_recogs - n_recogs_in_pred)
                preds = np.pad(
                    preds,
                    pad_widths,
                    'constant',
                    constant_values=0,
                )
        else:
            label_enc = deepcopy(predictor.label_enc)

        if self.pred_dir:
            if self.save_preds_with_labels:
                if data_split.one_hot:
                    labels = np.vstack(
                        [row[1] for row in data_split]
                    )#.reshape(-1, 1)
                    preds = preds.reshape(labels.shape[0], preds.shape[-1])
                    contents = np.hstack([
                        data_split.label_enc.decode(
                            labels,
                            one_hot_axis=-1,
                        ).reshape(-1, 1),
                        preds,
                    ])
                else:
                    labels = data_split.label_enc.decode(
                        np.vstack([row[1] for row in data_split]),
                    ).reshape(-1, 1)
                    contents = np.hstack([labels, preds])

                logger.debug(
                    '%s: eval dsplit: save_preds_with_labels: type(labels) = %s',
                    type(self).__name__,
                    type(labels),
                )
                logger.debug(
                    '%s: eval dsplit: save_preds_with_labels: len(labels) = %s',
                    type(self).__name__,
                    len(labels),
                )
                logger.debug(
                    '%s: eval dsplit: save_preds_with_labels: type(preds) = %s',
                    type(self).__name__,
                    type(preds),
                )
                logger.debug(
                    '%s: eval dsplit: save_preds_with_labels: len(preds) = %s',
                    type(self).__name__,
                    len(preds),
                )

                # NOTE Assumes samples are never shuffled in eval.
                data_split.data[
                    ['youtube_id', 'time_start', 'time_end']
                ].join(
                    pd.DataFrame(
                        contents,
                        columns=['target_labels'] + list(label_enc),
                        index=data_split.data.index,
                    ),
                    #on=['youtube_id', 'time_start', 'time_end'],
                    how='left',
                ).to_csv(
                    create_filepath(os.path.join(prefix, 'preds.csv')),
                    index=False,
                )
                del contents
            else:
                data_split.data[
                    ['youtube_id', 'time_start', 'time_end']
                ].join(
                    pd.DataFrame(
                        preds,
                        columns=list(label_enc),
                        index=data_split.data.index,
                    ),
                    #on=['youtube_id', 'time_start', 'time_end'],
                    how='left',
                ).to_csv(
                    create_filepath(os.path.join(prefix, 'preds.csv')),
                    index=False,
                )

        if self.eval_dir:
            if labels is None:
                labels = np.vstack([row[1].numpy for row in data_split])

            if data_split.one_hot:
                # NOTE when eval measures compare one hot vs prob vectors, then
                # the conversion of single class to prov vec of class needs
                # handled.
                labels = labels.argmax(axis=-1).reshape(-1, 1)

            logger.debug(
                '%s: eval dsplit: eval_dir: type(labels) = %s',
                type(self).__name__,
                type(labels),
            )
            logger.debug(
                '%s: eval dsplit: eval_dir: len(labels) = %s',
                type(self).__name__,
                len(labels),
            )
            logger.debug(
                '%s: eval dsplit: eval_dir: type(preds) = %s',
                type(self).__name__,
                type(preds),
            )
            logger.debug(
                '%s: eval dsplit: eval_dir: len(preds) = %s',
                type(self).__name__,
                len(preds),
            )

            if predictor is not None:
                # Add missing data split labels to the end of pred label enc
                missing_labels = set(data_split.label_enc.encoder) \
                    - set(label_enc.encoder)
                if missing_labels:
                    logger.warning(
                        'The predictor is missing labels (%d) within the data '
                        'set used to evaluate it. The predictions are being '
                        'padded with zeros for the sake of evaluation only.',
                        len(missing_labels),
                    )
                    logger.debug('missing labels = %s', missing_labels)

                    label_enc.append(missing_labels)

                    # Update end of preds
                    pad_widths = [(0, 0)] * len(preds.shape)
                    pad_widths[-1] = (0, len(missing_labels))
                    preds = np.pad(
                        preds,
                        pad_widths,
                        'constant',
                        constant_values=0,
                    )

            for measure in measures:
                if issubclass(measure, ConfusionMatrix):
                    measurements = measure(labels, preds, label_enc)
                    measurements.save(os.path.join(prefix, 'preds_cm.csv'))
                elif issubclass(measure, OrderedConfusionMatrices):
                    measurements = measure(
                        labels,
                        preds,
                        label_enc,
                        5,
                    )
                    if logging.root.isEnabledFor(logging.DEBUG):
                        # TODO should the above be logger.isEnabledFor?
                        log_all_ocm_measures(
                            measurements,
                            None if predictor is None
                                else predictor.known_label_enc,
                        )
                    measurements.save(os.path.join(prefix, 'preds_top-cm.h5'))
                else:
                    raise NotImplementedError('TODO: non-confusion matrix.')
                    measurements = measure(labels, preds)


@dataclass
class EvalConfig:
    """Configure the saving of predictions or evaluation measures.
    Defaults to saving nothing.

    Attributes
    ----------
    train: EvalDataSplitConfig = None
        The configuration for saving the predictions and evaluation measures of
        the training split of the data.
    validate: EvalDataSplitConfig = None
        The configuration for saving the predictions and evaluation measures of
        the validation split of the data.
    test: EvalDataSplitConfig = None
        The configuration for saving the predictions and evaluation measures of
        the testing split of the data.
    root_dir: str = ''
        An optional root directory that is appeneded to all paths accessed
        withing the EvalConfig.
    measures : str = 'ordered_confusion_matrix'
        A list of callables or a str stating confusion tensor or confusion
        matrix. If `ordered_confusion_matrix`, then the k = 5 ordered confusion
        matrices are stored.
    save_features : bool = False
        If true, saves the features alongside the predictions in same dir.
        Does not save the features otherwise, the default.
    """
    train: EvalDataSplitConfig = None
    validate: EvalDataSplitConfig = None
    test: EvalDataSplitConfig = None
    root_dir: str = ''
    measures: InitVar[list] = 'ordered_confusion_matrix'
    save_features: bool = False

    def __post_init__(self, measures):
        """Handles init of measures when a single str.

        Args
        ----
        see self
        """
        if isinstance(measures, str):
            if measures.lower() in {'confusion matrix', 'confusion_matrix'}:
                self.measures = [ConfusionMatrix]
            elif measures.lower().replace(' ', '_') in {
                'ordered_confusion_matrix',
                'ordered_confusion_matrices',
            }: # Assumes top 5
                self.measures = [OrderedConfusionMatrices]
            else:
                raise TypeError(f'Expected a list, not a str! Got {measures}')
        else:
            self.measures = measures

    def __bool__(self):
        return self.train or self.validate or self.test

    def eval(self, data_splits, predictor, prefix=None, experience=None):
        """Given the datasplits, performs the predictions and evaluations to
        be saved.

        Args
        ----
        data_splits : DataSplits
            The data splits to potentially be predicted on and evaluated.
        predictor : Callable
            A function of the predictor to perform predictions given a dataset
            within the data_splits object.
        prefix : str = None
            An optional prefix to add to the paths AFTER the root_dir. This
            would be useful for adding the step number and phase of that step,
            such as if inference on new unlabeled data, or inference on data
            after feedback update.
        """
        if prefix:
            prefix = os.path.join(self.root_dir, prefix)
        else:
            prefix = self.root_dir

        # NOTE relies on predictor to turn data_split into a DataLoader

        #for name, dsplit in data_splits._asdict().items():
        for name in ['train', 'validate', 'test']:
            dsplit = getattr(data_splits, name)
            if dsplit is not None and self.train:
                logger.info("Predicting `label` for `%s`'s %s.", prefix, name)

                reset_return_label = dsplit.return_label
                if dsplit.return_label:
                    dsplit.return_label = False
                if experience:
                    preds = predictor.predict(dsplit, experience)
                else:
                    preds = predictor.predict(dsplit)
                prefix_dir = os.path.join(prefix, name)

                # Optionally obtain and save feature extractions of ANN
                if self.save_features:
                    logger.info("Saving Features for %s's %s.", prefix, name)
                    # Expects extracts first, separates from preds
                    torch.save(
                        preds[1],
                        create_filepath(os.path.join(prefix_dir, 'features.pt'))
                    )
                    preds = preds[1]

                dsplit.return_label = True

                logger.debug(
                    '%s: eval dsplit: %s ; type(preds) = %s',
                    type(self).__name__,
                    name,
                    type(preds),
                )
                logger.debug(
                    '%s: eval dsplit: %s ; preds.shape = %s',
                    type(self).__name__,
                    name,
                    preds.shape,
                )

                getattr(self, name).eval(
                    dsplit,
                    preds,
                    self.measures,
                    prefix_dir,
                    predictor if hasattr(predictor, 'predict') else None,
                )
                dsplit.return_label = reset_return_label


#class DataSplits(NamedTuple):
@dataclass
class DataSplits:
    """Contains the KineticsUnifiedFeatures for train, validate, and test.

    Attributes
    ----------
    train: KineticsUnifiedFeatures = None
    validate: KineticsUnifiedFeatures = None
    test: KineticsUnifiedFeatures = None
    """
    train: KineticsUnifiedFeatures = None
    validate: KineticsUnifiedFeatures = None
    test: KineticsUnifiedFeatures = None
    ensure_knowns: InitVar[bool] = True

    def __post_init__(self, ensure_knowns=True):
        """Init handle knowns in train consistent across val and test encoders.

        Args
        ----
        see self
        """
        self.add_knowns(inplace=True)

    def append(self, data_splits, copy=True):
        """Given data_split update internal data_split.
        The updated split's label encoder is appended with the sorted new
        labels.

        Args
        ----
        data_splits : DataSplits
            The other data split to be used to update this object's data
            splits. Perhaps should call this append.
        copy : bool = True
            If True, copies the source and other data splits when combining
            them.

        Returns
        -------
        DataSplits
            The new Datasplits from appending new data and labels to the source
            data splits.
        """
        # Most basic is concat new data splits to end of current one. (in df)
        splits = []
        for name in ['train', 'validate', 'test']:
            split = getattr(self, name)
            data_split = getattr(data_splits, name)

            if data_split:
                if split is not None:
                    if copy:
                        split = deepcopy(split)
                    split.data = split.data.append(data_split.data)
                    split.label_enc.append(
                        sorted(
                            data_split.label_enc.keys()
                            - split.label_enc.keys()
                        )
                    )
                    splits.append(split)
                else:
                    if copy:
                        splits.append(deepcopy(data_split))
                    else:
                        splits.append(data_split)
            else:
                if copy:
                    splits.append(deepcopy(split))
                else:
                    splits.append(split)

        # NOTE support check for repeat or non-unique sample ids, which then
        # would mean to update those prior experiences.
        return DataSplits(*splits)

    def add_knowns(self, sort_diff=True, inplace=False):
        """Creates a new DataSplits with the label encoders having their
        classes be updated such that the first split's labels are in all
        following label encoders and with the same encodings is used, then any
        different from the next split are appended to the end in order as exist
        or sorted (default).

        Order is train, val, test. This ensures the following:
        len(train.label_enc) <= len(val.label_enc) <= len(test.label_enc)

        Args
        ----
        order : list
            List of str or ints of the order.
        sort : bool = True
            Sorts the later split's new labels before appending them to the end
            of the label encoder.
        inplace : bool = False
        """
        if not inplace:
            splits = []

        label_enc = None
        for name in ['train', 'validate', 'test']:
            split = getattr(self, name)
            if split is None:
                if not inplace:
                    splits.append(None)
                continue
            elif not hasattr(split, 'data'):
                # HotFix for annotation_path = None and docstr
                if not inplace:
                    splits.append(None)
                else:
                    setattr(self, name, None)
                continue
            if label_enc is None:
                label_enc = deepcopy(split.label_enc)
                if not inplace:
                    splits.append(split)
                continue

            new_labels = list(split.label_enc.keys() - label_enc.keys())

            if sort_diff:
                new_labels = sorted(new_labels)

            label_enc.append(new_labels)

            if not inplace:
                split = deepcopy(split)
                split.label_enc = label_enc
                splits.append(split)
            else:
                split.label_enc = label_enc

        if not inplace:
            return DataSplits(*splits)


class KineticsOWL(object):
    """Kinetics Open World Learning Pipeline for incremental recognition.
    This is the class that contains all other objects to run an experiment
    for Kinetics, including the Kinetics datasets, the predictor, and
    evaluation code. This serves as the glue between them all.

    Attributes
    ----------
    environment : KineticsOWLExperiment
    predictor : arn.models.predictor.OWHAPredictor
        predictor : arn.models.predictor.EVMPredictor
        arn.models.predictor.load_evm_predictor
        TODO docstr: support at least basic checking of multiple configurable
        types. Or maybe just parse all of them as options and support so in
        MultiType.
    feedback_type : str = 'oracle'
        Originally intended to indicate different types of feedback, this is
        intended to always have the value 'oracle' for now.
        Different types of feedback is now a future feature.
    feedback_amount : float = 1.0
        The maximum percentage of feedback as ground truth labels given by an
        oracle to provide to the predictor upon request. Valid values within
        range [0, 1.0].
    rng_state : int = None
        Random seed.
    eval_on_start : bool = False
        If False (the default), does not evaluate an untrained predictor. If
        True, evaluated an untrained predictor. May be a good idea to evaluate
        some untrained predictors, espcially if they were pre-trained.
    eval_config : EvalConfig = None
        The configuration of prediction and evaluation saving for the specified
        data splits: train, validate, and test.

        Considering multiple EvalConfigs, one for initial inference given new
        data, one for after fitting. Posisbly 2 for separate predict and
        novelty detection for each case.

        Defaults to no saving of predictions or configurations when None.
    post_feedback_eval_config : EvalConfig = None
        An optional EvalConfig for specifically after feedback has been
        provided on the same step to assess the predictor on the same data
        split given to it on that step.

        When None or False, this evaluation after feedback does not occur. When
        True, this uses the same EvalConfig object as `eval_config`, otherwise
        when an EvalConfig uses that object's configuration.
    experience : DataSplits = None
        If `maintain_experience` is True in __init__, then the simulation
        maintains the past experienced data for the predictor.
    tasks : str | list = None
        Not Implemented atm! Future TODO!

        A singular or list of string identifiers corresponding to a column in
        the KineticsUnifed Datasets. These strings determine the task's
        expected output under the assumption of the same input, where a task is
        defined as learning a mapping of inputs to outputs.
    """
    def __init__(
        self,
        environment,
        predictor,
        #augmentation=None # sensors
        feedback_type='oracle',
        feedback_amount=1.0,
        rng_state=None,
        measures=None,
        eval_on_start=False,
        eval_config=None,
        post_feedback_eval_config=None,
        tasks=None,
        maintain_experience=True,
        labels=None,
        start_step=0,
        # configure state saving ...
    ):
        """Initialize the KineticsOWL experiment.

        Args
        ----
        environment : see self
        predictor : see self
        feedback_type : see self
        feedback_amount : see self
        rng_state : see self
        eval_on_start : see self
        eval_config : see self
        post_feedback_eval_config : see self
        tasks : see self
        maintain_experience : bool = True
            If False, the default, the past experienced samples are not saved
            in the simulation for use by the predictor. Otherwise, the
            experienced samples are saved by concatenating the new data splits
            to the end of the prior ones.
        labels : str = None
        start_step : int = 0
            The starting step of the experiment. Used to fast forwards the
            KOWLExperiment to the correct state.
        """
        # NOTE handle seed/rng_state if given, otherwise randomly select seed.
        self.rng_state = rng_state

        self.environment = environment
        self.predictor = predictor
        self.feedback_type = feedback_type
        self.feedback_amount = feedback_amount
        self.eval_on_start = eval_on_start
        self.eval_config = eval_config
        if post_feedback_eval_config is True:
            self.post_feedback_eval_config = eval_config
        else:
            self.post_feedback_eval_config = post_feedback_eval_config

        # NOTE will have to change this if handling multi-tasks in same
        # experiment!
        #if tasks is None:
        #     # NOTE support this in predictor and the datasets in labels
        #     #returned!
        #    self.tasks = ['labels', 'detect']

        # Maintain experience here for the predictor
        if maintain_experience:
            self.experience = DataSplits()
        else:
            self.experience = None

        if start_step > 0:
            self.fast_forward(start_step)

    @property
    def increment(self):
        return self.environment.increment

    def step(self, state=None):
        """The incremental step in incremental learning of Kinetics OWL."""
        # 1. Get new data (input samples only)
        logger.info("Getting step %d's data.", self.increment)
        eval_prefix = f'{self.predictor.uid}{os.path.sep}step-{self.increment}'
        new_data_splits = self.environment.step()

        logger.debug(
            'len(new_data_splits.train) = %d',
            0 if new_data_splits.train is None else len(new_data_splits.train),
        )
        logger.debug(
            'len(new_data_splits.validate) = %d',
            0 if new_data_splits.validate is None else len(new_data_splits.validate),
        )
        logger.debug(
            'len(new_data_splits.test) = %d',
            0 if new_data_splits.test is None else len(new_data_splits.test),
        )

        # 2. Inference/Eval on new data if self.eval_untrained_start
        if (self.increment == 1 and self.eval_on_start) or self.increment > 1:
            # NOTE Predict for the Task(s), useful when multiple tasks to be
            # handled by one predictor.
            #for task_id in self.tasks:
            #    pass

            logger.info(
                "Eval for new data, no feedback, for step %d.",
                self.increment - 1,
            )
            if self.experience:
                self.experience.train.return_label = False
            self.eval_config.eval(
                new_data_splits,
                self.predictor,
                #    if not self.eval_config.save_features
                #    else self.predictor.extract_predict,
                f'{eval_prefix}_new-data_predict',
                experience=self.experience
                    if isinstance(self.predictor, OWHARecognizer) else None,
            )
            if self.experience:
                self.experience.train.return_label = True
            # NOTE novelty detect task is based on the NominalDataEncoder for
            # the current time step as it knows when something is a known or
            # unknown class at the current time step.
            #   Keep experience/datasplit label encoders in sync.
            #   Use proper novelty detection measures of performance!
            #       - Confusion Matrix : which class is confused w/ novelty?
            #       - Difference to actual novelty occurrence (by sample idx)
            #           If early detection, negative, otherwise positive.

        if self.feedback_type:
            # 3. Opt. Feedback on this step's new data
            logger.info(
                "Requesting feedback (%s: %f) for step %d's data.",
                self.feedback_type,
                self.feedback_amount,
                self.increment - 1,
            )
            if self.increment == 1:
                # Provide full feedback on initial inc
                new_data_splits.train.data['feedback'] \
                    = new_data_splits.train.data['labels']
            elif self.feedback_amount > 0:
                # Provide the uids the predictor may request from and amount
                feedback_uids = self.predictor.feedback_request(
                    torch.stack(list(new_data_splits.train)),
                    new_data_splits.train.data['sample_index'].values,
                    self.feedback_amount,
                )
                logger.debug('feedback_uids = %s', feedback_uids)
                feedback_mask = \
                    new_data_splits.train.data['sample_index'].isin(
                    feedback_uids[:int(np.floor(
                        self.feedback_amount * len(new_data_splits.train)
                    ))]
                )
                new_data_splits.train.data.loc[feedback_mask, 'feedback'] \
                    = new_data_splits.train.data.loc[feedback_mask, 'labels']
                logger.info('sum(feedback_mask) = %d', sum(feedback_mask))
                logger.debug('feedback_mask = %s', feedback_mask)

            new_data_splits = self.environment.feedback(new_data_splits)

            logger.info(
                "Updating with feedback (%s: %f) for step %d's data.",
                self.feedback_type,
                self.feedback_amount,
                self.increment - 1,
            )
            if self.experience:
                # Add new data to experience
                self.experience = self.experience.append(new_data_splits)

                logger.debug(
                    'len(self.experience.train) = %d',
                    0 if self.experience.train is None else \
                        len(self.experience.train),
                )
                logger.debug(
                    'len(self.experience.validate) = %d',
                    0 if self.experience.validate is None else \
                        len(self.experience.validate),
                )
                logger.debug(
                    'len(self.experience.test) = %d',
                    0 if self.experience.test is None else \
                        len(self.experience.test),
                )

                # TODO the predictor is still given all labels in train even if
                # it does not recieve samples for an unknown label in train!
                # This needs fixed!!! (currently OWHARecognizer ignores them,
                # but need to establish better barriers to information.
                # Predictor code should never have access to the evaluator's
                # insights.)

                if self.increment == 1 or self.feedback_amount > 0:
                    # 4. Opt. Predictor Update/train on new data w/ feedback
                    #label_col = self.experience.train.label_col
                    self.experience.train.label_col = 'feedback'
                    if self.experience.validate is not None:
                        self.experience.validate.label_col = 'feedback'
                    if self.experience.test is not None:
                        self.experience.validate.label_col = 'feedback'
                    self.predictor.fit(
                        self.experience.train,
                        self.experience.validate,
                    )
                    #self.experience.train.label_col = label_col
            elif self.increment == 1 or self.feedback_amount > 0:
                label_col = new_data_splits.train.label_col
                new_data_splits.train.label_col = 'feedback'
                self.predictor.fit(
                    new_data_splits.train,
                    new_data_splits.validate,
                )
                new_data_splits.train.label_col = label_col

            logger.info(
                "Post-feedback Eval (%s: %f) for step %d.",
                self.feedback_type,
                self.feedback_amount,
                self.increment - 1,
            )
            if self.experience:
                self.experience.train.return_label = False
            # 5. Opt. Predictor eval post update
            self.post_feedback_eval_config.eval(
                new_data_splits,
                self.predictor,
                #    if not self.eval_config.save_features
                #    else self.predictor.extract_predict,
                f'{eval_prefix}_post-feedback_predict',
                experience=self.experience
                    if isinstance(self.predictor, OWHARecognizer) else None,
            )
            if self.experience:
                self.experience.train.return_label = True

        # NOTE 6. Opt. Evaluate the updated predictor on entire experience

    def run(self, max_steps=None, tqdm=None):
        """The entire experiment run loop."""
        for i in range(self.increment, self.environment.total_increments):
            logger.info(
                "Starting step (init step at zero): %d. Increment %d / %d",
                i,
                self.increment,
                self.environment.total_increments - 1,
            )
            self.step()

    def fast_forward(self, exclusive_end_step):
        """Run through the steps of the experiment such that it is ready on the
        given end state.
        """
        logger.info(
            'Fast forwarding the experiment by %d steps',
            exclusive_end_step,
        )
        for i in range(exclusive_end_step):
            # 1. Get new data (input samples only)
            logger.info(
                "Fast forward: Getting step %d's data.",
                self.increment,
            )
            new_data_splits = self.environment.step()

            logger.debug(
                'Fast forward: len(new_data_splits.train) = %d',
                0 if new_data_splits.train is None \
                    else len(new_data_splits.train),
            )
            logger.debug(
                'Fast forward: len(new_data_splits.validate) = %d',
                0 if new_data_splits.validate is None \
                    else len(new_data_splits.validate),
            )
            logger.debug(
                'Fast forward: len(new_data_splits.test) = %d',
                0 if new_data_splits.test is None \
                    else len(new_data_splits.test),
            )
            if self.feedback_type == 'oracle':
                # 3. Opt. Feedback on this step's new data
                logger.info(
                    'Fast forward: '
                    "Requesting feedback (%s) for step %d's data.",
                    self.feedback_type,
                    self.increment - 1,
                )
                new_data_splits = self.environment.feedback(new_data_splits)

                logger.info(
                    'Fast forward: '
                    "Updating with feedback (%s) for step %d's data.",
                    self.feedback_type,
                    self.increment - 1,
                )
                if self.experience:
                    # Add new data to experience
                    self.experience = self.experience.append(new_data_splits)

                    logger.debug(
                        'Fast forward: len(self.experience.train) = %d',
                        0 if self.experience.train is None else \
                            len(self.experience.train),
                    )
                    logger.debug(
                        'Fast forward: len(self.experience.validate) = %d',
                        0 if self.experience.validate is None else \
                            len(self.experience.validate),
                    )
                    logger.debug(
                        'Fast forward: len(self.experience.test) = %d',
                        0 if self.experience.test is None else \
                            len(self.experience.test),
                    )


class KineticsOWLExperiment(object):
    """The Dataloading and handling for a Kinetics OWL experiment.

    Attributes
    ----------
    experience : list = None
        The prior history experienced within the experiment. This will be the
        indices experienced per each KineticsUnifiedFeatures object. so this
        will be a list of lists of indices, where the first index is for the
        ordered KineticsUnifiedFeatures obejcts in order of apperance, and each
        of those lists will contain the indices in the order they were obtained
        by sample.
    start : DataSplits
        The starting increment's data as a KineticsUnifiedFeatures object.

        huh... docstr does not CAP gen on MultiType ... | KineticsUnified
    steps : arn.data.docstr_workarounds.get_steps = None
        List of DataSplits containing the source KineticsUnifiedFeature objects
        representing the order to increment over them.

        Each step has the evaluator's (oracle's) knowledge of the labels. The
        predictor's known label encoder is managed elsewhere, preferably within
        the predictor object as label_enc.
    _inc_splits_per_dset : int = 5
        The number of incremental splits per dataset. Set this to default to
        5 given the K600 dataset introduced very few samples relative to K400:
        1056 new samples in validation and 4359 new samples in test.
    _increment : int = 0
        The current increment of the experiment. Starts at zero, increments
        after a step is complete. After initial increment is increment = 1.
    """
    def __init__(
        self,
        start,
        steps=None,
        inc_splits_per_dset=5,
        intro_freq_first=False,
        seed=0,
        repeat_samples=False,
    ):
        """Initialize the Kinetics Open World Learning Experiment.

        Args
        ----
        start : see self
        steps : see self
        inc_splits_per_dset : see self _inc_splits_per_dset
        intro_freq_first : bool = False
            see get_increments
        seed : int = 0
            The seed for the random number generator
        repeat_samples: bool = False
            If False, the default, the loading of future datasets removes prior
            seen samples based on their unique identifier. If True, no checking
            or removal of samples is done.
        """
        self._increment = 0
        self._inc_splits_per_dset = inc_splits_per_dset
        self.start = start

        # NOTE possible that experience should be in the environment/experiment
        # rather than the simulation, but this is an abstraction/semantics
        # issue that doesn't affect practical end result.
        logger.debug(
            'train n_classes = %d, len(start.train) = %d',
            len(start.train.label_enc),
            len(start.train),
        )
        logger.debug(
            'validate n_classes = %d, len(start.validate) = %d',
            len(start.validate.label_enc),
            len(start.validate),
        )
        logger.debug(
            'test n_classes = %d, len(start.test) = %d',
            len(start.test.label_enc),
            len(start.test),
        )

        if steps is None:
            self.steps = steps
        else:
            if repeat_samples:
                prior_sample_uids = None
            else:
                prior_sample_uids = set()
                for split in ['train', 'validate', 'test']:
                    split = getattr(self.start, split)
                    if split is not None:
                        prior_sample_uids.update(
                            get_filename(split.data, ext=None)
                        )
            self.steps = []
            known_label_enc = deepcopy(self.start.train.label_enc)
            for i, step in enumerate(steps):
                if prior_sample_uids:
                    # rm any prior seen samples.
                    for step_split in ['train', 'validate', 'test']:
                        split = getattr(step, step_split)
                        if split is None:
                            logger.warning(
                                'Source steps[%d], split %s is None',
                                i,
                                step_split,
                            )
                            continue
                        step_uids = get_filename(split.data, ext=None)
                        mask = step_uids.isin(prior_sample_uids)
                        if mask.any():
                            logger.info(
                                "Prior seen samples (total %d of %d) exist in steps["
                                "%d]'s dsplit %s of source dataset.",
                                mask.sum(),
                                len(mask),
                                i,
                                step_split,
                            )
                            logger.debug(
                                'prior seen samples in steps[%d] dsplit %s:\n'
                                '%s',
                                i,
                                step_split,
                                step_uids[mask],
                            )

                            # Drop the indices from step.
                            split.data = split.data[~mask]

                            # Add new labels to prior_sample_uids
                            prior_sample_uids.update(step_uids)

                self.steps += get_increments(
                    inc_splits_per_dset,
                    step,
                    known_label_enc,
                    seed=seed + i,
                    intro_freq_first=intro_freq_first,
                    #label_col=step.label_col,
                )

        # TODO given resulting steps, split up a list of DataSplits into the
        # same order, and then concat to end of DataFrame, such that it will
        # load the different files as visual transforms of those samples.

        # TODO For simplicity and to avoid re-evaluating the original data,
        # replace the orignal with the visual transforms.

    @property
    def increment(self):
        """The current increment or steps taken."""
        return self._increment

    @property
    def increments_per_dataset(self):
        return self._inc_splits_per_dset

    @property
    def total_increments(self):
        """Start increment + src steps * increments per dataset"""
        if self.steps:
            return 1 + len(self.steps)
        return 1

    # NOTE def reset(self, state):

    def feedback(self, data_splits, test=False):
        """Feedback request from agent. For implementation simplicty for 100%
        oracle feedback, e.g., every sample requested gets the ground truth
        label, this just provides labels with the sample in dataset.
        """
        # Oracle, exhaustive, no budget : labels are simply provided.
        if data_splits.train and not data_splits.train.return_label:
            data_splits.train.return_label = True
            begin_unk_idx = len(data_splits.train.label_enc)
        else:
            begin_unk_idx = None
        if data_splits.validate and not data_splits.validate.return_label:
            data_splits.validate.return_label = True
            data_splits.validate.begin_unk_idx = begin_unk_idx
        if test and data_splits.test and not data_splits.test.return_label:
            data_splits.test.return_label = True
            data_splits.test.begin_unk_idx = begin_unk_idx
        return data_splits

    def step(self):
        """An incremental step's train, val, and test dataloaders?

        Returns
        -------
        DataSplits
            A NamedTuple of a Torch Dataset objects for the current increment's
            new data
        """
        # NOTE Manage the location of data and tensors to avoid memory issues.
        if self.increment == 0:
            self._increment += 1
            return self.start
        if self.increment >= self.total_increments:
            logger.info('Experiment Complete: step datasets exhausted.')
            return None
        data = self.steps[self.increment - 1]
        self._increment += 1
        return data
