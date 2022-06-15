"""Kinetics Open World Learning pipeline object.

Environment: incremental learning experiment over Kinetics data
Predictor: arn.owhar.OWHARPredictor
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
)
from arn.models.owhar import OWHAPredictor

from exputils.data.labels import NominalDataEncoder
from exputils.data.confusion_matrix import ConfusionMatrix
from exputils.data.ordered_confusion_matrix import OrderedConfusionMatrices
from exputils.io import create_filepath

import logging
logger = logging.getLogger(__name__)


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

    # Stratified folds for known class samples, ensuring presence and balance
    # across incs
    skf = StratifiedKFold(
        n_increments,
        random_state=seed,
        shuffle=np_gen is not None,
    )
    known_incs = [
        known_df.iloc[test] for train, test in
        skf.split(known_df['sample_index'], known_df[label_col])
    ]

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

    knowns_splits = []
    unknowns_splits = []
    for tmp_dset in src_datasplit:
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

    # TODO randomization criteron that forces certain classes to be separate,
    # e.g. classes farther from each other in some class hierarchy should be
    # separated across the increments such that there is sufficiently novel
    # classes in each increment.

    # NOTE assumes train contains all unknown classes.
    unknown_df = unknowns_splits[0]

    # Randomize unique, unknown classes across n increments. Last w/ remainder
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

    # Loop through and create the incremental KineticsUnified datasets
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
            inc_dataset = deepcopy(tmp_dataset)
            inc_dataset.label_enc = deepcopy(label_enc)

            # Get all of the unknowns introduced at this increment
            unks = unknown_df[unknown_df[label_col].isin(inc_uniques)]
            logger.debug(
                'increment %d: split %d: total new unknown samples to be '
                'spread across remaining increments = %d',
                i,
                k,
                len(unks),
            )

            # Stratified shuffle split across this and remaining incs
            if persistent_unks is not None:
                unknown_incs = [
                    unks.iloc[test] for train, test in
                    persist_unk_skf.split(unks['sample_index'], unks[label_col])
                ]
                logger.debug(
                    'len(unknown_incs) at i=%d, k=%d: %d; remainder = %d',
                    i,
                    k,
                    len(unknown_incs),
                    remainder,
                )
                persistent_unk_df = unknown_incs.pop(0)
            else:
                # Handle remainder == 1 case, no stratified splitting
                persistent_unk_df = unks

            for j in range(i):
                if persistent_unknowns[j][k]: # not None or not an empty stack
                    persistent_unk_df = persistent_unk_df.append(
                        persistent_unknowns[j][k].pop(0)
                    )

            # Save this increment's stack of persistent unknowns across incs
            if persistent_unks is not None:
                persistent_unks[k] = unknown_incs

            logger.debug(
                'increment %d: split %d: unknown samples = %d',
                i,
                k,
                len(persistent_unk_df),
            )
            logger.debug(
                'increment %d: split %d: known samples = %d',
                i,
                k,
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


def get_steps(step_1, step_2):
    """Hotfix for docstr to load in 2 KineticsUnified datasets. TODO update
    docstr to parse and generate CAPs for types within lits when specified.

    Args
    ----
    step_1 : DataSplits
        The first DataSplit of KineticsUnified Datasets
    step_2 : DataSplits
        The second DataSplit of KineticsUnified Datasets

    Returns
    -------
    list
        List of step 1 and step 2
    """
    return [step_1, step_2]


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

    def eval(self, data_split, preds, measures, prefix=None):
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

        if isinstance(preds, torch.Tensor):
            if preds.device.type == 'cuda':
                preds = preds.cpu().numpy()
            else:
                preds = preds.numpy()

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
                    )
                    contents = [labels, preds]

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

                pd.DataFrame(
                    contents,
                    columns=['target_labels'] + list(data_split.label_enc),
                ).to_csv(
                    create_filepath(os.path.join(prefix, 'preds.csv')),
                    index=False,
                )
            else:
                # TODO single class prediction case ['pred']
                # TODO verify the datasplit and predictor encoders have the
                # same order for known classes. Perhaps check after every fit.
                pd.DataFrame(
                    preds,
                    columns=list(data_split.label_enc),
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

            for measure in measures:
                if issubclass(measure, ConfusionMatrix):
                    measurements = measure(labels, preds, data_split.label_enc)
                    measurements.save(os.path.join(prefix, 'preds_cm.csv'))
                elif issubclass(measure, OrderedConfusionMatrices):
                    measurements = measure(
                        labels,
                        preds,
                        data_split.label_enc,
                        5,
                    )
                    if logging.root.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            'OrderedConfusionMatrices: top-5 accuracy: %.4f',
                            measurements.accuracy(5),
                        )
                        logger.debug(
                            'OrderedConfusionMatrices: top-1 accuracy: %.4f',
                            measurements.accuracy(),
                        )
                        cm = measurements.get_conf_mat()
                        logger.debug(
                            'OrderedConfusionMatrices (CM): top-1 '
                            'accuracy: %.4f',
                            cm.accuracy(),
                        )
                        logger.debug(
                            'OrderedConfusionMatrices (CM): top-1 MCC: %.4f',
                            cm.mcc(),
                        )
                        logger.debug(
                            'OrderedConfusionMatrices (CM): top-1 '
                            'Mutual Information: %.4f',
                            cm.mutual_information(),
                        )
                        logger.debug(
                            'OrderedConfusionMatrices (CM): top-1 '
                            'Arithmetic Normalized Mutual Information: %.4f',
                            cm.mutual_information('arithmetic'),
                        )
                    measurements.save(os.path.join(prefix, 'preds_top-cm.h5'))
                else:
                    raise NotImplementedError('TODO: non-confusion matrix.')
                    measurements = measure(labels, preds)
                    # TODO scalars? store in dict?
                    # TODO Tensorboard hook?
                    # TODO some callable object: scalar, tensors, etc.
                    #   Simply take the callable object's name if available.


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
    """
    train: EvalDataSplitConfig = None
    validate: EvalDataSplitConfig = None
    test: EvalDataSplitConfig = None
    root_dir: str = ''
    measures: InitVar[list] = 'ordered_confusion_matrix'

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

    def eval(self, data_splits, predict, prefix=None):
        """Given the datasplits, performs the predictions and evaluations to
        be saved.

        Args
        ----
        data_splits : DataSplits
            The data splits to potentially be predicted on and evaluated.
        predict : Callable
            A function of the predictor to perform predictions given a dataset
            within the data_splits object.
        prefix : str = None
            An optoinal prefix to add to the paths AFTER the root_dir. This
            would be useful for adding the step number and phase of that step,
            such as if inference on new unlabeled data, or inference on data
            after feedback update.
        """
        if prefix:
            prefix = os.path.join(self.root_dir, prefix)
        else:
            prefix = self.root_dir

        # NOTE relies on predictor to turn data_split into a DataLoader

        for name, dsplit in data_splits._asdict().items():
            if dsplit is not None and self.train:
                logger.info("Predicting `label` for `%s`'s %s.", prefix, name)

                reset_return_label = dsplit.return_label
                if dsplit.return_label:
                    dsplit.return_label = False
                preds = predict(dsplit)
                n_classes = len(dsplit.label_enc)
                if preds.shape[1] < n_classes:
                    # Relies on data split label enc including all prior known
                    # classes the predictor has seen.
                    preds = np.hstack((
                        preds,
                        np.zeros([preds.shape[0], n_classes - preds.shape[1]]),
                    ))
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
                    os.path.join(prefix, f'{name}'),
                )
                dsplit.return_label = reset_return_label


class DataSplits(NamedTuple):
    """Contains the KineticsUnifiedFeatures for train, validate, and test

    Attributes
    ----------
    train: KineticsUnifiedFeatures = None
    validate: KineticsUnifiedFeatures = None
    test: KineticsUnifiedFeatures = None
    """
    train: KineticsUnifiedFeatures = None
    validate: KineticsUnifiedFeatures = None
    test: KineticsUnifiedFeatures = None

    def update(self, data_split):
        """Given data_split update internal data_split."""
        # Most basic is concat new data splits to end of current one.
        # TODO but what about their metadata? how is that accesible from this?
        if data_split.train:
            if self.train is not None:
                self.train = torch.utils.data.ConcatDataset(
                    [self.train, data_split.train],
                )
            else:
                self.train = data_split.train

        if data_split.validate:
            if self.validate is not None:
                self.validate = torch.utils.data.ConcatDataset(
                    [self.validate, data_split.validate],
                )
            else:
                self.validate = data_split.validate

        if data_split.test:
            if self.test is not None:
                self.test = torch.utils.data.ConcatDataset(
                    [self.test, data_split.test],
                )
            else:
                self.test = data_split.test

        # TODO support check for repeat or non-unique sample ids, which then
        # would mean to update those prior experiences.


class KineticsOWL(object):
    """Kinetics Open World Learning Pipeline for incremental recognition.
    This is the class that contains all other objects to run an experiment
    for Kinetics, including the Kinetics datasets, the predictor, and
    evaluation code. This serves as the glue between them all.

    Attributes
    ----------
    environment : KineticsOWLExperiment
    predictor : OWHAPredictor
        predictor : arn.models.owhar.EVMPredictor
        arn.models.owhar.load_evm_predictor
        TODO docstr: support at least basic checking of multiple configurable
        types. Or maybe just parse all of them as options and support so in
        MultiType.
    feedback : str = 'oracle'
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
        feedback='oracle',
        rng_state=None,
        measures=None,
        #inc_splits_per_dset : 10
        eval_on_start=False,
        eval_config=None,
        post_feedback_eval_config=None,
        tasks=None,
        maintain_experience=False,
        labels=None,
        # configure state saving ...
    ):
        """Initialize the KineticsOWL experiment.

        Args
        ----
        environment : see self
        predictor : see self
        feedback : see self
        rng_state : see self
        eval_on_start : see self
        eval_config : see self
        post_feedback_eval_config : see self
        tasks : see self
        maintain_experience : bool = False
            If False, the default, the past experienced samples are not saved
            in the simulation for use by the predictor. Otherwise, the
            experienced samples are saved by concatenating the new data splits
            to the end of the prior ones.
        labels : str = None
        """
        # TODO handle seed/rng_state if given, otherwise randomly select seed.
        self.rng_state = rng_state

        self.environment = environment
        self.predictor = predictor
        self.feedback = feedback
        self.eval_on_start = eval_on_start
        self.eval_config = eval_config
        if post_feedback_eval_config is True:
            self.post_feedback_eval_config = eval_config
        else:
            self.post_feedback_eval_config = post_feedback_eval_config

        # TODO will have to change this if handling multi-tasks in same
        # experiment!
        # TODO handle datasets' label encs when it is set explicitly here?
        """
        if labels is None:
            self.label_enc = deepcopy(self.environment.)
        if isinstance(labels is None:
            self.label_enc = NominalDataEncoder(load_file_list(labels))
        elif isinstance(labels, list):
            self.label_enc = NominalDataEncoder(labels)
        else:
            raise TypeError(
                f'subset.labels.known unexpected type! {type(labels)}'
            )
        """

        #if tasks is None:
        #     # NOTE support this in predictor and the datasets in labels
        #     #returned!
        #    self.tasks = ['labels', 'detect']

        # Maintain experience here for the predictor
        if maintain_experience:
            self.experience = DataSplits()
        else:
            self.experience = None

    @property
    def increment(self):
        return self.environment.increment

    def step(self, state=None):
        """The incremental step in incremental learning of Kinetics OWL."""
        # 1. Get new data (input samples only)
        logger.info("Getting step %d's data.", self.increment + 1)
        new_data_splits = self.environment.step()

        logger.debug(
            'len(new_data_splits.train) = %d',
            len(new_data_splits.train),
        )
        logger.debug(
            'len(new_data_splits.validate) = %d',
            len(new_data_splits.validate),
        )
        logger.debug(
            'len(new_data_splits.test) = %d',
            len(new_data_splits.test),
        )

        # 2. Inference/Eval on new data if self.eval_untrained_start
        if (self.increment == 1 and self.eval_on_start) or self.increment > 1:
            # NOTE Predict for the Task(s), useful when multiple tasks to be
            # handled by one predictor.
            #for task_id in self.tasks:
            #    pass

            logger.info(
                "Eval for new data, no feedback, for step %d.",
                self.increment,
            )
            self.eval_config.eval(
                new_data_splits,
                self.predictor.predict,
                f'step-{self.increment}_new-data_predict',
            )
            """ TODO Novelty Detect
            self.eval_config.eval(
                new_data_splits,
                self.predictor.novelty_detect,
                f'step-{self.increment}_new-data_novelty-detect',
            )
            #"""
            # TODO novelty detect task is based on the NominalDataEncoder for
            # the current time step as it knows when something is a known or
            # unknown class at the current time step.
            #   Keep experience/datasplit label encoders in sync.
            #   Use proper novelty detection measures of performance!
            #       - Confusion Matrix : which class is confused w/ novelty?
            #       - Difference to actual novelty occurrence (by sample idx)
            #           If early detection, negative, otherwise positive.

        if self.feedback == 'oracle':
            # 3. Opt. Feedback on this step's new data
            logger.info(
                "Requesting feedback (%s) for step %d's data.",
                self.feedback,
                self.increment,
            )
            new_data_splits = self.environment.feedback(new_data_splits)

            logger.info(
                "Updating with feedback (%s) for step %d's data.",
                self.feedback,
                self.increment,
            )
            if self.experience:
                # Add new data to experience
                self.experience.update(new_data_splits)

                logger.debug(
                    'len(self.experience.train) = %d',
                    len(self.experience.train),
                )
                logger.debug(
                    'len(self.experience.validate) = %d',
                    len(self.experience.validate),
                )
                logger.debug(
                    'len(self.experience.test) = %d',
                    len(self.experience.test),
                )

                # 4. Opt. Predictor Update/train on new data w/ feedback
                self.predictor.fit(
                    self.experience.train,
                    self.experience.validate,
                )
            else:
                self.predictor.fit(
                    new_data_splits.train,
                    new_data_splits.validate,
                )

            logger.info(
                "Post-feedback Eval (%s) for step %d.",
                self.feedback,
                self.increment,
            )
            # 5. Opt. Predictor eval post update
            self.post_feedback_eval_config.eval(
                new_data_splits,
                self.predictor.predict,
                f'step-{self.increment}_post-feedback_predict',
            )
            """ TODO Novelty Detect
            self.post_feedback_eval_config.eval(
                new_data_splits,
                self.predictor.novelty_detect,
                f'step-{self.increment}_post-feedback_novelty-detect',
            )
            #"""

            # TODO 6. Opt. Evaluate the updated predictor on entire experience

    def run(self, max_steps=None, tqdm=None):
        """The entire experiment run loop."""
        for i in range(self.environment.total_increments):
            logger.info("Starting this run's step: %d", i + 1)
            logger.info("Increment: %d", self.increment + 1)
            self.step()


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
    steps : get_steps = None
        List of DataSplits containing the source KineticsUnifiedFeature objects
        representing the order to increment over them.

        Each step has the evaluator's (oracle's) knowledge of the labels. The
        predictor's known label encoder is managed elsewhere, preferably within
        the predictor object as label_enc.
    _inc_splits_per_dset : int = 10
        The number of incremental splits per dataset.
    _increment : int = 0
        The current increment of the experiment. Starts at zero, increments
        after a step is complete. After initial increment is increment = 1.
    """
    def __init__(
        self,
        start,
        steps=None,
        inc_splits_per_dset=10,
        intro_freq_first=False,
        seed=None,
    ):
        """Initialize the Kinetics Open World Learning Experiment.

        Args
        ----
        start : see self
        steps : see self
        inc_splits_per_dset : see self _inc_splits_per_dset
        intro_freq_first : bool = False
            see get_increments
        seed : int = None
            The seed for the random number generator
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
            'validate n_classes = %d, len(start.train) = %d',
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
            self.steps = []
            known_label_enc = deepcopy(self.start.train.label_enc)
            for i, step in enumerate(steps):
                self.steps += get_increments(
                    inc_splits_per_dset,
                    step,
                    known_label_enc,
                    seed=i,
                    intro_freq_first=intro_freq_first,
                )
                # TODO Opt. Clean val sets to rm samples from val if in prior
                # train.

                # TODO Opt. Clean test sets to rm samples from test if in prior
                # train/val.


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

    # TODO def reset(self, state):

    def feedback(self, data_splits, test=False):
        # Oracle, exhaustive, no budget : labels are simply provided.
        if data_splits.train and not data_splits.train.return_label:
            data_splits.train.return_label = True
        if data_splits.validate and not data_splits.validate.return_label:
            data_splits.validate.return_label = True
        if test and data_splits.test and not data_splits.test.return_label:
            data_splits.test.return_label = True
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
            raise ValueError('Experiment Complete: step datasets exhausted.')
        self._increment += 1
        return self.steps[self.increment - 1]


# TODO the following is all a workaround for the current docstr prototype to
# support the ease of swapping predictors by changing the config only, not the
# doc strings of KineticsOWL. This is what happens when reseach code meets
# prototype code.
def kinetics_owl_evm(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL.__init__
    predictor : EVMPredictor
    feedback : see KineticsOWL.__init__
    rng_state : see KineticsOWL.__init__
    eval_on_start : see KineticsOWL.__init__
    eval_config : see KineticsOWL.__init__
    post_feedback_eval_config : see KineticsOWL.__init__
    tasks : see KineticsOWL.__init__
    maintain_experience : see KineticsOWL.__init__
    labels : see KineticsOWL.__init__
    """
    return Kinetics_OWL(*args, **kwargs)


def kinetics_owl_owhapredictor_evm(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL.__init__
    predictor : OWHAPredictorEVM
    feedback : see KineticsOWL.__init__
    rng_state : see KineticsOWL.__init__
    eval_on_start : see KineticsOWL.__init__
    eval_config : see KineticsOWL.__init__
    post_feedback_eval_config : see KineticsOWL.__init__
    tasks : see KineticsOWL.__init__
    maintain_experience : see KineticsOWL.__init__
    labels : see KineticsOWL.__init__
    """
    return Kinetics_OWL(*args, **kwargs)
