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
from dataclasses import dataclass, InitVar
import logging
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
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
    # TODO should make this optionally save to a database, like PostgreSQL.
    pred_dir: str = None
    eval_dir: str = None
    file_prefix: str = ''
    save_preds_with_labels: bool = True

    def __bool__(self):
        return bool(self.pred_dir) or bool(self.eval_dir)

    def eval(self, data_split, preds, measures, prefix=None):
        prefix = os.path.join(prefix, self.file_prefix)
        labels = None

        if self.pred_dir:
            if isinstance(preds, torch.Tensor):
                preds = preds.numpy()
            if self.save_preds_with_labels:
                labels = [row[1] for row in data_split]
                if len(preds.shape) == 2 and len(preds.shape[1]) > 1:
                    contents = np.hstack(labels, preds)
                else:
                    contents = [labels, preds]
                pd.DataFrame(
                    contents,
                    columns=['target_labels', 'preds'],
                ).to_csv(
                    create_filepath(os.path.join(prefix, 'preds.csv')),
                    index=False,
                )
            else:
                pd.DataFrame(
                    preds,
                    columns=['preds'],
                ).to_csv(
                    create_filepath(os.path.join(prefix, 'preds.csv')),
                    index=False,
                )

        if self.eval_dir:
            if isinstance(preds, torch.Tensor):
                preds = preds.numpy()
            if labels is None:
                labels = [row[1] for row in data_split]

            for measure in measures:
                if issubclass(
                    measurements,
                    (ConfusionMatrix, OrderedConfusionMatrices),
                ):
                    measurements = measure(labels, preds, data_split.label_enc)
                    measurements.save(os.path.join(prefix, 'preds.csv'))
                else:
                    raise NotImplementedError('TODO: non-confusion matrix.')
                    measurements = measure(labels, preds)
                    # TODO scalars? store in dict?
                    # TODO Tensorboard hook?


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
    measures : list = 'ordered_confusion_matrix'
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

    def eval(self, data_splits, predict, measures, prefix=None):
        """Given the datasplits, performs the predictions and evaluations to
        be saved.

        Args
        ----
        data_splits : DataSplits
            The data splits to potentially be predicted on and evaluated.
        predict : Callable
            A function of the predictor to perform predictions given a dataset
            within the data_splits object.
        measures : list
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

        if data_splits.train is not None and self.train:
            logging.info("Predicting `label` for `%s`'s train.", prefix)
            self.train.eval(
                data_splits.train,
                predict(data_splits.train),
                self.measures,
                os.path.join(prefix, 'train_'),
            )
        if data_splits.validate is not None and self.validate:
            logging.info("Predicting `label` for `%s`'s val.", prefix)
            self.validate.eval(
                data_splits.validate,
                predict(data_splits.validate),
                self.measures,
                os.path.join(prefix, 'validate_'),
            )
        if data_splits.test is not None and self.test:
            logging.info("Predicting `label` for `%s`'s test.", prefix)
            self.test.eval(
                data_splits.test,
                predict(data_splits.test),
                self.measures,
                os.path.join(prefix, 'test_'),
            )


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

    Attributes
    ----------
    environment : KineticsOWLExperiment
    predictor : OWHAPredictor
    feedback : str = 'oracle'
    rng_state : int = None
        Random seed.
    eval_on_start : bool = False
        If False (the default), does not evaluate an untrained predictor. If
        True, evaluated an untrained predictor. May be a good idea to evaluate
        some untrained predictors, espcially if they were pre-trained.
    experience : DataSplits = None
        If `maintain_experience` is True in __init__, then the simulation
        maintains the past experienced data for the predictor.
    tasks : str | list = None
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
        tasks=None,
        maintain_experience=False,
        labels=None
        # configure logging ...
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

        # TODO will have to change this if handling multi-tasks in same
        # experiment!
        # TODO handle datasets' label encs when it is set explicitly here?
        if labels is None:
            self.label_enc = self.environment.start.train.label_enc
        elif isinstance(labels, str):
            self.label_enc = NominalDataEncoder(load_file_list(labels))
        elif isinstance(labels, list):
            self.label_enc = NominalDataEncoder(labels)
        else:
            raise TypeError(
                f'subset.labels.known unexpected type! {type(labels)}'
            )

        #if tasks is None:
        #     # NOTE support this in predictor and the datasets in labels
        #     #returned!
        #    self.tasks = ['labels', 'detect']

        # Maintain experience here for the predictor
        if maintain_experience:
            self.experience = DataSplits()
        else:
            self.experience = None

        # TODO callbacks or hooks would be wonderful for saving predictions and
        # eval measures!

    @property
    def increment(self):
        return self.environment.increment

    def step(self, state=None):
        """The incremental step in incremental learning of Kinetics OWL."""
        # 1. Get new data (input samples only)
        logging.info("Getting step %d's data.", self.increment + 1)
        new_data_splits = self.environment.step()

        # 2. Inference/Eval on new data if self.eval_untrained_start
        if (self.increment == 0 and self.eval_on_start) or self.increment > 0:
            # NOTE Predict for the Task(s), useful when multiple tasks to be
            # handled by one predictor.
            #for task_id in self.tasks:
            #    pass

            self.pred_eval.eval(
                new_data_splits,
                self.predictor.predict,
                self.measures,
                #prefix,
            )
            #pred = self.predictor.predict(new_data_splits.test)
            #measures = self.environment.eval(new_data_splits, pred, 'labels')

            self.pred_eval.eval(
                new_data_splits,
                self.predictor.novelty_detect,
                self.measures,
                #prefix,
            )
            logging.info(
                "Predicting `novelty_detection` for step %d's data.",
                self.increment,
            )
            detects = self.predictor.novelty_detect(new_data_splits)
            detect_measures = self.environment.eval(
                new_data_splits,
                detects,
                'novelty_detect',
            )

        if self.feedback == 'oracle':
            # 3. Opt. Feedback on new data
            logging.info(
                "Requesting feedback ({self.feedback}) for step %d's data.",
                self.increment,
            )
            new_data_splits = self.environment.feedback(new_data_splits)

            if self.experience:
                # Add new data to experience
                self.experience.update(new_data_splits)

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

            # TODO 5. Opt. Predictor eval post update
            measrures_post_feedback = self.environment.eval(
                new_data_splits,
                pred,
                'labels',
            )
            detect_measrures_post_feedback = self.environment.eval(
                new_data_splits,
                pred,
                'novelty_detect',
            )

            # TODO 6. Opt. Evaluate the updated predictor on entire experience
            #self.eval(self.experience, self.predictor.predict(self.experience))

    def run(self, max_steps=None, tqdm=None):
        """The entire experiment run loop."""
        for i in range(self.environment.total_increments):
            logging.info("Starting this run's step: %d", i + 1)
            logging.info("Increment: %d", self.increment + 1)
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
    steps : list = None
        List of DataSplits containing KineticsUnifiedFeature objects
        representing the order to increment over them.

        ? Is this suspposed to be the other two datasets that get
        incrementally stepped over? a list of them in order? so 2 different
        KUFs for k600 and k700?
    _inc_splits_per_dset : int = 10
        The number of incremental splits per dataset.
    _increment : int = 0
        The current increment of the experiment. Starts at zero, increments
        after a step is complete. After initial increment is increment = 1.
    label_encoder : exputils.data.labels.NominalDataEncoder
        Keep the labels consistent at the current step.
    measures : list = None
        String identifier of a measure or a callable that takes as input
        (target, predictions) and returns a measurement or measurement
        object, e.g., confusion matrix.
    """
    def __init__(
        self,
        start,
        steps=None,
        inc_splits_per_dset=10,
        measures=None,
        seed=None,
    ):
        """Initialize the Kinetics Open World Learning Experiment.

        Args
        ----
        start : see self
        steps : see self
        inc_splits_per_dset : see self _inc_splits_per_dset
        measures : see self
        seed : int = None
            The seed for the random number generator
        """
        self._increment = 0
        self._inc_splits_per_dset = inc_splits_per_dset
        self.start = start
        self.steps = steps

        # TODO LabelEncoder with the ability to be given a specific list of
        # labels as knowns such that the order is correctomundo.

        # NOTE possible that experience should be in the environment/experiment
        # rather than the simulation, but this is an abstraction/semantics
        # issue that doesn't affect practical end result.

    @property
    def increment(self):
        """The current increment or steps taken."""
        return self._increment

    @property
    def increments_per_dataset(self):
        return self._inc_splits_per_dset

    @property
    def total_increments(self):
        """Start increment + steps * increments per dataset in steps"""
        if self.steps:
            return 1 + len(self.steps) * self.increments_per_dataset
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

    def eval(self, dataset, pred, task='labels'):
        # TODO evaluate the given dataset on all measures.

        # TODO novelty detect task is based on the NominalDataEncoder for the
        # current time step as it knows when something is a known or unknown
        # class at the current time step.

        # Optional log/save predictions or eval measures
        # 'labels' is the 'classify' task
        raise NotImplementedError('KineticsOWLExperiment eval()')

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

        raise NotImplementedError('Stepping increments through step datasets.')
