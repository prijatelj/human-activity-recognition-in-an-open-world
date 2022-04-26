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
import logging
from typing import NamedTuple

import torch

from arn.data.kinetics_unified import (
    KineticsUnified,
    KineticsUnifiedFeatures,
    load_file_list,
)
from arn.models.owhar import OWHAPredictor

from exputils.data.labels import NominalDataEncoder
#from exputils.data.confusion_matrix import ConfusionMatrix


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

    @property
    def increment(self):
        return self.environment.increment

    def step(self, state=None):
        """The incremental step in incremental learning of Kinetics OWL."""
        # 1. Get new data (input samples only)
        logging.info("Getting step %d's data.", self.increment + 1)
        new_data_splits = self.environment.step()

        # 2. Inference/Eval on new data if self.eval_untrained_start
        if self.increment == 0 and self.eval_on_start:
            # NOTE Predict for the Task(s), useful when multiple tasks to be
            # handled by one predictor.
            #for task_id in self.tasks:
            #    pass

            # TODO data pass!
            logging.info(
                "Predicting `label` for step %d's data.",
                self.increment,
            )
            pred = self.predictor.predict(new_data_splits)
            self.environment.eval(new_data_splits, pred, 'labels')

            logging.info(
                "Predicting `novelty_detection` for step %d's data.",
                self.increment,
            )
            detect = self.predictor.novelty_detect(new_data_splits)
            self.environment.eval(new_data_splits, detect, 'novelty_detect')

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

                # TODO 4. Opt. Predictor Update/train on new data w/ feedback
                self.predictor.fit(self.experience)
            else:
                self.predictor.fit(
                    new_data_splits.train,
                    new_data_splits.validate,
                )

            # TODO 5. Opt. Predictor eval post update
            self.environment.eval(new_data_splits, pred)

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

    def feedback(self, data_splits):
        # Oracle, exhaustive, no budget : labels are simply provided.
        if data_splits.train and not data_splits.train.return_label:
            data_splits.train.return_label = True
        if data_splits.validate and not data_splits.validate.return_label:
            data_splits.validate.return_label = True
        #if data_splits.test and not data_splits.test.return_label:
        #    data_splits.test.return_label = True
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
