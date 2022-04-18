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
from typing import NamedTuple

import torch

from arn.data.kinetics_unified import KineticsUnified, KineticsUnifiedFeatures
from arn.models.owhar import OWHAPredictor

from exputils.data.labels import NominalDataEncoder


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
    feedback : str = None
    rng_state : int = None
        Random seed.
    measures : list = None
        String identifier of a measure or a callable that takes as input
        (target, predictions) and returns a measurement or measurement
        object, e.g., confusion matrix.
    eval_on_start : False
        If False (the default), does not evaluate an untrained predictor. If
        True, evaluated an untrained predictor. May be a good idea to evaluate
        some untrained predictors, espcially if they were pre-trained.
    experience : DataSplits = None
        If `maintain_experience` is True in __init__, then the simulation
        maintains the past experienced data for the predictor.
    """
    def __init__(
        self,
        environment,
        predictor,
        #augmentation=None # sensors
        feedback=None,
        rng_state=None,
        measures=None,
        #inc_splits_per_dset : 10
        eval_on_start=False,
        maintain_experience=False,
        task_ids=None,
    ):
        """Initialize the KineticsOWL experiment.

        Args
        ----
        see self
        """
        # TODO handle seed/rng_state if given, otherwise randomly select seed.
        self.rng_state = rng_state

        # TODO init the Kinetics data incremental loader(s)
        self.environment = environment
        self.predictor = predictor
        self.feedback = feedback
        self.measures = measures
        self.eval_on_start = eval_on_start

        if task_ids is None:
             #TODO support this in predictor and the datasets in labels
             #returned!
            self.task_ids = ['labels', 'detect']

        # Maintain experience here for the predictor
        if maintain_experience:
            self.experience = DataSplits()
        else:
            self.experience = None

    @property
    def increment(self):
        self.environment.increment

    def step(self, state):
        """The incremental step in incremental learning of Kinetics OWL."""
        # 2. Inference/Eval on new data if self.eval_untrained_start
        if self.increment == 0 and self.eval_on_start:
            # 1. Get new data (input samples only)
            new_data_splits = self.environment.step()

            # TODO Novelty Detection for the Task

            # TODO Predict for the Task
            pred = self.predictor.predict(new_data_splits) # TODO data pass!

            # TODO Optional log/save predictions or eval measures
            self.eval(new_data_splits, pred)

        if self.feedback == 'oracle':
            # 3. Opt. Feedback on new data
            new_data_splits = self.environment.feedback(new_data_splits)

            if self.experience:
                # TODO Add new data to experience
                raise NotImplementedError('Added new data to experience')

                self.experience.update(new_data_splits)

            # TODO 4. Opt. Predictor Update/train on new data w/ feedback
            self.predictor.fit(self.experience)

            # TODO 5. Opt. Predictor eval post update
            self.eval(new_data_splits, pred)

            # TODO 6. Opt. Evaluate the updated predictor on entire experience
            #self.eval(self.experience, self.predictor.predict(self.experience))

            raise NotImplementedError()

    def eval(self, dataset, pred):
        # TODO evaluate the given dataset on all measures.
        raise NotImplementedError()

    def run(self, max_steps=None, tqdm=None):
        """The entire experiment run loop."""

        raise NotImplementedError()
        while increment := self.environment.step():
            # TODO Infer: Classify, novelty detect, novelty recognize
            inc_preds, inc_detects = self.predictor.predict(
                increment.features,
                feedback_budget=increment.feedback_budget,
            )

            # TODO Request feedback and update.
            if self.feedback:
                if self.predictor.feedback_query():
                    raise NotImplementedError()


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
    """
    def __init__(
        self,
        start,
        steps=None,
        inc_splits_per_dset=10,
        #feature_extract=True,
        #shared_dataloader_kwargs=None,
        #maintain_predictor_experience=True,
        seed=None,
    ):
        """Initialize the Kinetics Open World Learning Experiment.

        Args
        ----
        start : see self
        steps : see self
        inc_splits_per_dset : see self _inc_splits_per_dset
        seed : int = None
            The seed for the random number generator
        """
        self._increment = 0
        self._inc_splits_per_dset = inc_splits_per_dset
        self.start = start
        self.steps = steps

        # TODO need to create a dataset from the provided start and step
        # datasets.

        # Experience: train, val, test? inferred by label presence?

        #if maintain_predictor_experience:
        #    # TODO, create an experience DataLoader that combines the
        #    # dataloaders or subsets of them from past increments.
        #    raise NotImplementedError()
        #    self.experience = None
        #else:
        #    self.experience = None
        self.experience = None

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
        return 1 + len(self.steps) * self.increments_per_dataset

    def feedback(self, data_splits):
        # TODO Oracle, exhaustive, no budget : labels are simply provided.
        if data_splits.train and not data_splits.train.return_label:
            data_splits.train.return_label = True
        if data_splits.validate and not data_splits.validate.return_label:
            data_splits.validate.return_label = True
        if data_splits.test and not data_splits.test.return_label:
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
        # TODO Manage the location of data and tensors to avoid memory issues.
        if self.increment == 0:
            self._increment += 1
            return self.start
        if self.increment >= self.total_increments:
            raise ValueError('Experiment Complete: step datasets exhausted.')

        raise NotImplementedError('Stepping increments through step datasets.')
