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
#TODO KU: feature extraction loading instead of images/vids, pair w/ labels.
from arn.data.kinetics_unified import KineticsUnified, KineticsUnifiedFeatures
from arn.models.owhar import OWHAPredictor


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

    def step(self, state):
        """The incremental step in incremental learning of Kinetics OWL."""
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
    start : KineticsUnifiedFeatures
        The starting increment's data as a KineticsUnifiedFeatures object.

        huh... docstr does not CAP gen on MultiType ... | KineticsUnified
    step : list = None
        List of KineticsUnifiedFeature objects representing the order to
        increment over them.

        ? Is this suspposed to be the other two datasets that get
        incrementally stepped over? a list of them in order? so 2 different
        KUFs for k600 and k700?
    _inc_splits_per_dset : int = 10
        The number of incremental splits per dataset.
    """
    def __init__(
        self,
        start,
        step=None,
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
        step : see self
        inc_splits_per_dset : see self _inc_splits_per_dset
        seed : int = None
            The seed for the random number generator
        """
        self._inc_splits_per_dset = inc_splits_per_dset
        self.start = start
        self.step = step

        # TODO make appropriate call to KineticsUnified, o.w. implement here

        #if maintain_predictor_experience:
        #    # TODO, create an experience DataLoader that combines the
        #    # dataloaders or subsets of them from past increments.
        #    raise NotImplementedError()
        #    self.experience = None
        #else:
        #    self.experience = None
        self.experience = None

    @property
    def num_increments(self):
        raise NotImplementedError()
        return self._inc_splits_per_dset

    def step(self):
        """An incremental step's train, val, and test dataloaders?

        Returns
        -------
        (object, object=None)
            A tuple of a Torch dataloader object for the specified increment's
            new data, and an optional other dataloader for the history of data
            the predictor recalls. The latter being a convenience where it is
            the predictor's stored experience object
        """
        raise NotImplementedError()
        # TODO Manage the location of data and tensors to avoid memory issues.
