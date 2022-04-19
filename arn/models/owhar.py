"""Open World Human Activity Recognition pipeline class."""
import torch

from arn.models.novelty_detector import WindowedMeanKLDiv
from arn.torch_utils import torch_dtype


class OWHAPredictor(object):
    """The OWHAR predictor class that contains and manages the predictor parts.
    Every OWL predictor consists of a feature representation model, fine tuner,
    and novelty detector. Extra parts include a novelty recognizer if separate
    from the pre-existing parts and optional feedback interpreter.

    Attributes
    ----------
    fine_tune: arn.models.fine_tune.FineTune
    novelty_detector: WindowedMeanKLDiv
    feedback_interpreter: arn.models.feedback.CLIPFeedbackInterpreter = None
    """
    def __init__(
        self,
        fine_tune,
        novelty_detector,
        feedback_interpreter=None,
        dtype=torch.float32,
        #TODO label_enc
    ):
        """Initializes the OWHAR.

        Args
        ----
        see self
        """
        self.fine_tune = fine_tune
        self.novelty_detector = novelty_detector
        self.feedback_interpreter = feedback_interpreter
        self.dtype = torch_dtype(dtype)
        self._increment = 0

    @property
    def get_increment(self):
        """Increments correspond to how many times the predictor was fit."""
        return self._increment

    def fit(self, dataset, val_dataset=None, task_id=None):
        """Incrementally fit the OWHAPredictor's parts."""
        self._increment += 1
        self.fine_tune.fit(dataset, val=val_dataset)
        # NOTE update any other state for fitting, such as thresholds.

    def predict(self, dataset, task_id=None):
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
        self.fine_tune.predict(dataset)

    def novelty_detect(self, dataset, task_id=None):
        """Predictor performs novelty detection given the dataset, possibly
        conditioned on specific task set. Novelty detection is the same as
        anaomaly detection, outlier detection, out-of-distirbution detection,
        etc...

        Args
        ----
        dataset : torch.Dataset
        task_id : str = None
            The str identifier of the task to perform with the given inputs.
            This assumes the proper dataset input format is given for each task
            or that every task has the same input format.When task_id is None,
            default, it performs detection across all tasks and returns an
            novelty detection ansewr based on the data relative to all tasks.
        """
        raise NotImplementedError()
        self.fine_tune.feature_extract(dataset)

    # TODO def feedback_query(self, dataset, task_id=None):


# TODO class OWHAPredictorEVM(OWHAPredictor):
class OWHAPredictorEVM(object):
    """The OWHAR predictor class that contains and manages the predictor parts.

    Attributes
    ----------
    feature_repr : FeatureRepr | torch.Tensor = None
        The task's feature representation model used in this OWHAR. This is a
        torch model at its base.

        If pre-calculated (most likely for time), then simply a torch.Tensor or
        Dataset or Dataloader of that pre-extracted feature representations.

    fine_tuning : FineTuning | torch.Tensor | DataLoader
        The torch model used for fine tuning the FeatureRepr to the task, this
        is typically a small fully connected neural network, with 2-5 layers.

        input: feature_repr output
        output/encode:
            torch.Tensor([num_samples, batch, timeframes, fine_tune_repr_dim])
                A fine-tuned task repr of every video frame.

            torch.Tensor([num_samples, batch, fine_tune_repr_dim])
                A fine-tuned task repr of the entire video.
    evm : ExtremeValueMachine
    novelty_detector : NoveltyDetector | EVMWindowedMeanKLDiv
        The model that performs the novelty detection piece given
        classifier: EVM or softmax output of fine-tune w/ thresholding

    novelty_recog : NoveltyRecognizer
        The novelty recognition model used in this OWHAR. This handles only
        novelty recognition after detection has occurred.

        clustering: FINCH or HDBSCAN

        Not implemented for PAR EVAL 6 month, DARPA Eval 24 month.

    label_enc : exputils.data.labels.NominalDataEncoder
        The Dector/Recognizer's label encoder as that is the end of the
        OWHAR model.

        May need to make a Torch version of exputils...label_enc
    increment_count : int = 0
        The number of incremental training phases this OWHAR has completed.
        Starts at zero when no trainings have been completed.
    """
    def __init__(
        self,
        fine_tune,
        evm,
        novelty_detector,
        #novelty_recognizer,
        feedback_interpreter,
    ):
        """Initializes the OWHAR.

        Args
        ----
        feature_repr : FeatureRepr
        novelty_detector : NoveltyDetector
        novelty_recognizer : NoveltyRecognizer
        """
        # TODO init/load the Feature Representation model
        #self.feature_repr

        self.fine_tune = fine_tune
        self.evm = evm
        self.novelty_detector = novelty_detector
        #self.novelty_recognizer = novelty_recognizer
        self.feedback_interpreter = feedback_interpreter

        # TODO Data Management, should not be handled in the predictor, but
        # rather the data pipeline,  so perhaps a wrapper of the PRedictor with
        # some data stores for carrying the train and val data as a DataLoader.

    @property
    def get_increment(self):
        return self.evm.get_increment

    @property
    def label_enc(self):
        return self.evm.label_enc

    def known_probs(self, input_samples, is_feature_repr=True):
        if is_feature_repr:
            return self.evm.known_probs(
                self.fine_tune.extract(input_samples)
            )
        return self.evm.known_probs( # TODO frepr.extract()
            self.fine_tune.extract(self.feature_repr.extract(input_samples))
        )

    def predict(self, input_samples, is_feature_repr=True):
        """Classifies the input samples using the NoveltyDetector after getting
        the FeatureRepr of the samples.

        Parameters
        ----------
        input_samples : torch.tensor
            The task samples as expected by the FeatureRepr model if
            `is_feature_repr` is False. Otherwise, the input samples are
            treated as the feature repr encoded task samples as expected
            NoveltyDetector.
        is_feature_repr : bool = False

        Returns
        -------
        (torch.tensor, torch.tensor)
            The complete prediction output of the Predictor, including the
            classification probability vector(s) and the running novelty
            detection probability.
        """
        if is_feature_repr:
            return self.evm.predict(
                self.fine_tune.extract(input_samples)
            )
        return self.evm.predict( # TODO frepr.extract()
            self.fine_tune.extract(self.feature_repr.extract(input_samples))
        )

    def detect(self, input_samples, is_feature_repr=True):
        """Uses available NoveltyDetector/Recognizer to detect novelty in
        the given samples.
        """
        if is_feature_repr:
            return self.novelty_detector.detect(self.evm.known_probs(
                self.fine_tune.extract(input_samples)
            ))
        return self.novelty_detector.detect(self.evm.known_probs(
            self.fine_tune.extract(self.feature_repr.extract(input_samples))
        ))

    def fit_increment(
        self,
        input_samples,
        labels,
        is_feature_repr=True,
        val_input_samples=None,
        val_labels=None,
        val_is_feature_repr=True,
    ):
        """Incrementally fit the OWHAR."""
        #self.increment_count += 1
        if not is_feature_repr:
            # Fit when not frozen or apply special fitting overtime.
            input_samples = self.feature_repr.extract(input_samples)

        self.fine_tune.fit(input_samples, labels)
        # TODO all this casting is hot fixes and need better maintained by owhar
        test = self.fine_tune.extract(input_samples.to(self.fine_tune.device))
        self.evm.fit(
            test,
            labels.argmax(1).float().to("cpu"),
        )

        # NOTE update any other state for fititng, such as thresholds.
