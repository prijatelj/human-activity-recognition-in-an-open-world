"""Open World Human Activity Recognition pipeline class."""
import torch

from exputils.ml.generic_predictors import SupervisedClassifier

from arn.models.novelty_detector import WindowedMeanKLDiv

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
        # print(input_samples)
        # input_samples = torch.Tensor(input_samples)
        # print(input_samples)
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
            # TODO fit when not frozen or apply special fitting overtime.
            input_samples = self.feature_repr.extract(input_samples)

        self.fine_tune.fit(input_samples, labels)
        # TODO all this casting is hot fixes and need better maintained by owhar
        test = self.fine_tune.extract(input_samples.to(self.fine_tune.device))
        self.evm.fit(
            test,
            labels.argmax(1).float().to("cpu"),
        )

        # TODO update any other state for fititng, such as thresholds.
