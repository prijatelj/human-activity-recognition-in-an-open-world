"""Open World Human Activity Recognition pipeline class."""
import torch

from exputils.ml.generic_predictors import SupervisedClassifier

from arn.models import generics

class OpenWorldHumanActivityPredictor(SupervisedClassifer):
    """Pipeline class of the different parts of the OWHAR.

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



    novelty_detector : NoveltyDetector
        The model that performs the novelty detection piece given
    novelty_recog : NoveltyRecognizer
        The novelty recognition model used in this OWHAR. This handles both
        novelty detection and novelty recognition

        classifier: EVM or softmax output of fine-tune w/ thresholding
        clustering: FINCH or HDBSCAN

    label_enc : exputils.data.labels.NominalDataEncoder
        The Dector/Recognizer's label encoder as that is the end of the
        OWHAR model.

        May need to make a Torch version of exputils...label_enc
    increment_count : int = 0
        The number of incremental training phases this OWHAR has completed.
        Starts at zero when no trainings have been completed.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the OWHAR.

        Args
        ----
        feature_repr : FeatureRepr
        novelty_detector : NoveltyDetector
        novelty_recognizer : NoveltyRecognizer
        """
        # TODO init/load the Feature Representation model
        self.feature_repr

        # TODO init/load the task fine tuning model
        self.feature_repr

        # TODO init/load the NoveltyDetector model
        self.novelty_detector

        # TODO init/load the NoveltyRecognizer model
        self.novelty_recognizer

        self.increment_count = increment_count

        # TODO Data Management, should not be handled in the predictor, but
        # rather the data pipeline,  so perhaps a wrapper of the PRedictor with
        # some data stores for carrying the train and val data as a DataLoader.

        raise NotImplementedError()

    @property
    def label_enc(self):
        return self.novelty_detector.label_enc

    def predict(self, input_samples, is_feature_repr=False):
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
            return self.novelty_detector.predict(
                self.fine_tune.extract(input_samples)
            )
        return self.novelty_detector.predict( # TODO frepr.extract()
            self.fine_tune.extract(self.feature_repr.extract(input_samples))
        )

    def detect(self, input_samples):
        """Uses available NoveltyDetector/Recognizer to detect novelty in
        the given samples.
        """
        return self.novelty_detector.detect(self.predict(input_samples))

    def recognize(self, input_samples, is_feature_repr=False):
        """Passes off to self.predict()."""
        return self.predict(input_samples)

    def fit_increment(
        self,
        input_samples,
        labels,
        is_feature_repr=False,
        val_input_samples,
        val_labels,
        val_is_feature_repr=False,
    ):
        """Incrementally fit the OWHAR."""
        self.increment_count += 1

        if not is_feature_repr:
            # TODO fit when not frozen or apply special fitting overtime.
            input_samples = self.feature_repr.extract(input_samples)

        self.fine_tune.fit(input_samples, labels)
        self.novelty_detect.fit(self.fine_tune.extract(input_samples), labels)

        # TODO update any other state for fititng, such as thresholds.
