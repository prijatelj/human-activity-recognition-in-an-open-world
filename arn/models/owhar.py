"""Open World Human Activity Recognition pipeline class."""
import torch

from exputils.ml.generic_predictors import SupervisedClassifier

from arn.models import generics

class OpenWorldHumanActivityRecognizer(SupervisedClassifier):
    """Pipeline class of the different parts of the OWHAR.

    Attributes
    ----------
    feature_repr : FeatureRepr
        The task's feature representation model used in this OWHAR. This is a
        torch model at its base.
    novelty_detector : NoveltyDetector | NoveltyRecognizer
        The novelty detector model used in this OWHAR. This is either a
        NoveltyDetector, with no ability to perform novelty recognition, or is
        a NoveltyRecognizer, in which case this simply points to the same
        object as `novelty_recognizer`.
    novelty_recognizer : None | NoveltyRecognizer, opt.
        The novelty recognizer model used in this OWHAR. A NoveltyRecognizer is
        not always included.
    label_enc : exputils.data.labels.NominalDataEncoder
        The Dector/Recognizer's label encoder as that is the end of the
        OWHAR model.
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
        # TODO init/load the task Feature Representation model
        self.feature_repr

        # TODO init/load the NoveltyDetector model if only a detector
        self.novelty_detector

        # TODO init/load the NoveltyRecognizer model which is also the detector
        self.novelty_recognizer

        self.increment_count = increment_count

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
        torch.tensor
            The resulting Torch tensor.
        """
        raise NotImplementedError()
        # TODO Gets the freature representation of the samples
        # TODO uses the NoveltyDetector of the OWHAR model for classification.
        return self.novelty_detector.predict(self.get_feature_repr(input_samples))

    def get_feature_repr(self, input_samples):
        """Encodes the task input samples with using the FeatureRepr model."""
        # TODO Obtain the feature representation of the input samples
        # TODO consider finishing TorchAnnExtractor and use it here.
        raise NotImplementedError()
        return self.feature_repr.encode(input_samples)

    def detect(self, input_samples):
        """Uses available NoveltyDetector/Recognizer to detect novelty in
        the given samples.
        """
        raise NotImplementedError()
        return self.novelty_detector.detect(input_samples)

    def recognize(self, input_samples):
        """Pass through that """
        raise NotImplementedError()
        if self.novelty_recognizer:
            return self.novelty_recognizer.recognize(input_samples)
        raise ValueError('No novelty_recognizer present!')

    def fit_increment(self, input_samples, labels):
        """Incrementally fit the OWHAR."""
        raise NotImplementedError()
        self.increment_count += 1
