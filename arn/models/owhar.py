"""Open World Human Activity Recognition pipeline class."""
import torch

from arn.models import generics

class OpenWorldHumanActivityRecognizer(SupervisedClassifier):
    """Streamline class of the different parts of the OWHAR.

    Parameters
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
    increment_count : int
        The number of incremental training phases this OWHAR has completed.
        Starts at zero when no trainings have been completed.
    """
    def __init__(self, *args, **kwargs):
        """HA HA HAR"""
        # TODO init/load the task Feature Representation model
        self.feature_repr

        # TODO init/load the NoveltyDetector model if only a detector
        self.novelty_detector

        # TODO init/load the NoveltyRecognizer model which also serves as the
        # NoveltyDetector
        self.novelty_recognizer

        self.increment_count = 0

        raise NotImplementedError()

    @property
    def label_enc(self):
        return self.novelty_detector.label_enc

    def classify(self, input_samples):
        """Classifies """
        # TODO uses the end of the OWHAR model for classification.
        raise NotImplementedError()
        return

    def get_feature_repr(self, input_samples):
        """
        """
        # TODO Obtain the feature representation
        raise NotImplementedError()
        return

    def detect(self, input_samples):
        """Uses available NoveltyDetector/Recognizer to detect novelty in
        the given samples.
        """
        raise NotImplementedError()
        return self.novelty_detector.detect(*args, **kwargs)

    def recognize(self, input_samples):
        """Pass through that """
        raise NotImplementedError()
        if self.novelty_recognizer:
            return self.novelty_recognizer.recognize(*args, **kwargs)
        raise ValueError('No novelty_recognizer present!')
