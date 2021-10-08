"""Generic and Abstract classes for pipelining."""
from abc import ABC, abstractmethod

# NOTE see `exputils` up-stream for details, e.g. generic save() and load()
# Overwrite these as necessary in new children classes.
from exputils.data.labels import NominalDataEncoder
from exputils.ml.generic_predictors import SupervisedClassifier


class NoveltyDetector(SupervisedClassifier):
    """Abstract class for online supervised learning detectors.
    This will serve as the pipeline that wraps or includes the Pytorch model
    used so any data prep or model prep is handled in the child inheriting from
    this class, such as save and load in niche cases, esp. if we want to save
    the model hyperparams with the weights, where Pytorch model load/save
    covers generic weight handling moreso than generic model handling.

    Also, this keeps the label encoder with the predictor.

    Attributes
    ----------
    label_encoder : NominalDataEncoder
        In case we have to classify writers, the nominal data encoder is a
        convenience wrapper of scikit learn encoders so everything is always
        there in one place.

        It may need extended to handle "unknown", but for now the functionality
        is there and able to be coded in (e.g. include unknown as a label, and
        use reduce about unknowns to unknown after adding new unique unknown
        labels to encoder).

        For now I assume this is writer. This may not see much use given this
        is a detector not a classifier.
    every_other_discrete_label_thing : NominalDataEncoder
        If there are more
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        #self.label_enc = self.char_enc
        pass

    @abstractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()

    def detect(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()


class NoveltyRecognizer(OnlineDetector):

    def init(self, *args, **kwargs):
        pass

    def recognize(self, features, labels):
        raise NotImplementedError()


class FeatureExtractor(StatefulIterable, Dataset):
    """Feature extraction abstract class."""
    # NOTE be careful with this and children feature extractor chaining. It may
    # not be the most efficient method!
    def __getitem__(self, idx):
        super(FeatureExtractor, self).__getitem__(idx)
        #item = self.iterable[idx]
        #item.image = self.extract(item.image)
        #return item

    @abstractmethod
    def extract(self, image):
        """Every feature extractor will extract features from a sample or
        multiple samples at once.
        """
        raise NotImplementedError()


# TODO ANN pretrained (e.g. ResNet50 on ImageNet) repr as an encoding, a
# generic layer extraction class that access that network's selected layer.
class TorchANNExtractor(FeatureExtractor):
    """Load a pretrained torch network and obtain the desired layer encoding of
    the input as the feature extraction of that input.
    """
    def __init__(self, network, layer='fc', pretrained=True):
        if not isinstance(network, str):
            raise TypeError(
                f'`network` is expected to be a str, not {type(network)}',
            )
        if not hasattr(models, network):
            raise ValueError('`network` is not a valid torchvision model.')

        self.network = getattr(models, network)(pretrained=pretrained)

        if not hasattr(self.network, layer):
            raise NotImplementedError(
                f'`layer` is not an attribute of `network`. `layer` = {layer}'
            )

        # TODO i think this only works for sequential models by overwriting the
        # given layer with the identity functio. This needs confirmed & being
        # able to grab any layer or section of an ANN by name would be better.
        setattr(self.network, layer, Identity())

    def extract(self, image):
        return self.network(image).detach().numpy()
