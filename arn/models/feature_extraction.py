"""Feature Representation classes."""
import torch

class FeatureExtractor(nn.Module):
    """Feature extraction of a layer of a pre-existing PyTorch model.

    Attributes
    ----------
    model : nn.Module
        The PyTorch model whose feature representation from the given `layer`
        is to be extracted.
    layer : str
        The fully-qualified string name of the layer to be extracted from
        the model for every input.
    """
    def __init__(self, model, layer, copy_model=True):
        if not hasattr(model, submodule_name):
            raise NotImplementedError(
                f'`layer` is not an attribute of `model`. `layer` = {layer}'
            )
        if copy_model:
            self.model = model
        else:
            self.model = model

        self.layer = layer

    def forward(self, x):
        """Every feature extractor will extract features from a sample or
        multiple samples at once.
        """
        raise NotImplementedError()

        x = self.model(x)


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
            raise ValueError(
                '`network` is not a valid torchvision model: `{network}`'
            )

        self.network = getattr(models, network)(pretrained=pretrained)

        if not hasattr(self.network, layer):
            raise NotImplementedError(
                f'`layer` is not an attribute of `network`. `layer` = {layer}'
            )

        # TODO i think this only works for sequential models by overwriting the
        # given layer with the identity function. This needs confirmed & being
        # able to grab any layer or section of an ANN by name would be better.
        setattr(self.network, layer, Identity())

    def extract(self, image):
        return self.network(image).detach().numpy()
