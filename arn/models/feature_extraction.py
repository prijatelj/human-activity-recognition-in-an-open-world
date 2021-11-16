"""Feature Representation classes."""
import torch

# A pip installable 3rd party package:
#from bidict import OrderedBidict
# Could use exputils.data.labels.NominalDataEncoder instead


# TODO Sam: This does NOT have to be nn.Module. Your Feature Extractor does NOT
# need to inherit from this class below, it just needs to do the black box
# functionality as defined. You may need to create a class or code for each X3D
# and TimeSformer, although if it can be done with the same module then great.

# TODO Sam: If you make it a class object or a function(s), it doesn't matter
# as long as it works as expected and doesn't cause Out of Memory issues or is
# abysmally slow. However, please make it importable ``` from
# arn.models.wherever import whatever ```

# NOTE Sam, that our feature extractors are probably going to remain frozen for
# times sake and we just use a small (like 5 layered max) fuly connected
# network for fine tuning.

# TODO Please make the torch.nn.Module for that N layer fully connected
# network for classification. We will probably use either 2 layers or 5, no
# more. This is trivial in torch.
class FeatureExtractor(torch.nn.Module):
    """Feature extraction of a layer of a pre-existing PyTorch model.

    Attributes
    ----------
    model : nn.Module | str
        The PyTorch model whose feature representation from the given `layer`
        is to be extracted. If the model is in memory, and to hvae data run
        through it, then this is a nn.Module. Otherwise, the model's features
        are already extracted and saved to disk so this is simply a str
        identifier.
    layer : str
        The fully-qualified string name of the layer to be extracted from
        the model for every input.
    load_path : str | open_file | None
        If loading pre-extracted features, then this is the str filepath or the
        open file object (if it remains open cuz cannot fit in memory).
        Otherwise this is None
    sample_map : OrderedBidict(int: str) | None
        A sample bidirectional map of integer of index to sample identifier.  I
        don't necessarily care how this is implemented, but I do want to be
        able to easily idenitify what feature extraction goes to what sample,
        esp. when loading from pre-extracted features! So we can do eval
        correctly and other down stream tasks.

        For Kinetics, the sample id is probably youtube id, time start, time
        end, or simply the video path name scheme they use, which is the same
        info.

        Keep this none if the sample id map to feature extraction is able to be
        done already with existing external object, but please comment how that
        is done.

    Notes
    -----
    If model in memory:
        Then given that model and the layer to extract from, the black box is
        then fwd data -> feature representation of that data for the given
        layer.

        So [N, FRAMES, OTHER_DATA_DIMS] in, [N, FRAMES, FEATURE_REPR_DIM] out.

        Use __init__ for this.

    If feature extraction on Disk:
        Given the path to that feature extraction, load it to memory.
        So path(s) in, [N, FRAMES, FEATURE_REPR_DIM] out.

    Expected output always: [N, FRAMES, FEATURE_REPR_DIM]
        This allows me to use the EVM on it, OR use softmax with thresholding,
        OR put into another network for finetuning.
    """
    def __init__(self, model, layer, copy_model=True):
        """Given the torch model and layer, setup all attirbutes above as
        necessary to enable using the feature extractions per sample with
        knowledge of which sample the feature extraction is for.

        For Kinetics, the sample id is probably youtube id, time start, time
        end, or simply the video path name scheme they use, which is the same
        info.

        Attributes
        ----------
        copy_model : bool = True
            Only exists in the case we were gonna do things with the existing
            model separate from this. If not, then forget it, do not copy.
            This expects this code to modify the given model, if it does not,
            then definitely do NOT copy.
        """
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

        Returns
        -------
        torch.Tensor
            The [n_samples, frames, feature_repr_dims] tensor for the given
            data samples.
        """
        raise NotImplementedError()

        x = self.model(x)

    @staticmethod
    def load(filepath):
        """Given a filepath(s), load the pre-calculated feature extractions.

        Returns
        -------
        FeatureExtractor
            Either this class to bundle all the objects pertaining to the
            extracted features or some other object that is easy to use and
            keeps the things together. The objects are the attributes above.
        """
        return FeatureExtractor()

class TimesformerFeatureExtractor(FeatureExtractor):



# NOTE Ignore the following, Sam.

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
