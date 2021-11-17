"""Fine tuning models. These follow fine tuning in Pre-Trained Models"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

# TODO Fine Tuning state: init, load, save
# TODO Train/fit code given feature repr input
# TODO extract/predict code given feature repr input


class FineTune(object):
    """The Fine Tuning module of the open world human activity predictor.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be used for fine tuning. This is expected to support
        FineTuneFCANN.
    fit_args : dict()
        The hyperparameters for fitting the fine tuning model. This includes
        the optimizer, epochs, etc. to run a complete run of fitting the model.

        Though, batch size is handled by DataLoaders I believe. Let dataloaders
        handle what they can and their outputs be intput args. Everything will
        be torch.Tensors in and torch.Tensors out.

    eval_args : dict() = None
        The hyperparameters for evaluating the fine tuning model, if any.
        Often, there are none.

    [add anything else you deem necessary as state for the FineTune object.]
    """
    def __init__(self, model, fit_args, eval_args=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expected model typed as `torch.nn.Module`, not {type(model)}'
            )
        self.model = model

        # TODO Save the fit args

        # TODO Save the eval/fwd pass args if any, probs not.

        raise NotImplementedError()

    def fit(self, features, labels):
        """Fits the model with fit_args and the given features and labels in a
        supervised learning fashion.
        """

        # TODO perhaps not here, but should be easy to update the output size
        # based on the labels.

        raise NotImplementedError()

    def extract(self, features):
        """Given features, outputs the fully connected encoding of them.
        Args
        ----
        features : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError()

        return self.model.fcs(features)

    def predict(self, features, labels):
        # TODO If this eval/fwd pass loop overlaps with training loop, reuse
        # the code by function call, otherwise just do eval loop here. more
        # likely the training loop would use this at least in validation.

        # NOTE for our paper, we want this with ability to find a threshold
        # from all train and val data.

        raise NotImplementedError(
            'Lowest priority. Mostly same as extract, just outputs class vecs.'
        )

    def save(self, filepath):
        # TODO Save the torch model, and the current parameter set (this class'
        # args).
        raise NotImplementedError()

        torch.save(model, filepath)

        # TODO save args and other state info

    @staticmethod
    def load(filepath, **kwargs):
        """Load all the pieces from a file(s), possibly be done thru a config
        file. For now, feel free to hard code expectation of FineTuneFCANN.

        Returns
        -------
        FineTune
            An instance of FineTune is returned given the files containing
            state information.
        """
        raise NotImplementedError()

        model = torch.load(filepath) # handles torch model when saved as .pt

        # TODO load the other parameters . . .
        #   If not loaded from file or whatever, then can simply be given by
        #   **kwargs.

        return FineTune(model, )


class FineTuneFCANN(nn.Module):
    """Fully Connected Dense ANN for fine tuning."""
    def __init__(
        self,
        input_size,
        model_width=512,
        n_layers=5,
        out_features=512,
        n_classes=29,
        nonlinearity=nn.ReLU,
    ):
        """Fine-tuning ANN consisting of fully-connected dense layers."""
        self.nonlinearity = nonlinearity
        super().__init__()
        ord_dict = OrderedDict()
        ord_dict["fc0"] = nn.Linear(input_size, model_width)
        ord_dict["nl0"] = nonlinearity()

        for x in range(1,n_layers-1):
            ord_dict[f'fc{x}'] = nn.Linear(model_width, model_width)
            ord_dict[f'fc{x}'] = nonlinearity()

        ord_dict[f'fc{n_layers - 1}'] = nn.Linear(model_width, out_features)
        self.fcs = nn.Sequential(ord_dict)
        self.classifier = nn.Linear(out_features, n_classes)

    def forward(self, x):
        """Returns the last fully connected layer and the probs classifier"""
        x = self.fcs(x)
        classification = F.log_softmax(self.classifier(x), dim=1)
        return x, classification

    def get_finetuned_feature_extractor(self):
        """Returns ANN that outputs the final fully connected layer."""
        return self.fcs

    def load_interior_weights(self, state_dict):
        temp = []
        for x in state_dict:
            if x[:3] != 'fcs':
                temp.append(x)
        for x in temp:
            state_dict.pop(x)
        self.load_state_dict(state_dict,strict=False)
