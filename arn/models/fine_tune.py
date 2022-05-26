"""Fine tuning models. These follow fine tuning in Pre-Trained Models"""
import copy
from collections import OrderedDict
from functools import partial
import logging

import torch
nn = torch.nn
F = torch.nn.functional

from arn.data.kinetics_unified import get_kinetics_uni_dataloader
from arn.torch_utils import torch_dtype


class FineTune(object):
    """The Fine Tuning module of the open world human activity predictor.

    Attributes
    ----------
    model : FineTuneFC
        docstr needs to support subclasses of a given class: torch.nn.Module
        The model to be used for fine tuning. This is expected to support
        FineTuneFC.
    batch_size : int = 1000
    epochs : int = 25
        Number of epochs to use during fitting.
    device : str | torch.device = 'cpu'
        the device on which model should be trained
        default: cpu
    dtype : torch.dtype = torch.float32
    shuffle : bool = True
        If True, shuffle the data when fitting. If False, no shuffling.
    num_workers : int = 0
        Number of works to ues for the DataLoader.
    pin_memory : bool = False
        Pin memory for data loaders.
    loss : torch.nn.modules.loss._Loss = None
        TODO in docstr add support for `see FineTuneFC.__init__`
    lr : float = 0.001
        The learning rate for the optimizer. Optimizer is ADAM.
    """
    def __init__(
        self,
        model,
        batch_size=1000,
        epochs=25,
        device='cpu',
        dtype=torch.float32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        loss=None,
        lr=0.001,
    ):
        """Init the FineTune model.

        Args
        ----
        see self
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expected model typed as `torch.nn.Module`, not {type(model)}'
            )

        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Multiprocessing params for DataLoaders
        self.pin_memory = pin_memory #num_workers > 0

        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)

        # Optimizer things
        if loss is None:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = loss
        self.lr = lr
        self.optimizer_cls = torch.optim.Adam

        # TODO Torch checkpoint stuffs

        # TODO TensorBoard logging stuffs

    def configure_optimizer(self):
        """Re-init optimizer every fitting."""
        return self.optimizer_cls(
            self.model.parameters(),
            self.lr,
        )

    def fit(self, dataset, val_dataset=None, verbose=False):
        """Fits the model with fit_args and the given features and labels in a
        supervised learning fashion.

        Args
        ----
        dataset : KineticsUnifiedFeatures | torch.utils.data.DataLoader
            The dataset to be turned into a DataLoader or the DataLoader itself
            used for fitting the model.
        val_dataset : KineticsUnifiedFeatures
            Same as `dataset`, except used for validation during the fitting
            process.
        verbose : bool = False
            If True, prints out progress to std out.
        """
        dataset = get_kinetics_uni_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        train_len = len(dataset)

        if val_dataset is not None:
            val_dataset = get_kinetics_uni_dataloader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False, #self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            val_len = len(val_dataset)

        model = self.model.to(self.device)
        #criterion = torch.nn.BCEWithLogitsLoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        optimizer = self.configure_optimizer()

        best_cls_loss = 99999999999999999
        best_val_acc = 0

        for epoch in range(self.epochs):
            if verbose:
                print("Epoch: " + str(epoch) + "---------------")
            tot_cls_loss = 0.0
            right = 0

            model.train()
            for i, x in enumerate(dataset):
                torch.autograd.set_grad_enabled(True)

                sfeatures, slabels = x
                sfeatures = sfeatures.to(self.device)
                slabels = slabels.to(self.device).float()

                prediction = model(sfeatures)[1]

                right += torch.sum(
                    torch.eq(
                        torch.argmax(prediction, dim=-1),
                        torch.argmax(slabels, dim=-1)
                    ).int()
                ).cpu().numpy()

                loss = self.loss(prediction, slabels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_cls_loss += loss.item()
            tacc = str(right / train_len)
            right = 0
            tot_cls_loss = 0.0

            if val_dataset is None: # Skip validation
                continue

            model.eval()
            for i, x in enumerate(val_dataset):
                torch.autograd.set_grad_enabled(False)
                sfeatures, slabels = x
                sfeatures = sfeatures.cuda()
                slabels = slabels.cuda().float()
                prediction = model(sfeatures)[1]
                right += torch.sum(
                    torch.eq(
                        torch.argmax(prediction, dim=-1),
                        torch.argmax(slabels, dim=-1)
                    ).int()
                ).cpu().numpy()
                loss = self.loss(prediction, slabels)
                tot_cls_loss += loss.item()
            if best_cls_loss > tot_cls_loss:
                best_cls_loss = tot_cls_loss
                # torch.save(model.state_dict(), save_path +'finetune_best.pt')
                best_model = copy.deepcopy(model).cpu()
                if verbose:
                    print("New Best " + str(tot_cls_loss))
                    print("Train Accuracy: " + tacc)
                    print("Val Accuracy: " + str(right / val_len))

        self.model = model

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        classifications = self.model(inputs)[1]

        loss = self.loss(classifications, labels)
        acc = (
            labels.argmax(1) == F.softmax(classifications, 1).argmax(1)
        ).to(float).mean()

        return loss

    def extract(self, features):
        """Given features, outputs the fully connected encoding of them.
        Args
        ----
        features : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        self.model.eval()

        if isinstance(features, torch.Tensor):
            return self.model.fcs(features)

        dataset = get_kinetics_uni_dataloader(
            features,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        preds = []
        for i, x in enumerate(dataset):
            preds.append(self.model.fcs(x))

        if self.batch_size > 1:
            return torch.concat(preds)
        return torch.stack(preds)

    def predict(self, features):
        # Why softmax when the model has softmax? Should be just torch.exp()
        self.model.eval()

        logging.debug(
            '%s.predict() given features of type: %s',
            self.__name__,
            type(features),
        )

        if isinstance(features, torch.Tensor):
            return F.softmax(self.model(features)[1].detach(), dim=-1)

        dataset = get_kinetics_uni_dataloader(
            features,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        preds = []
        for i, x in enumerate(dataset):
            preds.append(F.softmax(self.model(x)[1], dim=-1))

        if self.batch_size > 1:
            return torch.concat(preds)
        return torch.stack(preds)

    def save(self, filepath):
        torch.save(self.model, filepath)

    @staticmethod
    def load(filepath, input_size=400, **kwargs):
        """Load all the pieces from a file(s), possibly be done thru a config
        file. For now, feel free to hard code expectation of FineTuneFC.

        Returns
        -------
        FineTune
            An instance of FineTune is returned given the files containing
            state information.
        """
        state_dict = torch.load(filepath) # handles torch model when a .pt
        model = FineTuneFC(input_size, **kwargs)
        model.load(state_dict)

        return FineTune(model, **kwargs)


def dense_layer(
    ord_dict,
    layer_num,
    input_size,
    width,
    dropout=None,
    activation=None,
):
    """Creates the 'dense layer' with opt. dropout and activation."""
    ord_dict[f'fc{layer_num}'] = nn.Linear(input_size, width)

    if dropout and isinstance(dropout, float):
        ord_dict[f'Dropout{layer_num}'] = nn.Dropout(dropout, True)
    elif dropout: # rm this
        ord_dict[f'Dropout{layer_num}'] = nn.Dropout(
            dropout[layer_num],
            True,
        )

    if activation is not None:
        ord_dict[f'{activation.__name__}{layer_num}'] = activation()


def get_dense_layers(
    input_size,
    width=512,
    depth=5,
    feature_repr_width=None,
    activation=nn.LeakyReLU,
    dropout=None,
    dropout_feature_repr=True,
    act_on_input=False,
):
    """Create the ordered dictionary of sequential dense layers."""
    if depth < 1:
        raise ValueError('Depth less than 1!')
    if feature_repr_width is None:
        feature_repr_width = width

    if (
        dropout and not isinstance(dropout, float)
        and (
            (dropout_feature_repr and len(dropout) != depth)
            or (not dropout_feature_repr and len(dropout) != depth-1)
        )
    ):
        raise ValueError('Length of dropout does not match depth!')

    ord_dict = OrderedDict()

    if act_on_input:
        # TODO test this, dunno if it works w/o an input layer provided.
        if dropout and isinstance(dropout, float):
            ord_dict[f'Dropout-input'] = nn.Dropout(dropout, True)
        elif dropout:
            ord_dict[f'Dropout-input'] = nn.Dropout(dropout[0], True)

        if activation is not None:
            ord_dict[f'{activation.__name__}-input'] = activation()

    dense_layer(
        ord_dict,
        0,
        input_size,
        width if depth == 1 else feature_repr_width,
        dropout if depth == 1 and dropout_feature_repr else None,
        activation,
    )

    # NOTE that depth includes input and the feature_repr_width can be diff
    for x in range(1, depth-1):
        dense_layer(ord_dict, x, width, width, dropout, activation)

    # Final dense / fully connected layer as output feature representation
    if depth > 1:
        dense_layer(
            ord_dict,
            depth - 1,
            width,
            feature_repr_width,
            dropout if dropout_feature_repr else None,
            activation,
        )
    return ord_dict


class FineTuneFC(nn.Module):
    """Fully Connected Dense ANN for fine tuning.

    Attributes
    ----------
    fcs : torch.nn.Sequential
        The fully connected sequential dense ANN whose output is the
        penultamite layer of the network just before the linear, softmaxed
        layer.
    classifier : torch.nn.Linear
        The remaining portion of the sequential model that takes as input
        the output of fcs and maps that to a dense layer. Note that this does
        not apply log_softmax to the output of the linear, only puts it into
        the correct number of dimensions (number of classes). The log_softmax
        is applied to the first dimension in the `forward()` method.
    """
    def __init__(
        self,
        input_size,
        width=512,
        depth=5,
        feature_repr_width=None,
        n_classes=29,
        activation=nn.LeakyReLU,
        dropout=None,
        dropout_feature_repr=True,
        act_on_input=False,
        residual_maps=None,
        input_name='input',
    ):
        """Fine-tuning ANN consisting of fully-connected dense layers.

        Args
        ----
        input_size : int
            The input size of the input linear layer.
        width : int = 512
            The width of the hidden layers within the ANN.
        depth : int = 5
            The depth or total Linear layers within the ANN. This includes the
            input layer, so the number of hidden layers is depth - 1. Does
            not include the final output layer used for classification, which
            is the size of the number of classes.
        feature_repr_width : int = None
            The width of the penultamite layer of this ANN, which is the layer
            just before the output(softmax) and serves as the feature
            representation of the ANN. By default, this is None and not set,
            which means the feature representation layer will use the same
            `width` as the other hidden layers.
        n_classes : int = 29
            The number of classes to expect for the output layer of this ANN.
        activation : torch.nn.Module = torch.nn.LeakyReLU
            The activation to apply after every linaer layer.
        dropout : float = None
            The probability for the dropout layers after the linear layers.
            Defaults to None, meaning no dropout is applied.
        dropout_feature_repr : bool = True
            If True, dropout is applied to the last hidden layer.
        act_on_input : bool = False
            If True, dropout and the activation is applied to the inputs first.
        residual_maps : OrderedDict = None
            see MultiInputModules
        input_name : 'input'
            The name to use for input name to ResidualConnections. Unused if
            residual_maps is None.
        """
        super().__init__()
        dense_layers = get_dense_layers(
            input_size,
            width,
            depth,
            feature_repr_width,
            activation,
            dropout,
            dropout_feature_repr,
            act_on_input,
        )
        if residual_maps is None:
            self.fcs = nn.Sequential(dense_layers)
        else:
            self.fcs = ResidualConnections(
                dense_layers,
                residual_maps,
                input_name,
            )
        self.classifier = nn.Linear(feature_repr_width, n_classes)

    def forward(self, x):
        """Returns the last fully connected layer and the probs classifier"""
        x = self.fcs(x)
        return x, self.classifier(x)

    def load_interior_weights(self, state_dict):
        # TODO is this necessary?
        temp = []
        for x in state_dict:
            if x[:3] != 'fcs':
                temp.append(x)
        for x in temp:
            state_dict.pop(x)
        self.load_state_dict(state_dict, strict=False)


def get_residual_map(residual_maps):
    """Helper function to return the residual maps.

    Args
    ----
    residual_maps: list
        List of tuples where ecah tuple is (str, str, list(str))
    """
    ord_dict = OrderedDict()
    for residual_map in residual_maps:
        if residual_map[1] in {'cat', 'concat'}:
            join_method = torch.cat
        elif residual_map[1] in {'cat1', 'concat1'}:
            join_method = partial(torch.cat, dim=1)
        elif residual_map[1] in {'cat-1', 'concat-1'}:
            join_method = partial(torch.cat, dim=-1)
        elif residual_map[1] == 'stack':
            join_method = torch.stack
        elif residual_map[1] in 'stack1':
            join_method = partial(torch.stack, dim=1)
        elif residual_map[1] in 'stack-1':
            join_method = partial(torch.stack, dim=-1)
        elif residual_map[1] == 'add':
            join_method = lambda x: torch.add(*x)
        else:
            join_method = residual_map[1]

        ord_dict[residual_map[0]] = (join_method, residual_map[2])
    return ord_dict


class ResidualConnections(nn.Module):
    """Given a set of modules and a residual mappings, forms the model.
    Designed primarily with torch.nn.Sequential's OrderedDict in mind for
    `modules`.

    Attributes
    ----------
    modules : OrderedDict
        An OrderedDict of modules as if in a torch.nn.Sequential.
    residual_maps : OrderedDict
        An ordered dict with keys as the module name in modules that will
        recieve the output of joining the given residual connections. The
        values are then a tuple of two elements with the first being a
        callable, such as `torch.concat`, and the second being a list of
        inputs to join in order.

        The residual mappings to modify the existing modules' connections
        to support the joining of output from named layers and replace the
        existing input to the output layer with the resulting residual map.

        The inputs are expected to be ordered by occurrence in OrderedDict.
    input_name = 'input'
        The name used to signify the input given to this module in `forward()`.
    """
    def __init__(self, modules, residual_maps, input_name='input'):
        """Stores the key components for constructing the residual connections.

        Args
        ----
        see self
        """
        super.__init__()
        self.modules = modules
        if input_name in modules:
            raise KeyError(f"'{input_name}' cannot be a key within `modules`.")
        self.input_name = input_name
        self.residual_maps = residual_maps
        self.residual_inputs = set()

        # TODO For user convenience, check for out of order inputs to outputs
        # mapping. Ensure output is after its inputs.
        for key, value in residual_maps.items():
            self.residual_inputs.update(value[1])

    def forward(self, x):
        """Runs through the given modules and forms the residual connections.
        Stores the outputs of every layer used as inputs in residual maps.
        """
        # Store the outputs names in residual inputs for future residuals
        outputs = OrderedDict({self.input_name: x})
        res_map_count = 0
        for name, module in self.modules.items():
            if name in self.residual_maps:
                # Check if the residual maps' inputs have completed.
                join_method, inputs = self.residual_maps[name]
                try:
                    input_tensors = [outputs[i] for i in inputs]
                except KeyError as err:
                    # If they have not, error for simplicity stating incomplete
                    # residual mapped inputs. Should be checked in init.
                    raise KeyError(
                        'Residual map input {i} has yet to be computed. '
                        'Inputs out of order!'
                    ) from err

                # If they have completed, run them through the join method and
                # give as input to this layer.
                x = join_method(input_tensors)

                joined_name = f'ResidualConnection{res_map_count}'
                res_map_count += 1

                if joined_name in self.residual_inputs:
                    outputs[joined_name] = x
            x = module(x)

            if name in self.residual_inputs:
                outputs[name] = x
        return x
