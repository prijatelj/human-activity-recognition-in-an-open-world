"""Fine tuning models. These follow fine tuning in Pre-Trained Models"""
import copy
from collections import OrderedDict
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
        self.loss = loss
        self.lr = lr
        self.optimizer_cls = torch.optim.Adam

        # TODO Torch checkpoint stuffs

        # TODO TensorBoard logger stuffs

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

            for i, x in enumerate(dataset):
                torch.autograd.set_grad_enabled(True)

                sfeatures, slabels = x
                sfeatures = sfeatures.to(self.device)
                slabels = slabels.to(self.device).float()

                prediction = model(sfeatures)[1]

                right += torch.sum(
                    torch.eq(
                        torch.argmax(prediction, dim=1),
                        torch.argmax(slabels, dim=1)
                    ).int()
                ).cpu().numpy()

                loss = self.loss(prediction, slabels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tot_cls_loss += loss.item()
            tacc = str(right / train_len)
            right = 0
            tot_cls_loss = 0.0

            if val_dataset is None: # Skip validation
                continue

            for i, x in enumerate(val_dataset):
                torch.autograd.set_grad_enabled(False)
                sfeatures, slabels = x
                sfeatures = sfeatures.cuda()
                slabels = slabels.cuda().float()
                prediction = model(sfeatures)[1]
                right += torch.sum(
                    torch.eq(
                        torch.argmax(prediction, dim=1),
                        torch.argmax(slabels, dim=1)
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
                    print("Val Accuracy: " + str(right / v_len))

        self.model = model

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        classifications = self.model(inputs)[1]

        loss = self.loss(classifications, labels)
        acc = (
            labels.argmax(1) == F.softmax(classifications, 1).argmax(1)
        ).to(float).mean()

        #self.log('train_loss', loss)
        #self.log('train_accuracy', acc)

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
        return self.model.fcs(features.to(self.device, self.dtype))

    def predict(self, features):
        # TODO If this eval/fwd pass loop overlaps with training loop, reuse
        # the code by function call, otherwise just do eval loop here. more
        # likely the training loop would use this at least in validation.

        # NOTE for our paper, we want this with ability to find a threshold
        # from all train and val data.
        features, prediction = self.model(features.to(self.device, self.dtype))
        # Why softmax when the model has softmax?
        prediction = F.softmax(prediction.detach(), dim=1)
        return prediction

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
    inputs,
    width,
    dropout=None,
    activation=None,
):
    """Creates the 'dense layer' with opt. dropout and activation."""
    ord_dict[f'fc{layer_num}'] = nn.Linear(inputs, width)

    if dropout and isinstance(dropout, float):
        ord_dict[f'Dropout{layer_num}'] = nn.Dropout(dropout, True)
    elif dropout:
        ord_dict[f'Dropout{layer_num}'] = nn.Dropout(
            dropout[layer_num],
            True,
        )

    if activation is not None:
        ord_dict[f'{activation.__name__}{layer_num}'] = activation()


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
        """
        super().__init__()
        if depth < 2:
            raise ValueError('Depth less than 2 is not supported!')
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
        dense_layer(ord_dict, 0, input_size, width, dropout, activation)

        # NOTE that depth includes input and the feature_repr_width can be diff
        for x in range(1, depth-1):
            dense_layer(ord_dict, x, width, width, dropout, activation)

        # Final dense / fully connected layer as output feature representation
        dense_layer(
            ord_dict,
            depth - 1,
            width,
            feature_repr_width,
            dropout if dropout_feature_repr else None,
            activation,
        )

        self.fcs = nn.Sequential(ord_dict)
        self.classifier = nn.Linear(feature_repr_width, n_classes)

    def forward(self, x):
        """Returns the last fully connected layer and the probs classifier"""
        x = self.fcs(x)
        return x, self.classifier(x)

    def load_interior_weights(self, state_dict):
        temp = []
        for x in state_dict:
            if x[:3] != 'fcs':
                temp.append(x)
        for x in temp:
            state_dict.pop(x)
        self.load_state_dict(state_dict, strict=False)
