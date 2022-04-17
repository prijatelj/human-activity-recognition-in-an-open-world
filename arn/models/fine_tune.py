"""Fine tuning models. These follow fine tuning in Pre-Trained Models"""
import copy
from collections import OrderedDict
import logging

import torch
nn = torch.nn
F = torch.nn.functional


class FineTune(object):
    """The Fine Tuning module of the open world human activity predictor.

    Attributes
    ----------
    model : FineTuneFC
        docstr needs to support subclasses of a given class: torch.nn.Module
        The model to be used for fine tuning. This is expected to support
        FineTuneFC.
    fit_args : dict
        The hyperparameters for fitting the fine tuning model. This includes
        the optimizer, epochs, etc. to run a complete run of fitting the model.

        Though, batch size is handled by DataLoaders I believe. Let dataloaders
        handle what they can and their outputs be intput args. Everything will
        be torch.Tensors in and torch.Tensors out.
    device : torch.device
        the device on which model should be trained
        default: cpu
    """
    def __init__(
        self,
        model,
        batch_size=1000,
        epochs=25,
        device='cpu',
        dtype=torch.float32,
    ):
        """Init the FineTune model.

        Args
        ----
        model : see self
        batch_size : int = 1000
        epochs : int = 25
        device : str | torch.device = 'cpu'
        dtype : torch.dtype = torch.float32
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expected model typed as `torch.nn.Module`, not {type(model)}'
            )

        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        # TODO this is 2nd time using this, should make it a func util.
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
            if isinstance(dtype, torch.dtype):
                self.dtype = dtype
            else:
                raise TypeError('Expected torch.dtype for dtype not: {dtype}')
        else:
            raise TypeError('Expected torch.dtype for dtype not: {dtype}')

    def fit(
        self,
        features_t,
        labels_t,
        features_v=None,
        labels_v=None,
        verbose=False,
    ):
        """Fits the model with fit_args and the given features and labels in a
        supervised learning fashion.
        features_t labels_t:
            features and labels that the model should be trained on.
        features_v labels_v:
            features and labels that the model should be validated on.
        """

        # TODO move to pytorch_lightning
        # trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
        # trainer.fit(model, train_loader, val_loader)

        features_t = features_t.float()
        t_len = len(features_t.float())
        dataset = torch.utils.data.TensorDataset(features_t, labels_t)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if features_v is not None:
            v_len = len(features_v)
            print(v_len)

            dataset_val = torch.utils.data.TensorDataset(features_v, labels_v)
            dataloader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=self.batch_size,
                shuffle=True,
            )

        model = self.model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        best_cls_loss = 99999999999999999
        best_val_acc = 0

        for epoch in range(self.epochs):
            if verbose:
                print("Epoch: " + str(epoch) + "---------------")
            tot_cls_loss = 0.0
            right = 0
            for i, x in enumerate(dataloader):
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
                loss = criterion(prediction, slabels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tot_cls_loss += loss.item()
            tacc = str(right / t_len)
            right = 0
            tot_cls_loss = 0.0

            if features_v is None: # Skip validation
                continue

            for i, x in enumerate(dataloader_val):
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
                loss = criterion(prediction, slabels)
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

    def extract(self, features):
        """Given features, outputs the fully connected encoding of them.
        Args
        ----
        features : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.model.fcs(features.to(self.device, dtype=self.dtype))

    def predict(self, features):
        # TODO If this eval/fwd pass loop overlaps with training loop, reuse
        # the code by function call, otherwise just do eval loop here. more
        # likely the training loop would use this at least in validation.

        # NOTE for our paper, we want this with ability to find a threshold
        # from all train and val data.
        features, prediction = self.model(features)
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
        dropout_prob=None,
    ):
        """Fine-tuning ANN consisting of fully-connected dense layers.

        Args
        ----
        input_size : int
            The input size of the input linear layer.
        width : int = 512
            The width of the hidden layers within the ANN.
        depth : int = 5
            The depth or number of hidden layers within the ANN.
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
        dropout_prob : float = None
            The probability for the dropout layers after the linear layers.
            Defaults to None, meaning no dropout is applied.
        """
        super().__init__()
        if out_features is None:
            out_features = width

        ord_dict = OrderedDict()
        ord_dict["fc0"] = nn.Linear(input_size, width)
        ord_dict[f"{activation.__name__}0"] = activation()

        for x in range(1, depth-1):
            ord_dict[f'fc{x}'] = nn.Linear(width, width)
            if dropout_prob:
                ord_dict[f'Dropout{x}'] = nn.Dropout(dropout_prob)
            ord_dict[f'{activation.__name__}{x}'] = activation()

        # Final dense / fully connected layer as output feature representation
        ord_dict[f'fc{depth - 1}'] = nn.Linear(width, out_features)

        self.fcs = nn.Sequential(ord_dict)
        self.classifier = nn.Linear(out_features, n_classes)

    def forward(self, x):
        """Returns the last fully connected layer and the probs classifier"""
        x = self.fcs(x)
        classification = F.log_softmax(self.classifier(x), dim=1)
        return x, classification

    def load_interior_weights(self, state_dict):
        temp = []
        for x in state_dict:
            if x[:3] != 'fcs':
                temp.append(x)
        for x in temp:
            state_dict.pop(x)
        self.load_state_dict(state_dict, strict=False)


# TODO pytorch lightning for experience and speed, replace/new FineTune?
"""
import pytorch_lightning as pl
class FineTuneFCLit(FineTuneFC, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adam(self.parameters(), **kwargs)

    def training_step(self, batch, batch_idx, df):
        inputs, indices = batch
        labels = df.iloc[indices]['labels']

        fine_tune_reprs, classifications = self(inputs)

        loss = self.loss(classifications, labels)
        #logging.info('Training loss: %d', loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()
#"""
