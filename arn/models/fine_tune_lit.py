"""FineTune written in Pytorch Lightning for simplicty."""
import pytorch_lightning as pl
import torch
nn = torch.nn

from arn.models.fine_tune import FineTuneFC
from arn.torch_utils import torch_dtype
from arn.data.kinetics_unified import get_kinetics_uni_dataloader


def init_trainer(
    default_root_dir=None,
    enable_checkpointing=True,
    gpus=None,
    max_epochs=1000,
    logger=None,
):
    """Hotfix docstr workaround for not being able to read Torch docs and not
    being able to accept/parse uknown kwargs to be passes as **kwargs.

    Args
    ----
    default_root_dir : str = None
    enable_checkpointing : bool = True
    gpus : int = None
    max_epochs : int = 1000
        Number of epochs to use during fitting.
    logger : pytorch_lightning.loggers.TensorBoardLogger = None

    Returns
    -------
    pytorch_lightning.Trainer
        docstr TODO check the module's namespace to support `as pl` and then
        pl.Trainer in docs.
    """
    return pl.Trainer(
        default_root_dir=default_root_dir,
        enable_checkpointing=enable_checkpointing,
        gpus=gpus,
        max_epochs=max_epochs,
        logger=logger,
    )


class FineTuneFCLit(FineTuneFC, pl.LightningModule):
    """The FineTune model exteneded as a pl.LightningModule

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

    TODO
    ----
    todo `see FineTuneFC`
        In docstr, add see across classes/objects in general, not just in
        functions for self.
    """
    def __init__(self, loss=None, *args, **kwargs):
        """Initialize the FineTune model

        Args
        ----
        loss : torch.nn.modules.loss._Loss = None
            TODO in docstr add support for `see FineTuneFC.__init__`
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
        super().__init__(*args, **kwargs)

        if loss is None:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = loss

    def configure_optimizers(self, optimizer_cls=None, **kwargs):
        if optimizer_cls is None:
            return torch.optim.Adam(self.parameters(), **kwargs)
        if issubclass(optimizer_cls, torch.optim.Optimizer):
            return optimizer_cls(self.parameters(), **kwargs)
        raise TypeError(' '.join([
            'Expected `optimizer_cls` subclass `torch.optim.Optimizer`,',
            f'but recieved {optimizer_cls}',
        ]))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self(inputs)

        loss = self.loss(classifications, labels)

        #logging.info('Training loss: %d', loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self(inputs)

        loss = self.loss(classifications, labels)

        #logging.info('Training loss: %d', loss)
        self.log('train_loss', loss)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


class FineTuneLit():
    """FineTune modified to manage Pytorch Lightning models.

    Attributes
    ----------
    model : FineTuneFCLit
        docstr needs to support subclasses of a given class: torch.nn.Module
        The model to be used for fine tuning. This is expected to support
        FineTuneFC.
    trainer : init_trainer
        The pl.Trainer trainer used for the fine tune model.
    batch_size : int = 1000
    device : str | torch.device = 'cpu'
        the device on which model should be trained
        default: cpu
    dtype : torch.dtype = torch.float32
    shuffle : bool = True
        If True, shuffle the data when fitting. If False, no shuffling.
    num_workers : int = 0
        Number of works to ues for the DataLoader.
    """
    def __init__(
        self,
        model,
        trainer,
        batch_size=1000,
        device='cpu',
        dtype=torch.float32,
        shuffle=True,
        num_workers=0,
        #*args,
        #**kwargs,
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
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)

        #self.trainer = pl.Trainer(*args, **kwargs)
        self.trainer = trainer

    def fit(self, dataset, val_dataset=None):
        """Fit the fine tuning model with the given train and val datasets.

        Args
        ----
        dataset : KineticsUnifiedFeatures | torch.utils.data.DataLoader
            The dataset to be turned into a DataLoader or the DataLoader itself
            used for fitting the model.
        val_dataset : KineticsUnifiedFeatures
            Same as `dataset`, except used for validation during the fitting
            process.
        """
        dataset = get_kinetics_uni_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        if val_dataset is not None:
            val_dataset = get_kinetics_uni_dataloader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=dataset,
            val_dataloaders=val_dataset,
        )

    def predict(self, features):
        return self.trainer.predict(features)[1]

    def extract(self, features):
        return self.trainer.predict(features)[0]
