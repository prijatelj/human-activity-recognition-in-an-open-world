"""FineTune written in Pytorch Lightning for simplicty."""
import logging

import pytorch_lightning as pl
import ray
from ray_lightning import RayPlugin
import torch
nn = torch.nn
F = torch.nn.functional

from arn.models.fine_tune import FineTuneFC
from arn.torch_utils import torch_dtype
from arn.data.kinetics_unified import get_kinetics_uni_dataloader


def init_ray_plugin(
    num_workers=1,
    num_cpus_per_worker=1,
    use_gpu=False,
):
    """Hotfix docstr workaround for not being able to read these docs and not
    being able to accept/parse uknown kwargs to be passes as **kwargs.

    Args
    ----
    num_workers : int = 1
    num_cpus_per_worker : int = 1
    use_gpu : bool = False

    Returns
    -------
    ray_lightning.RayPlugin
        The configured RayPlugin
    """
    #ray.init()
    return RayPlugin(
        num_workers=num_workers,
        num_cpus_per_worker=num_cpus_per_worker,
        use_gpu=use_gpu,
    )


def init_tensorboard_logger(
    save_dir,
    name=None,
    version=None,
    sub_dir=None,
    kwargs=None,
):
    """Hotfix docstr to init pl.loggers.tensorboard.TensorBoardLogger

    Args
    ----
    save_dir: str = None
    name: str = None
    version: str = None
    sub_dir: str = None
    kwargs: dict = None
    """
    if kwargs is None:
        return pl.loggers.TensorBoardLogger(
            save_dir,
            name=name,
            version=version,
            sub_dir=sub_dir,
        )
    else:
        return pl.loggers.TensorBoardLogger(
            save_dir,
            name=name,
            version=version,
            sub_dir=sub_dir,
            **kwargs,
        )


def init_trainer(
    default_root_dir=None,
    enable_checkpointing=True,
    gpus=None,
    max_epochs=1000,
    logger=None,
    strategy=None,
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
    logger : init_tensorboard_logger = None
    strategy : str = None

    Returns
    -------
    pytorch_lightning.Trainer
        docstr TODO check the module's namespace to support `as pl` and then
        pl.Trainer in docs.

    Notes
    -----
    strategy : init_ray_plugin = None
        ray plugin does not support pl.Trainer.predict()
    """
    return pl.Trainer(
        default_root_dir=default_root_dir,
        enable_checkpointing=enable_checkpointing,
        gpus=gpus,
        max_epochs=max_epochs,
        logger=logger,
        strategy=strategy,
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

    Notes
    -----
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
            #self.loss = nn.BCEWithLogitsLoss()
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        #fine_tune_reprs, classifications = self(torch.rand(inputs.shape).to('cuda'))

        #print(inputs)
        #print('labels shape: ', labels.shape)
        #print('classifications shape: ', classifications.shape)
        #print(labels.argmax(1))

        loss = self.loss(classifications, labels)
        acc = (
            labels.argmax(1) == F.softmax(classifications, 1).argmax(1)
        ).to(float).mean()

        #print(F.softmax(classifications, 1).argmax(1))

        #logging.info('Training loss: %d', loss)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        fine_tune_reprs, log_softmax_classifs = self(batch)
        #return fine_tune_reprs, torch.exp(log_softmax_classifs)
        return fine_tune_reprs, F.softmax(log_softmax_classifs, 1)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self(inputs)

        loss = self.loss(classifications, labels)
        acc = (
            labels.argmax(1) == F.softmax(classifications, 1).argmax(1)
        ).to(float).mean()

        #logging.info('Training loss: %d', loss)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        return {'loss': loss, 'accuracy': acc}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        if self.current_epoch == 1:
            self.logger.experiment.add_graph(
                self,
                torch.rand((1,1, 2048)).to('cuda'),
            )


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
    pin_memory : bool = False
        Pin memory for data loaders.
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
        pin_memory=False,
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

        # Multiprocessing params for DataLoaders
        self.pin_memory = pin_memory #num_workers > 0

        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)

        #self.trainer = pl.Trainer(*args, **kwargs)
        self.trainer = trainer

        if (
            self.trainer._accelerator_connector.strategy is not None
            and not isinstance(self.trainer._accelerator_connector.strategy, str)
            and self.trainer._accelerator_connector.strategy.use_gpu
        ):
            self.model.to('cuda')

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
            pin_memory=self.pin_memory,
        )

        if val_dataset is not None:
            val_dataset = get_kinetics_uni_dataloader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=dataset,
            val_dataloaders=val_dataset,
        )

    def _predict(self, features):
        # Work around because pl.Trainer.predict() does not support multiple
        # cpu processes for DataLoaders, but we want that for fitting, so if
        # set then have to turn it off when predicting.
        #reset_strategy = self.trainer._accelerator_connector.strategy
        """
        reset_strategy = self.trainer.training_type_plugin
        if reset_strategy:
            logging.debug(
                'self.trainer.training_type_plugin: %s',
                self.trainer.training_type_plugin,
            )
            self.trainer.training_type_plugin = None
        #"""
        preds = self.trainer.predict(
            model=self.model,
            dataloaders=get_kinetics_uni_dataloader(
                features,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
            ),
            return_predictions=True,
        )
        logging.debug('type(preds): %s', type(preds))
        logging.debug('len(preds): %d', len(preds))

        logging.debug('type(preds[0]): %s', type(preds[0]))
        logging.debug('len(preds[0]): %d', len(preds[0]))

        logging.debug('type(preds[0][0]): %s', type(preds[0][0]))
        logging.debug('preds[0][0].shape %s', preds[0][0].shape)

        logging.debug('type(preds[0][1]): %s', type(preds[0][1]))
        logging.debug('preds[0][1].shape: %s', preds[0][1].shape)

        #if reset_strategy:
        #    self.trainer.training_type_plugin = reset_strategy
        return preds

    def predict(self, features):
        return torch.stack([t[1] for t in self._predict(features)])

    def extract(self, features):
        return torch.stack([t[0] for t in self._predict(features)])
