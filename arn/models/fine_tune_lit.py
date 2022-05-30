"""FineTune written in Pytorch Lightning for simplicty."""
from collections import OrderedDict

import pytorch_lightning as pl
import ray
from ray_lightning import RayPlugin
import torch
nn = torch.nn
F = torch.nn.functional
import torchmetrics

from arn.models.fine_tune import FineTuneFC
from arn.data.kinetics_unified import get_kinetics_uni_dataloader

import logging
logger = logging.getLogger(__name__)


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
    strategy=None,
    logger=None,
    log_every_n_steps=50,
    #flush_logs_every_n_steps=None,
    track_grad_norm=-1,
    num_sanity_val_steps=2,
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
    strategy : str = None
    logger : init_tensorboard_logger = None
    log_every_n_steps : int = 50
    track_grad_norm : int = -1
    num_sanity_val_steps : int = 2

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
        strategy=strategy,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        track_grad_norm=track_grad_norm,
        num_sanity_val_steps=num_sanity_val_steps,
    )


#class FineTuneFCLit(FineTuneFC, pl.LightningModule):
class FineTuneFCLit(pl.LightningModule):
    """The FineTune model contained within a pl.LightningModule

    Attributes
    ----------
    model : FineTuneFC
        The model of this pytorch lightning module.
    loss : torch.nn.modules.loss._Loss = None
        TODO in docstr add support for `see FineTuneFC.__init__`
    lr : float = 0.001
        The learning rate for the optimizer.
    weight_decay : float = 0
        The learning rate for the optimizer.
    amsgrad : bool = False
        If True, uses AMSGrad variant of ADAM.

    Notes
    -----
    todo `see FineTuneFC`
        In docstr, add see across classes/objects in general, not just in
        functions for self.
    """
    def __init__(
        self,
        model,
        loss=None,
        lr=0.001,
        weight_decay=0,
        amsgrad=False,
        confusion_matrix=False,
        *args,
        **kwargs,
    ):
        """Initialize the FineTune model

        Args
        ----
        see self
        confusion_matrix : bool = False
            If True, records the ConfusionMatrix for train and val loops.
        """
        super().__init__(*args, **kwargs)
        self.model = model

        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

        self.lr = lr
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        if confusion_matrix:
            self.train_cm = torchmetrics.ConfusionMatrix(401)
            self.val_cm = torchmetrics.ConfusionMatrix(401)
        else:
            self.train_cm = None
            self.val_cm = None

        self.train_mcc = torchmetrics.MatthewsCorrCoef(401)
        self.val_mcc = torchmetrics.MatthewsCorrCoef(401)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self.model(inputs)

        #ipdb.set_trace()
        if len(classifications.shape) == 1:
            classifications = classifications.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        #print(labels.argmax(1).unique())
        #print(F.softmax(classifications, 1).argmax(1).unique())

        """
        logger.debug('within training_step()')
        logger.debug('self.training: %s', self.training)
        logger.debug('self.model.training: %s', self.model.training)
        logger.debug(
            'classifications.requires_grad: %s',
            classifications.requires_grad,
        )
        logger.debug('labels.requires_grad: %s', labels.requires_grad)
        #"""

        loss = self.loss(classifications, labels)
        #logger.debug('loss.requires_grad: %s', loss.requires_grad)
        labels_argmax = labels.argmax(1)
        classif_argmax = F.softmax(classifications, 1).argmax(1)
        acc = (labels_argmax == classif_argmax).to(float).mean()

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        self.log('train_acc_step', self.train_acc(classif_argmax, labels_argmax))
        self.log('train_mcc_step', self.train_mcc(classif_argmax, labels_argmax))
        #self.log('train_cm_step', self.train_cm(classif_argmax, labels_argmax))

        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == 1:
            self.logger.experiment.add_graph(
                self.model,
                torch.rand((1,1, self.model.fcs.fc0.in_features)).to('cuda'),
            )
        for name, params in self.named_parameters():
            # Log Weights
            self.logger.experiment.add_histogram(
                name,
                params,
                self.current_epoch,
            )

            # Log Gradients
            """
            self.logger.experiment.add_histogram(
                f'{name}-grad',
                params.grad,
                self.current_epoch,
            )
            #"""

        self.log('train_acc_epoch', self.train_acc.compute())

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        fine_tune_reprs, classifs = self.model(batch)
        return fine_tune_reprs, F.softmax(classifs, 1)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self.model(inputs)

        if len(classifications.shape) == 1:
            classifications = classifications.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        loss = self.loss(classifications, labels)

        labels_argmax = labels.argmax(1)
        classif_argmax = F.softmax(classifications, 1).argmax(1)

        acc = (labels_argmax == classif_argmax).to(float).mean()

        #logging.info('Training loss: %d', loss)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        self.log('val_acc_step', self.val_acc(classif_argmax, labels_argmax))
        self.log('val_mcc_step', self.val_mcc(classif_argmax, labels_argmax))
        #self.log('val_cm_step', self.val_cm(classif_argmax, labels_argmax))

        return OrderedDict({'loss': loss, 'accuracy': acc})

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


class FineTuneLit(object):
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

        #self.trainer = pl.Trainer(*args, **kwargs)
        self.trainer = trainer

        #"""
        if (
            self.trainer._accelerator_connector.strategy is not None
            and not isinstance(self.trainer._accelerator_connector.strategy, str)
            and self.trainer._accelerator_connector.strategy.use_gpu
        ):
            self.model.to('cuda')
        #"""

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
                shuffle=False, #self.shuffle,
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
                batch_size=self.batch_size,#1,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
            ),
            return_predictions=True,
        )
        logger.debug('type(preds): %s', type(preds))
        logger.debug('len(preds): %d', len(preds))

        logger.debug('type(preds[0]): %s', type(preds[0]))
        logger.debug('len(preds[0]): %d', len(preds[0]))

        logger.debug('type(preds[0][0]): %s', type(preds[0][0]))
        logger.debug('preds[0][0].shape %s', preds[0][0].shape)

        logger.debug('type(preds[0][1]): %s', type(preds[0][1]))
        logger.debug('preds[0][1].shape: %s', preds[0][1].shape)

        #if reset_strategy:
        #    self.trainer.training_type_plugin = reset_strategy
        return preds

    def predict(self, features):
        if self.batch_size > 1:
            return torch.concat([t[1] for t in self._predict(features)])
        return torch.stack([t[1] for t in self._predict(features)])

    def extract(self, features):
        if self.batch_size > 1:
            return torch.concat([t[0] for t in self._predict(features)])
        return torch.stack([t[0] for t in self._predict(features)])
