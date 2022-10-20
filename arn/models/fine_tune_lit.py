"""FineTune written in Pytorch Lightning for simplicty."""
from functools import partial
from collections import OrderedDict
import json

import numpy as np
import pytorch_lightning as pl
import ray
from ray_lightning import RayPlugin
from scipy.optimize import minimize, minimize_scalar
import torch
nn = torch.nn
F = torch.nn.functional
import torchmetrics

from exputils.data import ConfusionMatrix

from arn.models.fine_tune import FineTuneFC
from arn.data.kinetics_unified import get_kinetics_uni_dataloader

import logging
logger = logging.getLogger(__name__)


def crossover_error_rate(cm, default=1.0):
    fpr, fnr = cm.false_rates(cm.label_enc.unknown_idx)
    if np.isnan([fpr, fnr]).any():
        return default
    # TODO FIX ME TO BE A SCALAR!!!! Currently a vecotr of dim 2!
    return (fpr - fnr)**2


def negative_mcc_squared(cm):
    return -(cm.mcc() ** 2)


def get_attr_measure(cm, measure='mcc'):
    return getattr(cm, measure)()


def unk_thresh_opt(
    threshold,
    targets,
    pred_labels,
    probs,
    label_enc,
    measure_func=None,
    copy=True,
    is_key=False,
):
    """Calculates the crossover error rate of known vs unknown with given
    threshold that converts predicted known labels into 'unknown' if their
    probability is less than the threshold.

    Args
    ----
    targets : np.ndarray
        An integer vector as the target labels as discrete elements within a
        vector.
    pred_labels : np.ndarray
        An integer vector as the predicted labels as discrete elements within a
        vector.
    probs : np.ndarray
        A float vector as the predicted labels' probability value for
        predictions.
    label_enc : NominalDataEncoder
        The predictor's label encoder, denoting which are unknown to it.
    measure_func : callable = None
        A callable that takes in the resulting reduced confusion matrix and the
        rest of the positional arguments. Defaults to crossover_error_rate.
    copy : bool = True
        Copies the pred_labels to prevent side effect of changing source tensor

    Returns
    -------
    float
        The measure func the data given the threshold.
    """
    if measure_func is None:
        measure_func = crossover_error_rate
    if copy:
        pred_labels = pred_labels.copy()
    pred_labels[probs < threshold] = label_enc.unknown_key if is_key else \
        label_enc.unknown_idx
    cm = ConfusionMatrix(
        targets,
        pred_labels,
        labels=label_enc,
    ).reduce(['unknown'], 'known', inverse=True)
    return measure_func(cm)


def binary_minimize(func, args=None, bounds=None, max_divisions=20):
    """Binary search of a bounded space that minimizes the ouput of a function.
    Assumes a monotonic postively increasing function, thus favoring lower
    values towards the left bound. Longest run time is capped by the number of
    max_divisions.

    Args
    ----
    func:
    args: list = None
        The list of positional arguments to provide to func after the first
        arg, which is the optimal float threshold being searched for.
    bounds: list = None
        Default is [0, 1].
    max_divisions: int = 20
        The maximum number of divisions to perform for the binary search of the
        minimum.  Fun fact, divisions is equal to the number of decimal places
        needed to contain the final result due to dividing by 2 every time.

    Returns
    -------
    float
        The resulting threshold that yields the minimum measurement/output of
        func given the arguments.
    """
    raise NotImplementedError(
        "Use `scipy.optimize.minimize_scalar(..., method='bounded')`. "
        'This code is kept for future interest as a thought problem to '
        'compare to Brent or Golden search methods.'
    )

    thresh_left, thresh_right = bounds
    measure_left = func(thresh_left, *args)
    measure_right = func(thresh_left, *args)

    for divs in range(max_divisions):
        thresh = np.floor((thresh_left + thresh_right) / 2.0)
        measure = func(thresh, *args)

        if measure < measure_left:
            thresh_right = thresh
            measure_right = measure
        else:
            thresh_left = thresh
            measure_left = measure

        if thresh_left > thresh_right:
            break

    return left_thresh


def find_unknown_threshold(
    targets,
    preds,
    label_enc,
    start_thresh=0,
    method='bounded',
    bounds=None,
    maxiter=50,
):
    """Binary search for the best threshold to minimize the crossover error
    rate of binary known versus unknown (or equal error rate) to level of
    precision desired. This prefers lower valued thresholds, erroring on
    the side of predicting known.

    Args
    ----
    targets : torch.Tensor
        The target labels as discrete elements within a vector.
    preds : torch.Tensor
        A matrix where rows are samples matching size of targets and columns
        are correspond the the labels in the label_enc.
    label_enc : NominalDataEncoder
        The predictor's label encoder, denoting which are unknown to it.
    start_thresh : floot = 0.0
        The starting threshold.
    method : str = 'bounded'
        If a str, the method to use for scipy.optimize.minimize. If an int,
        the number steps to use in np.linspace for the linear scan.

    Returns
    -------
    float
        The threshold found to yield the best binary classification performance
        on the knowns versus unknowns given the predictor's label encoder.
    """
    if bounds is None:
        bounds = [0.0, 1.0]

    if isinstance(method, int):
        # Basic linear scan of threshold w/in bound given number of steps.
        preds_argmax = preds.argmax(1)
        preds_max = preds.max(1)
        opt_thresh = None
        opt_measure = np.inf
        for thresh in np.linspace(bounds[0], bounds[1], method):
            measure = unk_thresh_opt(
                thresh,
                targets,
                preds_argmax,
                preds_max,
                label_enc,
            )
            if measure < opt_measure:
                opt_measure = measure
                opt_thresh = thresh
        return opt_thresh
    if method in {'brent', 'golden', 'bounded'}:
        return minimize_scalar(
            unk_thresh_opt,
            args=(
                targets,
                preds.argmax(1),
                preds.max(1),
                label_enc,
            ),
            bounds=bounds,
            method=method,
            options={'maxiter': maxiter},
        ).x
    if method == 'binary':
        # Perform binary search to the given float's precision
        return binary_minimize(
            unk_thresh_opt,
            args=(
                targets,
                preds.argmax(1),
                preds.max(1),
                label_enc,
            ),
            bounds=bounds,
            divisions=maxiter,
        )
    return minimize(
        unk_thresh_opt,
        start_thresh,
        args=(
            targets,
            preds.argmax(1),
            preds.max(1),
            label_enc,
        ),
        bounds=[bounds],
        method=method,
    ).x[0]


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
    min_epochs=1,
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
        Number of maximum epochs to use during fitting.
    min_epochs : int = 1
        Number of minimum epochs to use during fitting.
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
        min_epochs=min_epochs,
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
    expect_one_hot : bool = True
        Parameter to expect one hot when True, or the class index integer for
        labels when False. This to be False when using CrossEntropyLoss for
        older versions of torch, such as v1.7.
    unk_thresh : float = None
        If given, the unknown threshold that is used to compare the highest
        probability score in the classificaiton probability vector upon
        prediction. If the max probability of a single class is less than the
        threshold then the class predicted is to be `unknown` rather than
        whatever the highest class predicted is. Defaults to None for no
        threshold to be used.
    device : str = 'cuda'

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
        expect_one_hot=True,
        save_hyperparameters=False,
        unk_thresh=None,
        device='cuda',
        load_state=None,
        *args,
        **kwargs,
    ):
        """Initialize the FineTune model

        Args
        ----
        see self
        confusion_matrix : bool = False
            If True, records the ConfusionMatrix for train and val loops.
        load_state : str = None
        """
        super().__init__(*args, **kwargs)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.model = model
        self.expect_one_hot = expect_one_hot

        self.unk_thresh = unk_thresh

        #self.device = torch.device(device)

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
            self.train_cm = torchmetrics.ConfusionMatrix(
                self.model.classifier.out_features
            )
            self.val_cm = torchmetrics.ConfusionMatrix(
                self.model.classifier.out_features
            )
        else:
            self.train_cm = None
            self.val_cm = None

        self.train_mcc = torchmetrics.MatthewsCorrCoef(
                self.model.classifier.out_features
        )
        self.val_mcc = torchmetrics.MatthewsCorrCoef(
                self.model.classifier.out_features
        )

        if load_state is not None:
            self.load_from_checkpoint(load_state, model=self.model)

    @property
    def n_classes(self):
        return self.model.n_classes

    def get_hparams(self, indent=None):
        hp = dict(
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            model_hparams=self.model.get_hparams(indent),
        )
        if indent:
            hp['loss'] = str(hp['loss'])
            return json.dumps(hp, indent=indent)
        return hp

    def on_save_checkpoint(self, checkpoint):
        if self.unk_thresh is not None:
            checkpoint['unk_thresh'] = self.unk_thresh

    def on_load_checkpoint(self, checkpoint):
        if 'unk_thresh' in checkpoint:
            self.unk_thresh = float(checkpoint['unk_thresh'])
        else:
            self.unk_thresh = None

    def set_n_classes(self, *args, **kwargs):
        self.model.set_n_classes(*args, **kwargs)

        # This breaks use for epochs, but allows tracking during fit & val
        # with growing classes over increments.
        self.train_mcc = torchmetrics.MatthewsCorrCoef(
                self.model.classifier.out_features
        )
        self.val_mcc = torchmetrics.MatthewsCorrCoef(
                self.model.classifier.out_features
        )

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

        if len(classifications.shape) == 1:
            classifications = classifications.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        if self.expect_one_hot:
            labels_argmax = labels.argmax(1)
        else:
            labels_argmax = labels

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

        loss = self.loss(classifications, labels_argmax)
        #logger.debug('loss.requires_grad: %s', loss.requires_grad)
        classif_argmax = F.softmax(classifications, 1).argmax(1)
        acc = (labels_argmax == classif_argmax).to(float).mean()

        self.log('train_loss', loss)
        #self.log('train_accuracy', acc)

        self.log('train_acc_step', self.train_acc(classif_argmax, labels_argmax))
        self.log('train_mcc_step', self.train_mcc(classif_argmax, labels_argmax))
        #self.log('train_cm_step', self.train_cm(classif_argmax, labels_argmax))

        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == 1:
            self.logger.experiment.add_graph(
                self.model,
                torch.rand((1,1, self.model.fcs.fc0.in_features)).to(self.device),
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
        #self.log('train_mcc_epoch', self.train_mcc.compute())

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list) and len(batch) in {1, 2}:
            # Batch len 2 is allowed as hotfix cuz using ANNEVM extract in fit
            batch = batch[0]
        fine_tune_reprs, classifs = self.model(batch)
        return fine_tune_reprs, F.softmax(classifs, 1)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        fine_tune_reprs, classifications = self.model(inputs)

        if len(classifications.shape) == 1:
            classifications = classifications.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        if self.expect_one_hot:
            labels_argmax = labels.argmax(1)
        else:
            labels_argmax = labels

        loss = self.loss(classifications, labels_argmax)

        classif_argmax = F.softmax(classifications, 1).argmax(1)

        acc = (labels_argmax == classif_argmax).to(float).mean()

        #logging.info('Training loss: %d', loss)
        self.log('val_loss', loss)
        #self.log('val_accuracy', acc)

        self.log('val_acc_step', self.val_acc(classif_argmax, labels_argmax))
        self.log('val_mcc_step', self.val_mcc(classif_argmax, labels_argmax))
        #self.log('val_cm_step', self.val_cm(classif_argmax, labels_argmax))

        return OrderedDict({'loss': loss, 'accuracy': acc})

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


# TODO need to add the option to SAVE the ckpt of a given file for FineTuneFCLit
#   specifically, at every step/increment of KOWL! Put w/in OWHAPredictor???
# TODO need to add the option to LOAD the ckpt of a given file for FineTuneFCLit


def load_fine_tune_fc_lit(checkpoint_path):
    """Loads a FineTuneFCLit model from the given checkpoint filepath.

    Args
    ----
    checkpoint_path : str
        Filepath to the checkpoint to load.

    Returns
    -------
    FineTuneFCLit
        The loaded FineTuneLit object.
    """
    return FineTuneFCLit.load_from_checkpoint(checkpoint_path)


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
    predict_batch_size : int = None
        An optional batch size for predict() only.
    shuffle : bool = True
        If True, shuffle the data when fitting. If False, no shuffling.
    num_workers : int = 0
        Number of works to ues for the DataLoader.
    pin_memory : bool = False
        Pin memory for data loaders.
    unk_thresh_method : str = 'bounded'
        Measure function for finding unknown threshold post-prediction using
        scipy.optimize.minimize_scalar. Only performs finding of threshold when
        fit is given a label enc and a validation dataset.
    device : str = 'cuda'
    """
    def __init__(
        self,
        model,
        trainer,
        batch_size=1000,
        predict_batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        unk_thresh_method='bounded',
        device='cuda',
    ):
        """Init the FineTune model.

        Args
        ----
        see self
        """
        #if isinstance(model, str):
        #    self.model = FineTuneFCLit.load_from_checkpoint(model)
        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            raise TypeError(
                'Expected model typed as `torch.nn.Module`, not {type(model)}'
            )
        self.device = torch.device(device)
        self.batch_size = batch_size
        if predict_batch_size:
            self.predict_batch_size = predict_batch_size
        else:
            self.predict_batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.unk_thresh_method = unk_thresh_method

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
            self.model.to(self.device)
        #"""

        # Record HParams
        with torch.utils.tensorboard.writer.SummaryWriter(
            log_dir=self.trainer.log_dir,
        ) as writer:
            writer.add_text(
                'hparams',
                self.get_hparams(indent=4),
                self.trainer.global_step,
            )

    @property
    def n_classes(self):
        return self.model.n_classes

    def get_hparams(self, indent=None):
        hp = dict(
            batch_size=self.batch_size,
            max_epochs=self.trainer.max_epochs,
            predict_batch_size=self.predict_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            model_hparams=self.model.get_hparams(indent),
        )
        if indent:
            return json.dumps(hp, indent=indent)
        return hp

    def fit(self, dataset, val_dataset=None, label_enc=None, reset_epoch=True):
        """Fit the fine tuning model with the given train and val datasets.

        Args
        ----
        dataset : KineticsUnifiedFeatures | torch.utils.data.DataLoader
            The dataset to be turned into a DataLoader or the DataLoader itself
            used for fitting the model.
        val_dataset : KineticsUnifiedFeatures
            Same as `dataset`, except used for validation during the fitting
            process.
        reset_epoch : bool = True
            If True, resets the epochs before fitting.
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

        if reset_epoch:
            self.trainer.fit_loop.epoch_progress.reset_on_epoch()

        self.trainer.fit(
            model=self.model,
            train_dataloaders=dataset,
            val_dataloaders=val_dataset,
        )

        # Find threshold for unknowns on val_dataset combined with train
        if (
            self.model.unk_thresh is not None
            and label_enc is not None
            and val_dataset is not None
        ):
            targets = np.concatenate([
                dataset.dataset.data[dataset.dataset.label_col].values,
                val_dataset.dataset.data[val_dataset.dataset.label_col].values,
            ]).squeeze()
            preds = torch.cat([
                self.predict(dataset),
                self.predict(val_dataset),
            ]).detach().numpy().squeeze()

            self.model.unk_thresh = find_unknown_threshold(
                targets,
                preds,
                label_enc,
                start_thresh=self.model.unk_thresh,
                method=self.unk_thresh_method,
            )

    def set_n_classes(self, *args, **kwargs):
        self.model.set_n_classes(*args, **kwargs)

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
                batch_size=self.predict_batch_size,
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
        if self.predict_batch_size > 1:
            return torch.cat([t[1] for t in self._predict(features)])
        return torch.stack([t[1] for t in self._predict(features)])

    def extract(self, features):
        if self.predict_batch_size > 1:
            return torch.cat([t[0] for t in self._predict(features)])
        return torch.stack([t[0] for t in self._predict(features)])

    def extract_predict(self, features):
        feature_extracts = []
        preds = []
        for extract, pred in self._predict(features):
            feature_extracts.append(extract)
            preds.append(pred)
        if self.predict_batch_size > 1:
            return torch.cat(feature_extracts), torch.cat(preds)
        return torch.stack(feature_extracts), torch.cat(preds)
