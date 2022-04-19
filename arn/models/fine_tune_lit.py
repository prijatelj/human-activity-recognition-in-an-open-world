"""FineTune but in Pytorch Lightning for simplicty."""
import pytorch_lightning as pl
import torch
nn = torch.nn

from arn.models.fine_tune import FineTuneFC
from arn.torch_utils import torch_dtype


class FineTuneFCLit(FineTuneFC, pl.LightningModule):
    """The FineTune model exteneded as a pl.LightningModule

    Attributes
    ----------
    see FineTuneFC
    """
    def __init__(self, *args, loss=None, **kwargs):
        """Initialize the FineTune model

        Args
        ----
        see FineTuneFC
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
        raise NotImplementedError()

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
    batch_size : int = 1000
    epochs : int = 25
        Number of epochs to use during fitting.
    device : str | torch.device = 'cpu'
        the device on which model should be trained
        default: cpu
    dtype : torch.dtype = torch.float32
    shuffle : bool = True
        If True, shuffle the data when fitting. If False, no shuffling.
    """
    def __init__(
        self,
        model,
        batch_size=1000,
        epochs=25,
        device='cpu',
        dtype=torch.float32,
        shuffle=True,
        *args,
        **kwargs,
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

        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)

        self.trainer = pl.Trainer(*args, **kwargs)

    def fit(self, dataset, val_dataset=None):
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        if isinstance(val_dataset, torch.utils.data.Dataset):
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        elif not isinstance(val_dataset, torch.utils.data.DataLoader):
            val_loader = None

        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    def predict(self, features):
        return self.model(features.to(self.device, self.dtype))[1]

    def extract(self, features):
        return self.model(features.to(self.device, self.dtype))[0]
