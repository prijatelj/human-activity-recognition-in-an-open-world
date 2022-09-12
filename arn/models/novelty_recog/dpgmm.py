"""A Dirichlet Process Gaussian Mixture Model implemeted in torch and pyro that
performs open world recognition (classification).
"""
from copy import deepcopy

import torch
F = torch.nn.functional
MultivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal

import pyro

from exputils.data.labels import NominalDataEncoder

from arn.models.novelty_recog.predictor import OWHARecognizer
from arn.torch_utils import torch_dtype

import logging
logger = logging.getLogger(__name__)


class DPGMMRecognizer(OWHARecognizer):
    """A Dirichlet Process Gaussian Mixture Model for open world recognition.
    A Dirichlet Process is the model of the class distribution and a
    multivariate normal is the distribution per class. Both are as simple as
    it gets to modeling a solution to this problem in probability theory. The
    Dirichlet process allows for learning new groups in unlabeled data or data
    with unknown classes. The multivariate normal distribution per class
    simplifies the class distribution to be easily managable and charaterized
    by the location vector and a scale (covariance) matrix.

    Attributes
    ----------
    _comp_probs : torch.Tensor
        The probability of each component of the DPGMM where each known and
        recognized unknown class gets a component MVN.
    _mvns : list(list(torch.Tensor))
        A list of parameters for the multivariate normal distributions per
        category.
    trainer :
        The torch internals for training the DPGMM using MCMC.
    dtype : str = None
        The dtype to use for the MultivariateNormal calculations based on the
        class features. Sets each class_features per known class to this dtype
        prior to finding the torch.tensor.mean() or torch.tensor.cov().
    device : str = None
        The device on which the internal tensors are stored and calculations
        are performed. When None, default, it is inferred upon fitting.
    see OWHARecognizer
    """
    def __init__(
        self,
        **kwargs,
    )
        super().__init__(**kwargs)

        self._categories = []
        self._mvns = []
        self.trainer = None

        if dtype is not None:
            self.dtype = torch_dtype(dtype)
        else:
            self.dtype = dtype
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = device

    def update_experience(self, dataset):
        """Updates the experience of the predictor."""
        raise NotImplementedError()

        # TODO adds known labels per sample, adds most recent unknown label

    def fit(self, dataset, val_dataset=None, **kwargs):
        """Fits the given data with known and unknown/unlabeled data.

        Args
        ----
        see OWHARecognizer.fit

        Returns
        -------
        torch.Tensor
        """
        # TODO updates experience with any new data (esp if called directly)
        self.update_experience()

        # TODO if any known labels, then fit an MVN per known class

        # TODO if any unknown/unlabeled data, then fit the DPGMM
        self._recognize_fit(dataset, **kwargs)

    def _recognize_fit(self, features, n_expected_classes=None, **kwargs):
        """Fits the DPGMM given the data and frozen known class components.

        Args
        ----
        dataset :
            The feature points with labels to be fit.
        n_expected_classes : None
        """
        # TODO fits the DPGMM given frozen known class components, with any
        # prior on previously recognized components

        # TODO Update recog_label_enc to new/modified recognized unknown labels
        # TODO updates experience with any new data (esp if called directly)
        self.update_experience() # just points with recogs, no labels

    def predict(self, dataset, **kwargs):
        """Performs inference of task given the data. Updates via recognize.

        Args
        ----
        see OWHARecognizer.predict

        Returns
        -------
        torch.Tensor
            A probability vector per sample (row) that starts with known
            classes and follows with all recognized unknown classes.
        """
        self._recognize_fit(dataset, **kwargs)

        # TODO get the probability a sample belongs to each component and
        # normalize

        return preds

    def detect(self, features, is_pred=False):
        """Inference on task reduced to known vs unknown classification.

        Args
        ----
        see OWHARecognizer.detect

        Returns
        -------
        torch.Tensor
            Each row is a single probability score of the sample being an
            unknown class.
        """
        if not is_pred:
            features = self.predict(features)

        # TODO Perform a likelihood ratio comparison per sample if it belongs
        # to the known classes' components or the unknown classes' components

        return
