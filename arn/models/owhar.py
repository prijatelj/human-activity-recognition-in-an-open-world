"""Open World Human Activity Recognition pipeline class."""
from datetime import datetime
from copy import deepcopy
import os

import torch

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

from arn.models.novelty_detector import WindowedMeanKLDiv
from arn.torch_utils import torch_dtype
from arn.data.kinetics_unified import KineticsUnifiedFeatures, KineticsUnified

import logging
logger = logging.getLogger(__name__)


def load_evm_predictor(*args, **kwargs):
    """Docstr hotfix cuz cannot easily link/use staticmethods as funcs.

    Args
    ----
    h5 : str
    skip_fit : bool = False
    """
    return EVMPredictor.load(load_cls=EVMPredictor, *args, **kwargs)


class EVMPredictor(ExtremeValueMachine):
    """Wraps the ExtremeValueMachine so it works with KineticsUnifiedFeatures.

    Attributes
    ----------
    skip_fit : bool = False
        If True, skips all calls to fit(), never training the EVM.
    uid : str = None
        The unique identifier to be used by EVMPredictor, if not given or given
        'datetime', will create the uid as
        `f"evm-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"`.
    save_dir : str = None
        If not None, will create the directory and save the EVM's state at that
        directory as `f'{self.uid}-{self.increment}.h5'` after every call to
        fit().
    see ExtremeValueMachine
    """
    def __init__(
        self,
        skip_fit=False,
        uid=None,
        save_dir=None,
        *args,
        **kwargs,
    ):
        """Docstr hotfix cuz otherwise this is unnecessary...

        Args
        ----
        skip_fit : see self
        uid : see self
        save_dir : see self
        see ExtremeValueMachine.__init__
        """
        super().__init__(*args, **kwargs)
        #self.store_preds # TODO speed up novelty detect from predict
        self.skip_fit = skip_fit
        self.uid = uid if isinstance(uid, str) and uid != 'datetime' \
            else f"evm-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"

        # TODO Enable checkpointing by providing a
        self.save_dir = create_filepath(save_dir)

        logger.info('Predictor UID `%s` init finished.', self.uid)

    def fit(self, dataset, val_dataset=None, *args, **kwargs):
        if self.skip_fit:
            return
        if val_dataset is not None:
            logger.warning(
                'Given a validation dataset, but '
                'the EVMPredictor does not support validation in fitting!'
            )
        # TODO handle being given unknowns in first dim of encoded labels!
        if isinstance(dataset, KineticsUnifiedFeatures):
            features = []
            labels = []
            extra_ns = []
            for feature, label in dataset:
                label = label.argmax(-1) - 1
                if label == -1: # Given unknown, rm from fitting.
                    extra_ns.append(feature)
                else:
                    features.append(feature)
                    labels.append(label)
            return super().fit(
                torch.stack(features), #torch.cat(features),
                torch.stack(labels), #torch.cat(labels),
                None if not extra_ns else torch.cat(extra_ns), #torch.stack(extra_ns),
                *args,
                **kwargs,
            )
        super().fit(dataset, *args, **kwargs)

        if self.save_dir:
            self.save(
                os.path.join(self.save_dir, f'{self.uid}-{self.increment}.h5')
            )

    def predict(self, features, unknown_last_dim=False):
        if isinstance(features, KineticsUnifiedFeatures):
            return super().predict(
                #torch.cat(list(features)),
                torch.stack(list(features)),
                unknown_last_dim,
            )
        return super().predict(features, unknown_last_dim)

    def novelty_detect(self, features, unknown_last_dim=False):
        unknown_dim = -1 if unknown_last_dim else 0

        if isinstance(features, KineticsUnifiedFeatures):
            return super().predict(
                #torch.cat(list(features)),
                torch.stack(list(features)),
                unknown_last_dim,
            )[:, unknown_dim]
        return super().predict(features,  unknown_last_dim)[:, unknown_dim]


class OWHAPredictor(object):
    """The OWHAR predictor class that contains and manages the predictor parts.
    Every OWL predictor consists of a feature representation model, fine tuner,
    and novelty detector. Extra parts include a novelty recognizer if separate
    from the pre-existing parts and optional feedback interpreter.

    This really should define the interface as an abstract/generic and be
    inheritted by any predictor class, as it is essentially being used now.

    Attributes
    ----------
    fine_tune: arn.models.fine_tune_lit.FineTuneLit
        fine_tune: arn.models.fine_tune.FineTune
    novelty_detector: WindowedMeanKLDiv = None
    feedback_interpreter: arn.models.feedback.CLIPFeedbackInterpreter = None
    label_enc : NominalDataEncoder = None
    _uid : str = None
        An optional str unique identifier for this predictor. When not
        given, the uid property of this class' object is the trainer
        tensorboard log_dir version number.
    skip_fit : int = -1
        If >= 0, skips fitting the model starting on the increment that matches
        skip_fit's value. Thus, skip_fit == 0 skips all fitting, skip_fit == 1
        skips all fitting after the first increment/starting data, 0th index.
        skip_fit serves as an exclusive upperbound to the range of incremental
        fitting allowed for this predictor. Values less than zero does not
        result in any skipping of calls to fit().
    """
    def __init__(
        self,
        fine_tune,
        novelty_detector=None,
        feedback_interpreter=None,
        dtype=None,
        label_enc=None,
        uid=None,
        skip_fit=-1,
    ):
        """Initializes the OWHAR.

        Args
        ----
        fine_tune: see self
        novelty_detector: see self
        feedback_interpreter: see self
        label_enc : str | NominalDataEncoder = None
            Filepath to be loaded or the actual NominalDataEncoder.
        uid : str = None
            An optional str unique identifier for this predictor. When not
            given, the uid property of this class' object is the trainer
            tensorboard log_dir version number. If 'datetime', saves datetime
            of init.
        skip_fit : see self
        """
        self.fine_tune = fine_tune
        self.novelty_detector = novelty_detector
        self.feedback_interpreter = feedback_interpreter
        self._increment = 0
        self.skip_fit = skip_fit
        self._uid = uid if uid != 'datetime'  \
            else f"owhap-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"
        self.dtype = torch_dtype(dtype)
        # TODO Use the dtype in all interior predictor parts unless None.
        if isinstance(label_enc, str):
            self.label_enc = NominalDataEncoder.load(label_enc)
        else:
            self.label_enc = label_enc

        logger.info('Predictor UID `%s` init finished.', self.uid)

    @property
    def increment(self):
        """Increments correspond to how many times the predictor was fit."""
        return self._increment

    @property
    def uid(self):
        """Returns a string Unique Identity for this predictor."""
        if self._uid:
            return self._uid
        return self.fine_tune.trainer.log_dir.rpartition(os.path.sep)[-1]

    def fit(self, dataset, val_dataset=None, task_id=None):
        """Incrementally fit the OWHAPredictor's parts. Update classes in
        classifier to match the training dataset. This assumes the training
        dataset contains all prior classes. This deep copy is convenient for
        ensuring the class indices are always aligned.
        """
        if self.skip_fit >= 0 and self._increment >= self.skip_fit:
            return
        if isinstance(dataset, KineticsUnified):
            if self.label_enc is None:
                self.fine_tune.set_n_classes(len(dataset.label_enc))
                self.label_enc = deepcopy(dataset.label_enc)
            elif set(dataset.label_enc) - set(self.label_enc):
                n_classes = len(dataset.label_enc)
                if n_classes != len(self.label_enc):
                    self.fine_tune.set_n_classes(n_classes)
                self.label_enc = deepcopy(dataset.label_enc)
        self._increment += 1
        self.fine_tune.fit(dataset, val_dataset=val_dataset)

    def predict(self, dataset, task_id=None):
        """Predictor performs the prediction (classification) tasks given
        dataset.

        Args
        ----
        dataset : torch.Dataset
        task_id : str = None
            The str identifier of the task to perform with the given inputs.
            This assumes the proper dataset input format is given for each task
            or that every task has the same input format.When task_id is None,
            default, it performs all tasks sequentially.
        feedback_budget : int | float = None
            TODO implement a feedback budget that allows the predictor to
            request feedback for only so many samples, so the selection of
            which samples to request feedback for matters.
        """
        return self.fine_tune.predict(dataset)

    def extract_predict(self, dataset, task_id=None):
        return self.fine_tune.extract_predict(dataset)

    def known_probs(self, dataset):
        """Backwards compat for older agents in eval."""
        return self.predict(dataset)

    def novelty_detect(self, dataset, task_id=None):
        """Predictor performs novelty detection given the dataset, possibly
        conditioned on specific task set. Novelty detection is the same as
        anaomaly detection, outlier detection, out-of-distirbution detection,
        etc...

        Args
        ----
        dataset : torch.Dataset
        task_id : str = None
            The str identifier of the task to perform with the given inputs.
            This assumes the proper dataset input format is given for each task
            or that every task has the same input format.When task_id is None,
            default, it performs detection across all tasks and returns an
            novelty detection ansewr based on the data relative to all tasks.
        """
        return self.fine_tune.feature_extract(dataset)

    # TODO def feedback_query(self, dataset, task_id=None):


# TODO class OWHAPredictorEVM(OWHAPredictor):
class ANNEVM(OWHAPredictor):
    """Combo of OWHAPredictor and EVMPredictor.

    Attributes
    ----------
    fine_tune: arn.models.fine_tune_lit.FineTuneLit
        fine_tune: arn.models.fine_tune.FineTune
    evm : EVMPredictor
    novelty_detector: WindowedMeanKLDiv = None
    feedback_interpreter: arn.models.feedback.CLIPFeedbackInterpreter = None
    label_enc : NominalDataEncoder = None
    _uid : str = None
        An optional str unique identifier for this predictor. When not
        given, the uid property of this class' object is the trainer
        tensorboard log_dir version number.
    """
    def __init__(
        self,
        fine_tune,
        evm,
        novelty_detector=None,
        feedback_interpreter=None,
        dtype=None,
        label_enc=None,
        uid=None,
    ):
        """Initializes the OWHAR.

        Args
        ----
        fine_tune: see self
        evm: see self
        novelty_detector: see self
        feedback_interpreter: see self
        label_enc : str | NominalDataEncoder = None
            Filepath to be loaded or the actual NominalDataEncoder.
        uid : str = None
            An optional str unique identifier for this predictor. When not
            given, the uid property of this class' object is the trainer
            tensorboard log_dir version number. If 'datetime', saves datetime
            of init.
        """
        self.fine_tune = fine_tune
        self.novelty_detector = novelty_detector
        self.feedback_interpreter = feedback_interpreter
        self._increment = 0
        self._uid = uid if uid != 'datetime'  \
            else f"owhap-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"
        self.dtype = torch_dtype(dtype)
        # TODO Use the dtype in all interior predictor parts unless None.
        if isinstance(label_enc, str):
            self.label_enc = NominalDataEncoder.load(label_enc)
        else:
            self.label_enc = label_enc

        logger.info('Predictor UID `%s` init finished.', self.uid)
