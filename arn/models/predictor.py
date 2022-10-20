"""Open World Human Activity Recognition predictor pipeline classes."""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import glob
import os
import re

import numpy as np
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

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


def get_chkpts_paths(dir_path, pattern):
    """Given the checkpoint directory and an opt. set of keys, dict of
    filepaths.
    dir_path : str
        The path which contains the checkpoint files.
    pattern : str = None
        The regex pattern to use to determine the checkpoint files and their
        increment number.
    """
    if not isinstance(pattern, re.Pattern):
        raise TypeError(
            f'Expected `pattern` of type `re.Pattern`, not `{type(pattern)}`'
        )
    chkpts = {}
    for fp in glob.iglob(os.path.join(dir_path, '*')):
        parsed = pattern.findall(fp)
        if parsed:
            chkpts[int(parsed[0])] = fp
    return OrderedDict((key, chkpts[key]) for key in sorted(chkpts))


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
        if (
            isinstance(dataset, tuple)
            and len(dataset) == 2
            and isinstance(dataset[0], torch.Tensor)
            and isinstance(dataset[1], torch.Tensor)
        ):
            return super().fit(dataset[0], dataset[1], *args, **kwargs)
        elif isinstance(dataset, KineticsUnifiedFeatures):
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
    fine_tune: arn.models.fine_tune_lit.FineTuneLit = None
        fine_tune: arn.models.fine_tune.FineTune
    _label_enc : NominalDataEncoder = None
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
    save_dir : str = None
        If given, the filepath to a directory to save all model checkpoints
        after fitting on an increment.
    load_inc_paths : str = None
        Dictionary of str filepaths which contain the state to be loaded via
        load_state() as the value and the key is the increment to which that
        loaded state was fit on.
    """
    def __init__(
        self,
        fine_tune=None,
        #dtype=None,# TODO perhaps have the predictor manage the dtypes overall
        label_enc=None,
        uid=None,
        skip_fit=-1,
        save_dir=None,
        start_increment=0,
        load_inc_paths=None,
        chkpt_file_prefix='version_[0-9]+',
    ):
        """Initializes the OWHAR.

        Args
        ----
        fine_tune: see self
        label_enc : str | NominalDataEncoder = None
            Filepath to be loaded or the actual NominalDataEncoder.
        uid : str = None
            An optional str unique identifier for this predictor. When not
            given, the uid property of this class' object is the trainer
            tensorboard log_dir version number. If 'datetime', saves datetime
            of init.
        skip_fit : see self
        save_dir : str = None
        start_increment : int = 0
        load_inc_paths : see self
        chkpt_file_prefix : str = 'version_[0-9]+'
            The file prefix to use to match to chkpt files in the chkpt
            director given by load_inc_paths, if given as a directory filepath.
        """
        self.fine_tune = fine_tune
        self._increment = int(start_increment)
        self.skip_fit = skip_fit
        self.save_dir = create_filepath(save_dir) if save_dir else None
        self._uid = uid if uid != 'datetime'  \
            else f"owhap-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"

        if isinstance(label_enc, str):
            self._label_enc = NominalDataEncoder.load(label_enc)
        else:
            self._label_enc = label_enc

        # TODO add predictor.experience, default None.

        if isinstance(load_inc_paths, str) and os.path.isdir(load_inc_paths):
            self.load_inc_paths = get_chkpts_paths(
                load_inc_paths,
                re.compile(
                    f'.*{chkpt_file_prefix}-(?P<increment>[0-9]+).*\..*'
                ),
            )
        else:
            self.load_inc_paths = load_inc_paths

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

    @property
    def label_enc(self):
        """Interface and forces to set label enc w/o assignment `=`."""
        return self._label_enc

    # TODO feedback request for ANN batching fun times
    #   If fitting a torch ANN w/ batching, will need to tmp rm all
    #   samples w/o a label before fitting and then restore after.
    def feedback_request(self, features, available_uids=None, amount=1.0):
        """The predictor's method of requesting feedback."""
        if available_uids is None:
            raise NotImplementedError('available_uids is necessary for ANNs.')
        idx = np.arange(len(available_uids))
        np.random.shuffle(idx)

        return available_uids[idx]

    def fit(self, dataset, val_dataset=None):
        """Incrementally fit the OWHAPredictor's parts. Update classes in
        classifier to match the training dataset. This assumes the training
        dataset contains all prior classes. This deep copy is convenient for
        ensuring the class indices are always aligned.
        """
        if self.load_inc_paths and self.increment + 1 in self.load_inc_paths:
            if self.skip_fit >= 0 and self._increment >= self.skip_fit:
                # NOTE Assumes if loading, you get 100% feedback from the label
                # enc for fine tune ANNs.
                self._label_enc = deepcopy(dataset.label_enc)

                n_classes = len(self.label_enc)
                if n_classes != self.fine_tune.n_classes:
                    self.fine_tune.set_n_classes(n_classes)

            skip_fit = self.skip_fit
            self.load_state(self.load_inc_paths[self.increment + 1])
            self.skip_fit = skip_fit

        if self.skip_fit >= 0 and self._increment >= self.skip_fit:
            # for saving checkpoints of state external to fine_tune.
            self._increment += 1
            return
        if isinstance(dataset, KineticsUnified):
            if (
                self.label_enc is None
                or set(dataset.label_enc) - set(self.label_enc)
            ):
                # Assumes dataset label enc is superset of label enc.
                self._label_enc = deepcopy(dataset.label_enc)

            n_classes = len(self.label_enc)
            if n_classes != self.fine_tune.n_classes:
                self.fine_tune.set_n_classes(n_classes)

        # NOTE moved this to end: self._increment += 1
        self.fine_tune.fit(dataset, val_dataset, label_enc=self.label_enc)

        if self.save_dir:
            self.fine_tune.trainer.save_checkpoint(os.path.join(
                self.save_dir,
                f'{self.uid}-{self.increment}.ckpt',
            ))
        self._increment += 1

    def predict(self, dataset):
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

    def extract(self, dataset):
        return self.fine_tune.extract(dataset)

    def extract_predict(self, dataset):
        return self.fine_tune.extract_predict(dataset)

    def known_probs(self, dataset):
        """Backwards compat for older agents in eval."""
        return self.predict(dataset)

    def novelty_detect(self, dataset):
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

    def load_state(self, ftune_chkpt):
        # NOTE this is specific to fine tune ANNs, not any Recognizer

        # TODO other attrs of OWHAPredictor, esp. _label_enc?
        #   As is w/o update to inherit OWHARecognizer, simply copy dset label
        #   enc if 100%. and only copy first label enc if 0%.

        if self.fine_tune is not None: # and self.increment < self.skip_fit:
            return
        try:
            self.fine_tune.model.load_from_checkpoint(
                ftune_chkpt,
                model=self.fine_tune.model.model,
            )
        except RuntimeError as e:
            logger.warning(
                'There was a class size mismatch in checkpoint to current '
                'model!'
            )
            e_msg = str(e)
            if 'model.classifier.weight' not in e_msg:
                raise e

            pat = re.compile('torch\.Size\(\[(?P<classes>[0-9]*)')
            classes = pat.findall(e_msg)
            if len(classes) < 1:
                raise e
            self.fine_tune.set_n_classes(int(classes[0]))
            self.fine_tune.model.load_from_checkpoint(
                ftune_chkpt,
                model=self.fine_tune.model.model,
            )



# TODO class OWHAPredictorEVM(OWHAPredictor):
class ANNEVM(object):
    """Simple combo of OWHAPredictor and EVMPredictor.

    Attributes
    ----------
    fine_tune: OWHAPredictor
        arn.models.fine_tune_lit.FineTuneLit
        fine_tune: arn.models.fine_tune.FineTune
    evm : EVMPredictor
    _uid : str = None
        An optional str unique identifier for this predictor. When not
        given, the uid property of this class' object is the trainer
        tensorboard log_dir version number.
    """
    def __init__(
        self,
        fine_tune,
        evm,
        uid=None,
    ):
        """Initializes the OWHAR.

        Args
        ----
        fine_tune: see self
        evm: see self
        novelty_detector: see self
        feedback_interpreter: see self
        uid : str = None
            An optional str unique identifier for this predictor. When not
            given, the uid property of this class' object is the trainer
            tensorboard log_dir version number. If 'datetime', saves datetime
            of init.
        """
        self.fine_tune = fine_tune
        self.evm = evm

        self._increment = 0

        if uid is None:
            self._uid = self.evm.uid.replace('evm', 'ann-evm')
        elif uid == 'datetime':
            self._uid = \
                f"ann-evm-{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.%f')}"
        else:
            self._uid = uid

        logger.info('Predictor UID `%s` init finished.', self.uid)

    @property
    def uid(self):
        """Returns a string Unique Identity for this predictor."""
        return self._uid

    @property
    def label_enc(self):
        return self.evm.label_enc

    def fit(self, dataset, val_dataset=None):
        self.fine_tune.fit(dataset, val_dataset, task_id)
        self.evm.fit(
            (
                self.fine_tune.extract(dataset),
                torch.stack([label for x, label in dataset]).argmax(1),
            ),
            # No passing of val_dataset for EVM until it uses it.
            #None if val_dataset is None else
            #    self.fine_tune.extract(
            #        val_dataset,
            #        torch.stack([label for x, label in dataset]),
            #),
        )

    def predict(self, features, unknown_last_dim=False):
        return self.evm.predict(self.fine_tune.extract(features))

    def extract_predict(self, dataset):
        extracts = self.fine_tune.extract(dataset)
        return extracts, self.evm.predict(extracts)
