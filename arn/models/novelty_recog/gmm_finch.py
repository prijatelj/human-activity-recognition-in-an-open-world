"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""
from copy import deepcopy

import h5py
import numpy as np
import torch
F = torch.nn.functional

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.models.novelty_recog.recognizer import h5_io
from arn.models.novelty_recog.gaussian import min_max_threshold
from arn.models.novelty_recog.gmm import (
    GMM,
    GMMRecognizer,
    recognize_fit,
)

import logging
logger = logging.getLogger(__name__)


class GMMFINCH(GMMRecognizer):
    """Gaussian Mixture Model per class using FINCH to find the components.

    Attributes
    ----------
    known_gmms : arn.models.novelty_recog.gmm.GMM = None
        List of Gaussian Mixture Model objects per class, where classes consist
        of knowns.
    thresholds : float | torch.Tensor = None
        Property.
        Global threshold for all distribs involved. If None, use the internal
        distribs thresholding for detection.
    see GMMRecognizer.__init__
    """
    def __init__(self, *args, **kwargs):
        """
        Args
        ----
        see GMMRecognizer.__init__
        """
        # NOTE uses self.known_label_enc and self._label_enc
        super().__init__(*args, **kwargs)
        self.known_gmms = None
        #self.thresholds = None

    def reset_recogs(self):
        """Resets the recognized unknown class-clusters, and label_enc"""
        super().reset_recogs()

        # Reset general recognizer to use just knowns
        self._label_enc = deepcopy(self.known_label_enc)

        logger.info('%s.reset_recogs() finished', type(self))

    def fit_knowns(self, features, labels, val_dataset=None):
        if self.load_inc_paths and self.increment in self.load_inc_paths:
            skip_fit = self.skip_fit
            self.load_state(self.load_inc_paths[self.increment])
            self.skip_fit = skip_fit

            print("@" * 20)
            print("gmm_finch.py line 67 in fit_knowns")
            print("self.label_enc: ", self.label_enc)
            print("self._known_label_enc: ", self._known_label_enc)
            print("self._recog_label_enc: ", self._recog_label_enc)
            print("self.known_label_enc", self.known_label_enc, type(self.known_label_enc))
            print("self.recog_label_enc", self.recog_label_enc, type(self.recog_label_enc))

            # Never refits given loaded state
            return

        # For each known class with labels, fit the GMM.
        knowns = iter(self.known_label_enc.items())
        next(knowns)

        if self.known_gmms is None:
            n_old_knowns = 0
            self.known_gmms = []
        else:
            n_old_knowns = len(self.known_gmms)

        for known, idx in knowns:
            class_mask = labels == idx
            if not class_mask.any():
                raise ValueError(
                    'Every known class must have at least one feature sample.'
                )
            prior_idx = idx - 1
            if prior_idx < n_old_knowns:
                self.known_gmms[prior_idx] = recognize_fit(
                    known,
                    features[class_mask],
                    counter=self.known_gmms[prior_idx].counter,
                    level=self.level,
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                    threshold_func=self.threshold_func,
                    min_samples=self.min_samples,
                    accepted_error=self.min_error_tol,
                    detect_likelihood=self.detect_likelihood,
                    batch_size=self.batch_size,
                )
            else:
                self.known_gmms.append(recognize_fit(
                    known,
                    features[class_mask],
                    counter=0,
                    level=self.level,
                    cov_epsilon=self.cov_epsilon,
                    device=self.device,
                    dtype=self.dtype,
                    threshold_func=self.threshold_func,
                    min_samples=self.min_samples,
                    accepted_error=self.min_error_tol,
                    detect_likelihood=self.detect_likelihood,
                    batch_size=self.batch_size,
                ))

        logger.info(
            "%s's %s.fit_knowns() finished fitting the GMMs per class",
            self.uid,
            type(self).__name__,
        )

        # If threshold_func is min_max_threshold, then it is global to the
        # known gmms and thus needs set.
        if (
            self.threshold_global
            and self.threshold_func == 'min_max_threshold'
        ):
            self._thresholds = min_max_threshold(
                self.known_gmms,
                features,
                self.detect_likelihood,
                self.batch_size,
            )
            # TODO For the initial increment only, assign the detect_likelihood
            # of this instance to the minimum maximum log_prob of the detected
            # unknowns in the validation dataset during prediction. Either that
            # or set the detect_likelihood to the difference between that
            # minimum maximum log_prob. All the data in validation are supposed
            # to be used to tune the model, further all the validation data are
            # known classes, so any detected unknowns is wrong and thus the
            # threshold found from the above should be adjusted using the
            # validation data.
            #   TODO do so programmatically. Also do not allow any initial step
            #   validation data to be treated in experience as unknown
            #   predictions. (so assign the predictions from recog w/o detect).
            #   TODO Could enable the use of min_error_tol to find the
            #   appropriate quantile of all falsely detected unknowns from the
            #   vlaidaiton data, this becomes a kind of prior mixed with risk
            #   assessment. Probably will just do without this.

    def recognize_fit(self, features, n_expected_classes=None, **kwargs):
        if not self.known_gmms:
            raise ValueError('Recognizer is not fit: self.known_gmms is None.')

        super().recognize_fit(features, n_expected_classes, **kwargs)

        # Update the general knowns + unknown recogs expanded
        self.update_label_enc(False)

    def recognize(
        self,
        features,
        detect=False,
        known_only=False,
        softmax=True,
    ):
        # TODO consider known_only arg.
        # For consistency w/ detect, may have to set it to True, and call it
        # with False in pipeline for current behavior.

        # Loop through all known gmms + unknown_recogs getting log_probs.
        recogs = torch.stack(
            [gmm.log_prob(features) for gmm in self.known_gmms],
            dim=1,
        )

        if self.has_recogs and not known_only:
            unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
            recogs = torch.cat([recogs, unknown_log_probs], dim=1)

        if detect:
            if not softmax:
                raise ValueError('Cannot perform detect w/o softmax!')
            if self.threshold_global:
                detect_unknowns = (recogs < self.thresholds).all(1)
            else:
                detect_unknowns = torch.stack(
                    [gmm.detect(features) for gmm in self.known_gmms],
                    dim=1,
                )
                if self.has_recogs and not known_only:
                    detect_unknowns = torch.cat(
                        (
                            detect_unknowns,
                            unknown_log_probs < self.unknown_gmm.thresholds,
                        ),
                        dim=1,
                    )
                detect_unknowns = detect_unknowns.all(1)

            recogs = F.pad(F.softmax(recogs, dim=1), (1, 0), 'constant', 0)

            # NOTE may want to change this to not set unknown to max prob.
            # Sets unknown to max prob value, scales the rest by 1 - max
            if detect_unknowns.any():
                recogs[detect_unknowns, 0] = \
                    recogs[detect_unknowns].max(1).values
                recogs[detect_unknowns, 1:] *= \
                    1 - recogs[detect_unknowns, 0].reshape(-1, 1)
            return recogs
        if not softmax:
            return recogs
        return F.softmax(recogs, dim=1)

    def detect(self, features, known_only=True):
        if self.threshold_global:
            recogs = self.recognize(
                features,
                known_only=known_only,
                softmax=False,
            )

            if self.has_recogs and not known_only:
                unknown_log_probs = self.unknown_gmm.comp_log_prob(features)
                recogs = torch.cat([recogs, unknown_log_probs], dim=1)

            detects = (recogs < self.thresholds).all(1)

            if detects.any():
                logger.debug('thresholds = %s', self.thresholds)
                logger.debug('detect_likelihood = %s', self.detect_likelihood)
                logger.debug('total detected = %s', detects.sum())
                logger.debug(
                    'detected mean log_prob = %s', recogs[detects].mean()
                )
                quantiles = torch.linspace(0, 1, 11).to(
                    recogs.device, recogs.dtype
                )
                logger.debug('quantiles for detection: %s', quantiles)
                #logger.debug(
                #    'detected quantiles log_prob = %s',
                #    torch.quantile(recogs[detects], quantiles)
                #)
                logger.debug(
                    'detected quantiles of max(1) log_prob = %s',
                    torch.quantile(recogs[detects].max(1).values, quantiles)
                )
                logger.debug(
                    'detected quantile min error tol (%f) of max(1) log_prob '
                    '= %s',
                    self.min_error_tol,
                    torch.quantile(
                        recogs[detects].max(1).values,
                        self.min_error_tol,
                    )
                )
            return detects

        detects = [gmm.detect(features) for gmm in self.known_gmms]

        if self.has_recogs and not known_only:
            detects.append(self.unknown_gmm.detect(features))

        return torch.stack(detects, dim=1).all(1)

    def log_prob(self, features, known_only=True):
        log_probs = [gmm.log_prob(features) for gmm in self.known_gmms]

        if self.has_recogs and not known_only:
            log_probs.append(self.unknown_gmm.log_prob(features))

        return torch.stack(log_probs, dim=1).sum(1)

    @h5_io('w')
    def save(self, h5):
        """Save known_gmms, but NOT gmm, as it is joined by the 2."""
        knowns = h5.create_group('known_gmms')
        for gmm in self.known_gmms:
            gmm.save(knowns.create_group(gmm.label_enc.unknown_key))

        super().save(h5)

    @h5_io
    @staticmethod
    def load(h5, class_type=None):
        # Call parent load() to load the init object w/ changes wrt ancestors.
        if class_type is None:
            loaded = GMMRecognizer.load(h5, GMMFINCH)
        else:
            loaded = GMMRecognizer.load(h5, class_type)

        loaded.known_gmms = [
            GMM.load(h5['known_gmms'][key]) for key in h5['known_gmms'].keys()
        ]

        if loaded.known_label_enc is None and '_known_label_enc' not in h5:
            # Old versions didn't save _known_label_enc. Able to recover:
            loaded.add_new_knowns([k for k in h5['known_gmms'].keys()])

        return loaded

    def load_state(self, h5, return_tmp=False, **kwargs):
        print('GMMFINCH type(self) = ', type(self))
        tmp = super().load_state(h5, True, **kwargs)

        print("@" * 20)
        print("gmm_finch.py line 311 in load_state")
        print("self.label_enc", self.label_enc, type(self.label_enc))
        print("self.known_label_enc", self.known_label_enc, type(self.known_label_enc))
        print("self.recog_label_enc", self.recog_label_enc, type(self.recog_label_enc))

        print("self._label_enc", self._label_enc, type(self._label_enc))
        print("self._known_label_enc", self._known_label_enc, type(self._known_label_enc))
        print("self._recog_label_enc", self._recog_label_enc, type(self._recog_label_enc))

        print("tmp.label_enc", tmp.label_enc, type(tmp.label_enc))
        print("tmp.known_label_enc", tmp.known_label_enc, type(tmp.known_label_enc))
        print("tmp.recog_label_enc", tmp.recog_label_enc, type(tmp.recog_label_enc))

        print("tmp._label_enc", tmp._label_enc, type(tmp._label_enc))
        print("tmp._known_label_enc", tmp._known_label_enc, type(tmp._known_label_enc))
        print("tmp._recog_label_enc", tmp._recog_label_enc, type(tmp._recog_label_enc))

        self.known_gmms = tmp.known_gmms

        if return_tmp:
            return tmp
