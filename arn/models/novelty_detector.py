"""The Generic and implemented novelty detectors."""
import copy
import logging

import numpy as np
import torch

from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

# TODO?
#class NoveltyDetector(object):
#    def detect(self, known_probs):
#        pass

class WindowedMeanKLDiv(ExtremeValueMachine):
    """A Novelty Detector consisting of the EVM and weighted KL Divergence.
    This method relies on a moving average using a sliding window.

    Attributes
    ----------
    detection_threshold : float
    has_world_changed : bool = False
    accuracy : float
    kl_threshold : float
    kl_threshold_decay : float
    mean_train : float
    std_dev_train : float
    window_size : int
    sliding_window : list()
    past_window : list()

    Note
    ----
    This method has a built in assumption that novelty will eventually occur
    and that it is better to detect novelty later than earlier to avoid a false
    positive detection. That is why this uses a weighted decay of moving
    averages method.
    """
    def __init__(
        self,
        detection_threshold,
        kl_threshold,
        kl_threshold_decay_rate,
        mean_train,
        std_dev_train,
        window_size,
        num_rounds,
        #*args,
        #**kwargs,
    ):
        """
        Args
        ----
        kl_threshold_decay_rate : determines KL Divergence threshold decay.
        """
        #super(EVMWindowedMeanKLDiv, self).__init__(*args, **kwargs)

        # General Detector threshold
        self.detection_threshold = detection_threshold
        self.has_world_changed = False
        self.accuracy = 0.0

        # Kullback-Liebler Divergence attributes
        self.kl_threshold = kl_threshold
        self.kl_threshold_decay = (num_rounds * kl_threshold_decay_rate) \
            / float(num_rounds)

        # Moving average attributes
        self.mean_train = mean_train
        self.std_dev_train = std_dev_train

        # Sliding window attributes
        self.window_size = window_size
        self.sliding_window = []

        # NOTE this past window is not needed to be stored in state by the
        # algorithm's use of it, but because it was saved in up-stream Kitware
        # code, we keep it.
        self.past_window = []

    def detect(self, known_probs, is_max_pred=False, logger=None):
        """Post probability score classification, perform detection using KL
        Divergence with a weighted threshold.

        Args
        ----
        known_probs : torch.Tensor
            Something confusing involving EVMWindowedMeanKLDiv.predict(input_samples).
            This is the probability vectors per sample of known classes.
        is_max_known_probs : bool = False
            If True, known_probs is the maximum probability per sample's predicted
            probability vector may be provided if precalculated.
        logger : logging
            A logging object to preserve the functionality up-stream.

        Returns
        -------
        torch.Tensor
            A vector of "probability" values [0, 1] indicating if novelty is to
            be detected at the current sample given all prior samples and prior
            runs captured in this objects state information, e.g. the KL
            Divergence threshold after decay.

        Note
        ----
        This is modified from the KL Divergence detection code worked on for
        the DARPA Sail-On project by the VAST group run by Terry E. Boult and
        by Kitware's programmers working on Sail-On, esp. Ameya Shringi.
        """
        # TODO this is notably bad design where it is essentially independent
        # of the EVM state, but for now it works. This is the issue when more
        # is done to handle novelty detection past the evm's prob vector
        # output.  The boundary between classification and novely detection
        # becomes blurred.
        if is_max_pred is None:
            known_probs = torch.max(known_probs, axis=1)

        round_size = len(known_probs)

        # Populate the sliding window of moving averages
        self.past_window = copy.deepcopy(self.sliding_window)
        self.sliding_window.extend(torch.mean(max_probs, axis=1))
        if len(self.sliding_window) >= self.window_size:
            window_size = len(self.sliding_window)
            self.sliding_window = \
                    self.sliding_window[window_size - self.window_size:]

        # Return zeros (no detection) if the sliding window length too small
        if len(self.sliding_window) < self.window_size:
            return torch.zeros(round_size)

        # Redundant case when accuracy is 1.0
        if self.accuracy == 1.0:
            return torch.ones(round_size)

        # Using KL divergence as written by VAST/Kitware
        with torch.no_grad():
            temp_world_changed = torch.zeros(round_size)

            p_past_and_current = torch.cat((
                torch.Tensor(self.past_window)[1:],
                torch.Tensor(self.sliding_window),
            ))
            p_window = p_past_and_current.unfold(0, self.window_size, 1)

            # Calculate Kullback-Leibler Divergence for nominal data
            mean = torch.mean(p_window, dim=1)
            std_dev = torch.std(p_window, dim=1)
            kl_epoch = (
                torch.log(self.std_dev_train/std_dev)
                + (
                    ((std_dev ** 2) + ((mean - self.mean_train) ** 2))
                    / (2 * (self.std_dev_train ** 2))
                ) - 0.5
            )

            if logger is not None:
                logger.info(f"max kl_epoch = {torch.max(kl_epoch)}")

            W = (kl_epoch / (2 * self.kl_threshold))#1.1

            logging.info(f"W = {W.tolist()}")

            W[0] = torch.clamp(W[0], min=self.accuracy)
            W = torch.cummax (W, dim=0)[0]

            temp_world_changed = \
                    torch.clamp(W , max=1.0)[len(W)-round_size:]
            temp_world_changed = torch.clamp(temp_world_changed, min=0)

            approx_world_changed = list(np.around(
                temp_world_changed.detach().numpy(),
                4,
            ))
            if logger is not None:
                self.logger.info(
                    f"self.temp_world_changed = {approx_world_changed}",
                )

            self.accuracy = temp_world_changed[-1]

        if self.accuracy > self.detection_threshold:
            self.has_world_changed = True

        self.kl_threshold = self.kl_threshold - self.kl_threshold_decay

        return temp_world_changed
