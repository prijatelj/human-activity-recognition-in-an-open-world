"""The Generic and implemented novelty detectors."""
import copy
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


#from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

# TODO?
#class NoveltyDetector(object):
#    def detect(self, known_probs):
#        pass

class WindowedKLDivBayesFactor(object):
    """A Novelty Detector over time where samples are timesteps. This uses the
    predictor's output along with a distribution check of the KLdivergences
    from train to the pre-novelty to the train to the out of pre-train novelty.

    Attributes
    ----------
    window_size : int
        The number of samples to be included within the sliding window.
    detection_threshold : float = 0.5
        The chosen threshold for determining when a probability of novelty is
        large enough to be deemed a detection of novelty.
    distrib : torch.distributions.Distribution | tuple = None
        Distribution used to determine the KLDivergence between the train data
        and the window based on the parameters of that distribution found from
        the data. When given one torch.distributions.Distribution the same
        distirbution is used for train data, pre-novelty data, and test data,
        which assumes they are all of the same family of distributions. This
        Defaults to torch.distributions.Normal.

        Perhaps, if given 2 distirbution objects, could separate train distrib
        to window distrib (same for both pre-novelty and at eval time).

        Given this is at root for multiclass classificaiton, the distrib could
        be a Dirichelt. If only binary known vs unknown, then could be
        Binomial. Otherwise, when treated as regression w/ some threshold then
        is a linear regression, thus Gaussian makes sense, because of its
        relation to Mean Squared Error.
    sliding_window : list
        The list of samples that serves as the sliding window.
    has_world_changed : bool = False
        False if the world has not been deemed to have changed given prior
        observations, otherwise True.

    NOTE
    ----
    All this seems too hacky. Just do proper out-of-distribution detection with
    something like MCMC or a [C]VAE. Set the proper convergence criterion and
    let compute go as you can.
    """
    def __init__(
        self,
        window_size=200,
        detection_threshold=0.5,
        distrib=None
    ):
        self.window_size = window_size
        self.detection_threshold = detection_threshold
        self.has_world_changed = False
        self.sliding_window = []

        if distrib is None:
            self.distrib = torch.distributions.Normal
        self.in_sample_train = None
        self.in_sample_window = None
        self.out_sample_window = None

    def fit(self, train_pred, pre_novelty_pred, distrib=None):
        """Fit the train distrib.

        Args
        ----
        train_pred : torch.Tensor
        in_sample_windows : list
        """
        raise NotImplementedError()
        # TODO: fit the train data distrib, fit the in-sample window distrib
        self.in_sample_train = self.distrib()
        self.in_sample_window = self.distrib()

        self.in_sample_kl = torch.distributions.kl.kl_divergence(
            self.in_sample_train,
            self.in_sample_window
        )

        # This needs to fit the distrib of in-sample KL over the windows of
        # pre-novelty...

        # TODO should just fit some distrib to the train max-preds, the
        # in-sample windows' preds, and determine how out of distribution the
        # novel data is: Bayes Factor.

        # From these distribs determine the KLDiv of in sample window to train
        # (difference in in sample window to train).

        # From that, that establishes the minimum error/divergece to expect and
        # tolerate overtime in eval.

        # Then set a threshold based on validation data to determine how.

    def detect(
        self,
        known_probs,
        is_max_pred=False,
        distrib=None,
    ):
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

        Returns
        -------
        torch.Tensor
            A vector of "probability" values [0, 1] indicating if novelty is to
            be detected at the current sample given all prior samples and prior
            runs captured in this objects state information, e.g. the KL
            Divergence threshold after decay.
        """
        raise NotImplementedError()
        self.out_sample_window = self.distrib()
        self.out_sample_kl = torch.distributions.kl.kl_divergence(
            self.in_sample_train,
            self.in_sample_window
        )
        #return detections?


class WindowedMeanKLDiv(object):
    """A Novelty Detector consisting of the EVM and weighted KL Divergence.
    This method relies on a moving average using a sliding window.

    Attributes
    ----------
    detection_threshold : float = 0.5
        The chosen threshold for determining when a probability of novelty is
        large enough to be deemed a detection of novelty.
    has_world_changed : bool = False
        False if the world has not been deemed to have changed given prior
        observations, otherwise True.
    accuracy : float
    kl_threshold : float
    kl_threshold_decay : float
    mean_train : float
    std_dev_train : float
    window_size : int
        The number of samples to be included within the sliding window.
    sliding_window : list
        The list of samples that serves as the sliding window.
    past_window : list
        An extra window to use in computation. Why? idk, not my code (DSP).
        Could just double the sliding window size given the current
        implementation.

    Note
    ----
    This method has a built in assumption that novelty will eventually occur
    and that it is better to detect novelty later than earlier to avoid a false
    positive detection. That is why this uses a weighted decay of moving
    averages method.
    """
    def __init__(
        self,
        kl_threshold,
        kl_threshold_decay_rate,
        mean_train,
        std_dev_train,
        window_size,
        num_rounds,
        detection_threshold=0.5,
        #*args,
        #**kwargs,
    ):
        """
        Args
        ----
        kl_threshold : float
        kl_threshold_decay_rate : float
            determines KL Divergence threshold decay.
        mean_train : float
        std_dev_train : float
        window_size : int
        num_rounds : int
        detection_threshold : float = 0.5
        """
        #super(EVMWindowedMeanKLDiv, self).__init__(*args, **kwargs)

        # General Detector threshold
        self.detection_threshold = detection_threshold
        self.has_world_changed = False
        self.accuracy = 0.0

        # Kullback-Liebler Divergence attributes
        self.kl_threshold = kl_threshold
        self.kl_threshold_decay = kl_threshold_decay_rate / float(num_rounds)

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

    def detect(self, known_probs, is_max_pred=False):
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
        if not is_max_pred:
            known_probs = torch.max(known_probs, axis=1)

        round_size = len(known_probs)

        # Populate the sliding window of moving averages
        for x in range(len(self.sliding_window)):
            self.sliding_window[x] = self.sliding_window[x].detach()
        self.past_window = copy.deepcopy(self.sliding_window)
        self.sliding_window.extend(torch.mean(known_probs, axis=1))
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

            # Calculate Gaussian Kullback-Leibler Divergence for nominal data
            mean = torch.mean(p_window, dim=1)
            std_dev = torch.std(p_window, dim=1)
            kl_epoch = (
                torch.log(self.std_dev_train/std_dev)
                + (
                    ((std_dev ** 2) + ((mean - self.mean_train) ** 2))
                    / (2 * (self.std_dev_train ** 2))
                ) - 0.5
            )

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
            logger.info(
                f"self.temp_world_changed = {approx_world_changed}",
            )

            self.accuracy = temp_world_changed[-1]

        if self.accuracy > self.detection_threshold:
            self.has_world_changed = True

        self.kl_threshold = self.kl_threshold - self.kl_threshold_decay

        return temp_world_changed
