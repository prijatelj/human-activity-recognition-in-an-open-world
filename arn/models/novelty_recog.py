"""Novelty Recognition module of the baseline OWL HAR predictor."""
import hdbscan

import logging
logger = logging.getLogger(__name__)


class HDBSCANRecog(object):
    """Hierarchical DBSCAN clustering approach to novel class recognition.
    Recognition involves detecting novel/outlier samples to the distribution of
    classes within the feature representation space and then learning new
    clusters as new classes as enough samples are obtained.

    This detector is stateless beyond its initial hyperparameters and relies on
    being given the data samples as labeled and unlabeled.

    Note that a novel sample is synonymous to a sample that is an anomaly, an
    outlier, or out-of-distribution.

    Attributes
    ----------
    _novel_class_count : int = 0
        Internal state of the novel unknown classes recognized by this
        recognizer.  Used to maintain different class labeles of format
        'unknown_#'.
    """
    def __init__(self):
        """Initialize the recognizer.

        Args
        ----
        min_new_class_size : int
            Minimum number of samples for a cluster of outlier points to be
            considered a new class cluster.
        new_class_density : str = 'nearest_neighbor'
            String identifier of the critera to use for determining a new
            recognized unknown cluster's density param. Options are
            'nearest_neighbor' where it uses the same density as the nearest
            neighboring class cluster and 'minimum' which it uses the minimum
            density of any class cluster.

            Any cluster formed that is not within this density is not
            considered a new class cluster.
        """
        self.min_new_class_size = min_new_class_size
        self._novel_class_count = 0

        # TODO hmm.
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_dize=min_new_class_size,
            prediction_data=True,
        )

    @property
    def novel_class_count(self):
        return self._novel_class_count

    def predict(
        features,
        labels,
        class_preds=None,
        expected_classes=None,
    ):
        """Given data samples, detect novel samples. If given unlabeled data,
        decide if any of those samples form new clusters that consitute as a
        new class based on the initial hyperparameters.

        Recognized novel unknown classes are represented as 'unk_#' where # is
        the newest class number.

        Criteria for a new class cluster:
        - min number of samples
        - which density param is acceptable?
            min density of any class XOR density level of nearest class.

        Args
        ----
        features : arn.data.kinetics_unified.KineticsUnified
        labels : arn.data.kinetics_unified.KineticsUnified = None
        class_preds : = None
            If the labels have a prior predicted class from another predicotr,
            e.g., an ANN, then that may be used to inform the clustering???
        expected_classes : int = None
            The number of expected classes, thus including the total known
            classes + unknown class + any recognized unknown classes.

        Returns
        -------

            The predicted class label.
        """
        # TODO Obtain initial known class clusters
        #   Run HDBSCAN on features, saving all hierarchy info.
        #   Get the single linkage tree to serve as cluster hierarchy.

        # TODO per class, find optimal hierarchy density param given labels:
        #    Min hierarchy param, max class samples in class' cluster.

        # TODO Per class, determine label of sample
        #   if within class' density neighborhood, add to that class.
        #       aka: if clustered within same class' cluster.
        #   else an outlier:
        #       if not w/in definitions of a cluster, mark as unknown
        #       else mark that cluster as a new class.

        # New supervised labels given override unsupervised labels.

        return pred_classes

        # TODO outside of this, update pred label enc, and handle environment
        # label_enc != pred label_enc
