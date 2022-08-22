"""Novelty Recognition with HDBSCAN for hierarchical density class-clusters."""
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
        core_dist_n_jobs : int = 4
            The number of prallel jobs to run in core distance computations of
            HDBSCAN.
        """
        self.min_new_class_size = min_new_class_size
        self._novel_class_count = 0

        self.core_dist_n_jobs = core_dist_n_jobs

        # TODO hmm.
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_new_class_size,
            prediction_data=True,
        )

    @property
    def novel_class_count(self):
        return self._novel_class_count

    def predict(
        features,
        labels,
        label_enc,
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
            The sample feature representation vectors to be class clustered.
        labels : arn.data.kinetics_unified.KineticsUnified = None
            The labels for the features which includes elements as a label from
            label_enc including unknown, 'unlabeled' when there is no class
            assigned rather than being known as "unknown" or "other", or
            unknown_# as from this recognizer.

            unlabeled samples are to be predicted.

            All known labeled samples must be included within their class
            cluster.
        class_preds : = None
            If the labels have a prior predicted class from another predictor,
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
        clusterer = self.clusterer.fit(features)

        # TODO per class, find optimal hierarchy density param given labels:
        #    Min distance to cut, max class samples in one class' cluster.
        #    Includes keeping track of which cluster we care about here.
        #for label in label_enc:

        # TODO Per class, determine label of unlabeled sample
        #   if within class' density neighborhood, add to that class.
        #       aka: if clustered within same class' cluster.
        #   else an outlier:
        #       if not w/in definitions of a cluster, mark as unknown
        #       else mark that cluster as a new class.

        # New supervised labels given override unsupervised labels.

        return pred_classes
        # TODO should mark in dataframe which samples are labeled by recognizer
        # rather than oracle label: new col in experience dataframe.

        # TODO outside of this, update pred label enc, and handle environment
        # label_enc != pred label_enc

    def recognize(self, features, preds, label_enc, copy=True):
        raise NotImplementedError(
            'Placeholder: Copied over GaussRecog.recog when it used hdbscan.'
        )
        if copy:
            label_enc = deepcopy(label_enc)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_samples,
            gen_min_span_tree=True,
        )
        clusters = clusterer.fit(features)
        labels =  clusters.labels_
        bad_clusters = []
        for label in range(max(labels)+1):
            class_features = features[labels == torch.tensor(label)]
            if class_features.shape[0] < self.min_samples:
                # raise ValueError(
                #     f"Label {label} features' samples < min_samples."
                # )
                continue
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(
                    class_features.mean(0),
                    class_features.T.cov(),
                )

            if (
                mvn.log_prob(class_features).mean()
                < torch.log(torch.tensor(self.min_density))
            ):
                bad_clusters.append(labels)
                # raise ValueError(
                #     f"Label {label} features' normalized density "
                #     '< min_density.'
                # )
            #TODO: this might need to be changed, depending on the other parts of the loop
            self._gaussians.append(mvn)
            self._thresholds.append(mvn.log_prob(class_features).min())
        new_classes = 0
        remapper = {}
        remapper[-1] = -1
        for x in range(max(labels)+1):
            if x in bad_clusters:
                remapper[x] = -1
            else:
                remapper[x] = new_classes + self._novel_class_count
                new_classes += 1

        self._novel_class_count += new_classes
        new_classes = []
        still_unknown = []
        # for x in range(max(clusters.labels_) + 1):
        #     new_classes.append([])
        #
        #
        # for i,x in enumerate(clusters.labels_):
        #     if x == -1:
        #         still_unknown.append(i)
        #     else:
        #         new_classes[x].append(i)
        #
        #
        #
        # print("do it here")


        # Defaults recognition to unknown assuming only given unknowns
        recogs = torch.full(
            preds.shape,
            label_enc.unknown_idx,
            dtype=preds.dtype,
        )

        for x in range(len(recogs)):
            label = remapper[labels[x]]
            if label == -1:
                continue
            recogs[x] = label


        for i, x in enumerate(new_classes):
            for y in x:
                recogs[y] = i + len(label_enc)

        # TODO Perform clustering checking if any clusters that fit criteria:
        #   1. samples >= self.min_samples
        #   2. mvn.log_prob(samples).mean() >= torch.log(self.min_density)
        #   May have to use logic from DBSCAN or similar density approaches for
        #   optimal finding of clusters.

        # TODO once a new class cluster is recognized, add its identifier to
        # the label encoder and add the new label to the sample's corresponding
        # indices in recogs
        for x in remapper:
            index = remapper[x]
            if index == -1:
                continue
            new_recog = f'unknown_{index}'
            label_enc.append(new_recog)

        # TODO increment self._novel_class_count by the number of new class
        # clusters recognized = f'unknown_{self._novel_class_count}'
        # self._novel_class_count += 1

        return recogs, label_enc
