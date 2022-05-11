"""Quick interpreter/docstr test run of FineTuneLit."""
import logging

import pandas as pd
import plotly.express as px
import torch

from exputils.data import ConfusionMatrix
from exputils.data.labels import NominalDataEncoder

from arn.data.kinetics_unified import get_kinetics_uni_dataloader


class ToyClassify2D4MVNs(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to. The Gaussian distributions are labeled their index which
    starts at zero at the top most Gaussian centered at [0, 1] and labels the
    rest that follow clockwise around the unit circle.

    Attributes
    ----------
    locs : list = None
        TODO docstr support: list(float) | list(list(float)) = None
        Defaults to 4 gaussian locations = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    scales : float = 0.2
        TODO docstr support: float | list(float) = 0.2
        The scales of the gaussians in the mixture.
    labels : list = None
    seed : int = 0
    """
    def __init__(self, locs=None, scales=0.2, labels=None, seed=0):
        """
        Args
        ----
        see self
        """
        if seed is not None:
            # Set seed: Seems I cannot carry an individual RNG easily...
            torch.manual_seed(seed)

        # TODO create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        if not isinstance(scales, list):
            scales = [scales] * len(locs)

        self.mvns = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.Tensor(loc),
                torch.eye(2) * scales[i],
            )
            for i, loc in enumerate(locs)
        ]

        if labels:
            self.label_enc = NominalDataEncoder(labels)
        else:
            self.label_enc = None

    def eq_sample_n(self, num, randperm=True):
        if randperm:
            idx = torch.randperm(num * len(self.mvns))
            return (
                torch.cat([mvn.sample_n(num) for mvn in self.mvns])[idx],
                torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten()[idx],
            )
        return (
            torch.cat([mvn.sample_n(num) for mvn in self.mvns]),
            torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten(),
        )


def visualize_space(features, labels, preds):
    # To visualize the results:
    df = pd.DataFrame(features.tolist(), columns=['x','y'])
    df['label'] = labels.tolist()
    df['pred'] = preds.argmax(-1).tolist()

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='label',
        symbol='label',
    )

    fig.show()

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='pred',
        symbol='pred',
    )

    return fig


def evaluate(inc_idx, split, features, labels, predictor):
    preds = predictor.predict(get_kinetics_uni_dataloader(
        features,
        collate_fn=lambda batch: batch[0][0].unsqueeze(0)
    ))
    cm = ConfusionMatrix(
        #labels.tolist(),
        labels.argmax(-1).tolist(),
        preds.argmax(-1).squeeze().tolist(),
    )

    logging.info(
        'Increment %d: %s eval: NMI = %.4f',
        inc_idx,
        split,
        cm.mutual_information('arithmetic'),
    )
    logging.info('Increment %d: %s eval: MCC = %.4f', inc_idx, split, cm.mcc())
    logging.info(
        'Increment %d: %s eval: Accuracy = %.4f',
        inc_idx,
        split,
        cm.accuracy(),
    )


def increment(
    inc_idx,
    predictor,
    toy_sim,
    inc_train_num_each,
    inc_test_num_each,
    train_features=None,
    train_labels=None,
    test_features=None,
    test_labels=None,
):
    logging.info('Starting increment #: %d', inc_idx)

    logging.info('Increment %d: Generate train samples', inc_idx)
    # Generate the train samples
    inc_train_features, inc_train_labels = toy_sim.eq_sample_n(
        inc_train_num_each
    )
    inc_train_labels = torch.nn.functional.one_hot(inc_train_labels.to(int)).to(float)

    # Append the new train samples to the old samples
    if train_features is not None and train_labels is not None:
        train_features = torch.cat([train_features, inc_train_features])
        train_labels = torch.cat([train_labels, inc_train_labels])
    else:
        train_features = inc_train_features
        train_labels = inc_train_labels

    logging.info('Increment %d: Fit train samples', inc_idx)
    # Incremental fit by keeping prior points
    predictor.fit([train_features, train_labels])

    evaluate(inc_idx, 'train', train_features, train_labels, predictor)

    logging.info('Increment %d: Generate test samples', inc_idx)
    # Generate the incremental test samples
    inc_test_features, inc_test_labels = toy_sim.eq_sample_n(
        inc_test_num_each,
    )
    inc_test_labels = torch.nn.functional.one_hot(inc_test_labels.to(int)).to(float)

    # Append the new test samples to the old samples
    if test_features is not None and test_labels is not None:
        test_features = torch.cat([test_features, inc_test_features])
        test_labels = torch.cat([test_labels, inc_test_labels]).to('int')
    else:
        test_features = inc_test_features
        test_labels = inc_test_labels

    evaluate(inc_idx, 'test', test_features, test_labels, predictor)

    # TODO record eval per inc, and show running train and test cm evals.

    return train_features, train_labels, test_features, test_labels #, run_cm


def run(
    predictor,
    toy_sim=None,
    visualize=False,
    train_num_each=30,
    test_num_each=100,
    inc_train_num_each=100,
    inc_test_num_each=100,
    total_increments=1,
):
    """The setup for all tests in this class.

    Args
    ----
    predictor : arn.models.fine_tune_lit.FineTuneLit
    toy_sim : ToyClassify2D4MVNs = None
        TODO docstr support:  allow required arg when its config args all have
        defaults.
    visualize : bool = False
    train_num_each : int = 32
    test_num_each : int = 100
    inc_train_num_each : int = 100
    inc_test_num_each : int = 100
    total_increments : int = 1
        The number of increments to perform. Always performs one pass, which is
        the initial start of incremental learning.
    """
    train_features = None
    train_labels = None
    test_features = None
    test_labels = None
    # TODO running_cm = None

    # Simulated incremental steps of environment/experiment
    for i in range(total_increments):
        train_features, train_labels, test_features, test_labels \
        = increment(
            i,
            predictor,
            toy_sim,
            inc_train_num_each,
            inc_test_num_each,
            train_features,
            train_labels,
            test_features,
            test_labels,
        )
