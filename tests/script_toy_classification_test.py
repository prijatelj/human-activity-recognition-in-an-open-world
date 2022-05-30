"""Quick interpreter/docstr test run of FineTuneLit."""
import pandas as pd
import plotly.express as px
import torch

from exputils.data import ConfusionMatrix
from exputils.data.labels import NominalDataEncoder

from arn.data.kinetics_unified import get_kinetics_uni_dataloader
from arn.scripts.sim_open_world_recog.sim_gen import SimClassifyGaussians

import logging
logger = logging.getLogger(__name__)


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


def evaluate(inc_idx, split, features, labels, predictor, running_cm=None):
    preds = predictor.predict(get_kinetics_uni_dataloader(
        features,
        collate_fn=lambda batch: batch[0][0].unsqueeze(0)
    ))
    cm = ConfusionMatrix(
        #labels.tolist(),
        labels.argmax(-1).tolist(),
        preds.argmax(-1).squeeze().tolist(),
    )

    logger.info(
        'Increment %d: %s eval: NMI = %.4f',
        inc_idx,
        split,
        cm.mutual_information('arithmetic'),
    )
    logger.info('Increment %d: %s eval: MCC = %.4f', inc_idx, split, cm.mcc())
    logger.info(
        'Increment %d: %s eval: Accuracy = %.4f',
        inc_idx,
        split,
        cm.accuracy(),
    )

    if running_cm is None:
        return cm

    running_cm = running_cm + cm

    logger.info(
        'Increment %d: Running: %s eval: NMI = %.4f',
        inc_idx,
        split,
        running_cm.mutual_information('arithmetic'),
    )
    logger.info('Increment %d: Running: %s eval: MCC = %.4f',
        inc_idx,
        split,
        running_cm.mcc(),
    )
    logger.info(
        'Increment %d: Running: %s eval: Accuracy = %.4f',
        inc_idx,
        split,
        running_cm.accuracy(),
    )

    return running_cm


def increment(
    inc_idx,
    predictor,
    sim,
    inc_train_num_each,
    inc_test_num_each,
    train_features=None,
    train_labels=None,
    running_train_cm=None,
    running_test_cm=None,
):
    logger.info('Starting increment #: %d', inc_idx)

    logger.info('Increment %d: Generate train samples', inc_idx)
    # Generate the train samples
    inc_train_features, inc_train_labels = sim.eq_sample_n(
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

    logger.info('Increment %d: Fit train samples', inc_idx)
    # Incremental fit by keeping prior points
    predictor.fit([train_features, train_labels])

    running_train_cm = evaluate(
        inc_idx,
        'train',
        inc_train_features,
        inc_train_labels,
        predictor,
        running_train_cm,
    )

    logger.info('Increment %d: Generate test samples', inc_idx)
    # Generate the incremental test samples
    inc_test_features, inc_test_labels = sim.eq_sample_n(inc_test_num_each)
    inc_test_labels = torch.nn.functional.one_hot(inc_test_labels.to(int)).to(float)

    running_test_cm = evaluate(
        inc_idx,
        'test',
        inc_test_features,
        inc_test_labels,
        predictor,
        running_test_cm,
    )

    return (
        train_features,
        train_labels,
        #test_features,
        #test_labels,
        running_train_cm,
        running_test_cm,
    )



def run(
    predictor,
    sim=None,
    visualize=False,
    inc_train_num_each=100,
    inc_test_num_each=100,
    total_increments=1,
    log_level='INFO',
):
    """The setup for all tests in this class.

    Args
    ----
    predictor : arn.models.fine_tune_lit.FineTuneLit
    sim : SimClassifyGaussians = None
        TODO docstr support:  allow required arg when its config args all have
        defaults.
    visualize : bool = False
    inc_train_num_each : int = 100
    inc_test_num_each : int = 100
    total_increments : int = 1
        The number of increments to perform. Always performs one pass, which is
        the initial start of incremental learning.
    log_level : str = 'INFO'
    """
    train_features = None
    train_labels = None
    #test_features = None
    #test_labels = None

    running_train_cm = None
    running_test_cm = None

    logging.basicConfig(
        #filename='../results/toy_test.log',
        #filemode='w',
        level=getattr(logger, log_level, None),
        format='%(asctime)s; %(levelname)s: %(message)s',
        datefmt=None,
    )

    # Simulated incremental steps of environment/experiment
    for i in range(total_increments):
        train_features, train_labels, running_train_cm, running_test_cm = \
        increment(
            i,
            predictor,
            sim,
            inc_train_num_each,
            inc_test_num_each,
            train_features,
            train_labels,
            running_train_cm,
            running_test_cm,
        )
