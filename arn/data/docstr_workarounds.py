"""The cli use of this project uses the prototype verison of docstr. As such,
some workarounds are necessary to make it work as desired.
"""
from collections import OrderedDict

from arn.data.kinetics_owl import KineticsOWL

import logging
logger = logging.getLogger(__name__)


def get_steps(**kwargs):
    """Hotfix for docstr to load in 2 KineticsUnified datasets. TODO update
    docstr to parse and generate CAPs for types within lits when specified.

    Note that the number of expected source DataSplits objects is based on the
    following args section.

    Note that kwargs is not sorted by order given in docstr config! Uses
    lexical sort of the Arg names.

    Args
    ----
    step_1 : arn.data.kinetics_owl.DataSplits
        The first DataSplit of KineticsUnified Datasets
    step_2 : arn.data.kinetics_owl.DataSplits
        The second DataSplit of KineticsUnified Datasets

    Returns
    -------
    list
        List of step 1 and step 2
    """
    kwargs = OrderedDict((key, kwargs[key]) for key in sorted(kwargs))
    steps = []
    for i, (key, step) in enumerate(kwargs.items()):
        logger.debug(
            'Loading step source DataSplit "%s" as %d-th step',
            key,
            i,
        )
        for split in ['train', 'validate', 'test']:
            step_split = getattr(step, split)
            if (
                step_split is not None
                and step_split.subset is not None
                and step_split.subset.labels is not None
                and step_split.subset.labels.name is None
            ):
                logger.warning(
                    "Removing step %dth's source %s split %s because "
                    'subset.labels.name is None',
                    i,
                    key,
                    split,
                )
                setattr(step, split, None)
        steps.append(step)
    return steps


def get_vtransforms(**kwargs):
    """Hotfix for docstr same as above get_steps, but for the visually
    transformed versions of the entire Kinetics Unified data.

    Args
    ----
    color_invert : arn.data.kinetics_owl.DataSplits
    color_jitter : arn.data.kinetics_owl.DataSplits
    gaussian_blur : arn.data.kinetics_owl.DataSplits
    gaussian_noise : arn.data.kinetics_owl.DataSplits
    rotation : arn.data.kinetics_owl.DataSplits
    vertical_flip : arn.data.kinetics_owl.DataSplits

    Returns
    -------
    list
        List of the DataSplits as ordered in Args.
    """
    return get_steps(**kwargs)


# NOTE the following is all a workaround for the current docstr prototype to
# support the ease of swapping predictors by changing the config only, not the
# doc strings of KineticsOWL. This is what happens when reseach code meets
# prototype code.
def kinetics_owl_evm(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL
    predictor : arn.models.predictor.EVMPredictor
    feedback_type : see KineticsOWL
    feedback_amount : see KineticsOWL
    rng_state : see KineticsOWL
    eval_on_start : see KineticsOWL
    eval_config : see KineticsOWL
    post_feedback_eval_config : see KineticsOWL
    tasks : see KineticsOWL
    maintain_experience : bool = True
        If False, the default, the past experienced samples are not saved
        in the simulation for use by the predictor. Otherwise, the
        experienced samples are saved by concatenating the new data splits
        to the end of the prior ones.
    labels : str = None
    """
    return KineticsOWL(*args, **kwargs)


def kinetics_owl_annevm(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL
    predictor : arn.models.predictor.ANNEVM
    feedback_type : see KineticsOWL
    feedback_amount : see KineticsOWL
    rng_state : see KineticsOWL
    eval_on_start : see KineticsOWL
    eval_config : see KineticsOWL
    post_feedback_eval_config : see KineticsOWL
    tasks : see KineticsOWL
    maintain_experience : bool = True
        If False, the default, the past experienced samples are not saved
        in the simulation for use by the predictor. Otherwise, the
        experienced samples are saved by concatenating the new data splits
        to the end of the prior ones.
    labels : str = None
    """
    return KineticsOWL(*args, **kwargs)


def kinetics_owl_naive_dpgmm(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL
    predictor : arn.models.novelty_recog.naive_dpgmm.NaiveDPGMM
    feedback_type : see KineticsOWL
    feedback_amount : see KineticsOWL
    rng_state : see KineticsOWL
    eval_on_start : see KineticsOWL
    eval_config : see KineticsOWL
    post_feedback_eval_config : see KineticsOWL
    tasks : see KineticsOWL
    maintain_experience : bool = True
        If False, the default, the past experienced samples are not saved
        in the simulation for use by the predictor. Otherwise, the
        experienced samples are saved by concatenating the new data splits
        to the end of the prior ones.
    labels : str = None
    """
    return KineticsOWL(*args, **kwargs)


def kinetics_owl_gauss_finch(*args, **kwargs):
    """Initialize the KineticsOWL experiment.

    Args
    ----
    environment : see KineticsOWL
    predictor : arn.models.novelty_recog.gauss_finch.GaussFINCH
    feedback_type : see KineticsOWL
    feedback_amount : see KineticsOWL
    rng_state : see KineticsOWL
    eval_on_start : see KineticsOWL
    eval_config : see KineticsOWL
    post_feedback_eval_config : see KineticsOWL
    tasks : see KineticsOWL
    maintain_experience : bool = True
        If False, the default, the past experienced samples are not saved
        in the simulation for use by the predictor. Otherwise, the
        experienced samples are saved by concatenating the new data splits
        to the end of the prior ones.
    labels : str = None
    """
    return KineticsOWL(*args, **kwargs)
