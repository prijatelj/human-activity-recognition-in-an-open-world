"""The cli use of this project uses the prototype verison of docstr. As such,
some workarounds are necessary to make it work as desired.
"""
def get_steps(step_1, step_2):
    """Hotfix for docstr to load in 2 KineticsUnified datasets. TODO update
    docstr to parse and generate CAPs for types within lits when specified.

    Args
    ----
    step_1 : DataSplits
        The first DataSplit of KineticsUnified Datasets
    step_2 : DataSplits
        The second DataSplit of KineticsUnified Datasets

    Returns
    -------
    list
        List of step 1 and step 2
    """
    steps = [step_1, step_2]
    for i, step in enumerate(steps):
        for split in ['train', 'validate', 'test']:
            if (
                step.validate is not None
                and step.validate.subset is not None
                and step.validate.subset.labels is not None
                and step.validate.subset.labels.name is None
            ):
                logger.warning(
                    "Removing step source %d's split %s because "
                    'subset.labels.name is None',
                    i,
                    split,
                )
                setattr(step, split, None)
    return steps


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
