"""Code for measuring the novelty reaction time (detection time) of the
predictors given their results on an experiment.
"""
from copy import deepcopy

import yaml

from arn.scripts.visuals.load_results import *
from arn.scripts.visuals.line_plots_inc import line_plot, qual_colors


def example_loading_dfs():
    """Just an example of loading the incremental steps' dataframes in order.
    The result a dict structured as the tree in the yaml file. You can change
    it how you want, but i did this so that info was present and may be used
    when comparing.

    You want to take the list of dataframes that are in order by steps and alt:
    post-feedback, new-data, ...

    Note the indices start at zero for every DataFrame.
    """
    dfs = load_incremental_ocms_df(
        'arn/scripts/exp2/visuals/line_plot.yaml',
        'test',
        'preds.csv',
        kowl=False,
    )
    return dfs


def load_kowl_inc_dsets_with_docstr(experiment_config_yaml_path):
    """Works, but ineficient load. Load for getting knowns and unknowns at time
    steps.
    """
    # Load in yaml config file
    with open(experiment_config_yaml_path) as openf:
        kowl = yaml.load(openf, Loader=yaml.CLoader)

    kowl_env = docstr_cap(kowl, return_prog=True).environment
    return [kowl_env.start] + kowl_env.steps


def calc_novelty_reaction(dfs, known_label_encs):
    """
    Args
    ----
    dfs : list(pd.DataFrame)
        A list of pandas DataFrames where each DataFrame corresponds to an
        increment in the experiment. Their indices of the samples indicate the
        order as time-steps and are assumed to be ascending order across all
        dataframes.
    known_label_encs : list(NominalDataEncoder)
        List of label encoders that define the known labels per step, where the
        length of this list matches the `len(dfs)`.

    Returns
    -------
    pd.DataFrame
          A DataFrame storing the recorded results in columns ['youtube_id',
          'time_start', 'time_end', 'target_label', 'index_occurs',
          'next_novelty_index', 'index_detection']. None if no novelty detected
          within the ['index_occurs', 'next_novelty_index') range. Note the
          inclusive start and exclusive end of that range, as typical in
          python.
    """
    raise NotImplementedError('TODO')

    return novelty_reaction_df
