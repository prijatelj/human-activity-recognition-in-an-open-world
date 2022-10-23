"""Utilities for loading results files to be analyzed."""
from functools import partial
import glob
import os
import re

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.express.colors import qualitative as qual_colors
from plotly.validators.scatter.marker import SymbolValidator
import yaml

from exputils.data import OrderedConfusionMatrices
from exputils.data.labels import NominalDataEncoder as NDE
from docstr.cli.cli import docstr_cap

import logging
logger = logging.getLogger(__name__)


def load_ocm_step(
    path,
    regex,
    re_groups,
    steps=None,
    reduce_known=False,
    reduce_unknown=True,
    known_label_encs=None,
    finetune_skip_fit='skip_fit-1',
):
    """Loads the OrderedConfusionMatrices and if steps given, returns the
    ConfusionMatrix with any desired reductions.

    Args
    ----
    path: str
        The path to the OrderedConfusionMatrices to load.
    regex: re.Pattern
        The regex pattern used to obtain the step number and if pre- or post-
        feedback from the filepath.
    re_groups: dict
        The expected named patterns being matched to extract int(step_num) and
        bool(pre-feedback). A dict of key pattern group name to the typer
        casting to use.
    steps: list =None
        A list of DataSplits containing the train KineticsUnified datasets.
    reduce_known: bool = False
        If True, reduces all knowns based on known label enc to 'known'.
    reduce_unknown: bool = True
        If True, reduces all unknowns based on known label enc to 'unknown'.
    known_label_encs: list = None
        A list of predictors' known label encoders that corresponds to the
        steps. When provided, will use these label encoders as the known label
        enc for calculating any reductions on the confusion matrix.

        This is recommended over steps and finetune_skip_fit.
    finetune_skip_fit: str = 'skip_fit-1*fine-tune'
        The str pattern to use to check if the path contains a 0% feedback ANN
        that thus needs to use the first step's train label enc.


    Returns
    -------
    Confusion
    """
    ocm = OrderedConfusionMatrices.load(path)
    matched = regex.match(path)
    for re_group, dtype in re_groups.items():
        setattr(ocm, re_group, dtype(matched.group(re_group)))

    # Collapse unknowns if steps
    # NOTE: OCM.reduce DNE, so discard ocm, get cm
    step_num = ocm.step_num
    pre_feedback = ocm.pre_feedback

    if known_label_encs:
        known_label_enc = known_label_encs[step_num]
    elif steps:
        if finetune_skip_fit:
            regex_0feedback = re.compile(finetune_skip_fit)
            match = regex_0feedback.findall(path)
            known_label_enc = steps[0].train.label_enc
        else:
            match = False

        if not match:
            if step_num > 0 and pre_feedback:
                # Assumes the known label enc is prior step's train label enc
                known_label_enc = steps[step_num - 1].train.label_enc
            else:
                # Assumes the known label enc is current step's train label enc
                known_label_enc = steps[step_num].train.label_enc
    else:
        return ocm

    cm = ocm.get_conf_mat()

    # TODO finetune anns 0% feedback have slight increase in post-feedback
    # perf in NMI (arithmetic) while accuracy and MCC remain the same. What
    # is exact cause? The difference would be the train label enc used from
    # the dataset to inform what is known at this time step. for 0%
    # feedback, the very first step's train label enc should ALWAYS be
    # used.
    # TODO Furthermore, this means the GMM label enc used for knowns is
    # also wrong! This is ONLY good for 100% feedback knowns.
    unknowns = set(cm.label_enc) - set(known_label_enc)

    if cm.label_enc.unknown_key is None:
        if 'unknown' in cm.label_enc:
            cm.label_enc._unknown_key = 'unknown'
            if not cm.label_enc.are_keys_sorted:
                cm.label_enc._update_argsorted_keys(cm.label_enc.encoder)
        else:
            label_enc = NDE(['unknown'], unknown_key='unknown')
            label_enc.append(list(cm.label_enc))
            cm.label_enc = label_enc

    if unknowns:
        if 'unknown' not in unknowns:
            unknowns.add('unknown')

        unknowns = np.array(list(unknowns))
        logger.debug(
            '%d unknowns at step %d',
            len(unknowns), # 0 if not unknowns else len(unknowns),
            step_num,
        )

        # if collapse unknowns
        if reduce_unknown:
            cm.reduce(
                unknowns,
                'unknown', #cm.label_enc.unknown_key,
                reduced_idx=0, #cm.label_enc.unknown_idx,
                inplace=True,
            )
            if reduce_known:
                # Collapse knowns if steps and reduce_known
                cm.reduce(
                    ['unknown'],
                    'known', #cm.label_enc.unknown_key,
                    reduced_idx=-1, #cm.label_enc.unknown_idx,
                    inverse=True,
                    inplace=True,
                )
        elif reduce_known:
            # Collapse knowns, preserving unknowns.
            cm.reduce(
                unknowns,
                'known', #cm.label_enc.unknown_key,
                reduced_idx=-1, #cm.label_enc.unknown_idx,
                inverse=True,
                inplace=True,
            )

    cm.step_num = step_num
    cm.pre_feedback = pre_feedback
    return cm


def load_inplace_results_tree(
    tree : dict,
    root_dir='',
    data_split='test',
    filename='preds_top-cm.h5',
    leaf_is_dir=True,
    pred_dir_path='step-*_*_predict',
    steps=None,
    reduce_known=False,
    reduce_unknown=True,
):
    """Inplace update of paths at leaves. Load either the
    OrderedConfusionMatrices or pandas DataFrame given the paths of prediction
    files within the different step directories. Sort the step directories by
    ascending order of their step number.

    Args
    ----
    tree : dict
        The dictionary whose leaves are str filepaths.
    root_dir : str = ''
        The root_dir string to be prefixed to the leaf string paths.
    data_split : str = 'test'
        The data split dir to access within each step directory.
    filename : str = 'preds_top-cm.h5'
    leaf_is_dir : bool = True
    pred_dir_path : str = 'step-*_*_predict'
        If loading dirs, includes pre_feedback dirs to be loaded from.
    steps : list = None
        If given, assumed to be a list of KineticsUnified objects that
        correspond to the step number based on their index.
    reduce_known : bool = False
        If True, and steps given, reduces known classes into 'known'
    reduce_unknown : bool = True
        If True, and steps given, reduces unknown classes into 'unknown'
    """
    logger.debug('Begin `load_inplace_results_tree`')
    if reduce_known and not steps:
        raise ValueError(
            '`reduce_known` is True and steps is None! Cannot compute'
        )
    if reduce_unknown and not steps:
        raise ValueError(
            '`reduce_unknown` is True and steps is None! Cannot compute'
        )
    regex = re.compile(
        '.*step-(?P<step_num>\d+)_(?P<pre_feedback>.*)_predict.*'
    )

    get_ocm = os.path.splitext(filename)[-1] == '.h5'

    func = partial(
        load_ocm_step,
        regex=regex,
        re_groups={'step_num': int, 'pre_feedback': lambda x: x == 'new-data'},
        steps=steps,
        reduce_known=reduce_known,
        reduce_unknown=reduce_unknown,
    ) if get_ocm else pd.read_csv

    def regex_cast(x):
        step_num_regex, pre_feedback_regex = regex.match(x).groups()
        return int(step_num_regex), pre_feedback_regex

    # Depth first traversal
    stack = [(tree, k, v) for k, v in tree.items()]
    while stack:
        ptr, key, value = stack.pop()
        if isinstance(value, dict):
            for k, v in value.items():
                stack.append((value, k, v))
        else:
            base_path = os.path.join(root_dir, value)
            if leaf_is_dir:
                ptr[key] = [func(item) for item in sorted(
                    glob.glob(
                        os.path.join(
                            base_path,
                            pred_dir_path,
                            data_split,
                            filename,
                        )
                    ),
                    key=regex_cast,
                )]
            else:
                if get_ocm:
                    # Pretty sure you just call func anyways on the singl obj.
                    ptr[key] = func(base_path)
                else:
                    # Loads DataFrames, but takes argmax.
                    df = pd.read_csv(base_path)

                    # Regex get step num
                    matched = regex.match(base_path)
                    step_num = int(matched.group('step_num'))
                    pre_feedback = bool(matched.group('pre_feedback'))
                    if steps:
                        df['pred_labels'] = df.columns[
                            df.values[:, 6:].argmax(1)
                        ]
                        step_known = steps[step_num].train.label_enc

                        # Collapse unknowns in targets if steps:
                        #   decode(encode(.)) both targets and predictions
                        df['target_labels'] = step_known.decode(
                            step_known.encode(df['target_labels'])
                        )
                        # Not necessary on pred_labels given they can only be
                        # known labels.

                        if reduce_known:
                            # Collapse knowns if steps and reduce_known
                            df['pred_labels'][df['pred_labels'] != 'unknown'] \
                                = 'known'
                    df['step_num'] = step_num
                    df['pre_feedback'] = pre_feedback

                    ptr[key] = df
    logger.debug('Finished `load_inplace_results_tree`')


def get_ocm_measures(ocm, measures, prefix_col, col_names, known_labels=None):
    calc_measures = []
    if isinstance(ocm, OrderedConfusionMatrices):
        cm = ocm.get_conf_mat()
    else:
        cm = ocm

    for idx, m_attr in enumerate(measures.values()):
        if isinstance(m_attr, dict):
            key, val = next(iter(m_attr.items()))
            if isinstance(val, dict):
                if key == 'accuracy':
                    if not isinstance(ocm, OrderedConfusionMatrices):
                        #logger.warning(
                        raise TypeError(
                            'Measure request for top-k accuracy, but `ocm` is '
                            'not an OrderedConfusionMatrices. Skipping. '
                            f'type(ocm) : {type(ocm)}',
                        )
                        # This silently breaks fig generated idk why.
                        del col_names[idx]
                        continue
                    # top-k accuracy
                    measure = getattr(ocm, 'accuracy')(**val)
                else:
                    measure = getattr(cm, key)(**val)
            else: # probs never used
                measure = getattr(cm, key)(val)
        else:
            measure = getattr(cm, m_attr)()
        calc_measures.append(measure)
    return pd.Series(prefix_col + calc_measures, index=col_names)


def add_uid_col(df, cols=None, uid_col_name='uid'):
    """Inplace add of unique identifier column."""
    if cols is None:
        cols = ['Feature Repr.', 'Classifier']

    tmp = df[cols[0]]
    for col in cols[1:]:
        tmp += '+' + df[col]

    df[uid_col_name] = tmp


def get_step_measures_ocm(
    tree,
    data_split,
    measures,
    cumulative=True,
    kenv=None,
    post_feedback_step=0.5,
    uid_col_name='uid',
):
    dtypes = {
        'Feature Repr.': str,
        'Classifier': str,
        'Step': int,
        'Pre-feedback': bool,
        'Data Split': str,
    }
    df = pd.DataFrame(
        [],
        columns = list(dtypes) + list(measures.keys()),
    ).astype(dtypes)

    # Get measures from ocms and fill DataFrame
    for frepr, fr_dict in tree.items():
        for classifier, ocms in fr_dict.items():
            cum_ocm = None
            for ocm in ocms:
                if cumulative and not ocm.pre_feedback: # only cumulative post
                    if cum_ocm is not None:
                        cum_ocm += ocm
                    else:
                        cum_ocm = ocm
                else:
                    cum_ocm = ocm

                if kenv is not None:
                    if ocm.step_num == 0:
                        known_labels = kenv.start.train.label_enc
                    else:
                        known_labels = kenv.steps[ocm.step_num].train.label_enc
                else:
                    known_labels = None

                df = df.append(
                    get_ocm_measures(
                        cum_ocm if cumulative and not ocm.pre_feedback else ocm,
                        measures,
                        [
                            frepr,
                            classifier,
                            ocm.step_num,
                            ocm.pre_feedback,
                            data_split,
                        ],
                        col_names=list(df.columns),
                        known_labels=known_labels,
                    ),
                    ignore_index=True,
                )
    if post_feedback_step:
        df['Step'][~df['Pre-feedback']] += post_feedback_step
    if uid_col_name and uid_col_name not in df.columns:
        add_uid_col(df, uid_col_name=uid_col_name)
    return df


def get_balance_df(kowl_dsplits, eval_split='test'):
    """Returns a DataFrame with the number of known and novel classes and their
    samples per increment at pre- and post-feedback
    """
    balance_stats = []
    for i, dsplits in enumerate(kowl_dsplits):
        eval_dsplit = getattr(dsplits, eval_split)
        if eval_dsplit is None:
            continue
        if i > 0:
            # pre_feedback
            pre_novel = set(eval_dsplit.label_enc) \
                - set(kowl_dsplits[i-1].train.label_enc)
            balance_stats.append([
                i,
                True,
                len(kowl_dsplits[i-1].train.label_enc),
                eval_dsplit.data['labels'].isin(
                    kowl_dsplits[i-1].train.label_enc
                ).sum(),
                len(pre_novel),
                eval_dsplit.data['labels'].isin(pre_novel).sum(),
            ])

        post_novel = set(eval_dsplit.label_enc) - set(dsplits.train.label_enc)
        # post_feedback
        balance_stats.append([
            i,
            False,
            len(dsplits.train.label_enc),
            eval_dsplit.data['labels'].isin(dsplits.train.label_enc).sum(),
            len(post_novel),
            eval_dsplit.data['labels'].isin(post_novel).sum(),
        ])

    return pd.DataFrame(
        balance_stats,
        columns=[
            'Step',
            'Pre-feedback',
            'known classes',
            'known samples',
            'novel classes',
            'novel samples'
        ],
    )


def load_incremental_ocms_df(
    yaml_path,
    data_split=None,
    filename=None,
    reduce_known=None,
    reduce_unknown=None,
    pred_dir_path=None,
    kowl=None,
    cumulative=None,
    get_ocms=False,
):
    """Convenience function for loading the yaml config that defines the
    measure objects to load and calculate using above functions.

    Returns
    -------
    list
        The resulting contents of the measure objects loaded and calculated.
    """
    # Load in yaml config file
    with open(yaml_path) as openf:
        config = yaml.load(openf, Loader=yaml.CLoader)
    root_dir = config.pop('root_dir', '')
    if data_split is None:
        data_split = config.pop('data_split', 'test')
    if filename is None:
        filename = config.pop('filename', 'preds_top-cm.h5')

    if cumulative is None:
        cumulative = config.pop('cumulative', True)

    if pred_dir_path is None:
        # 'step-*_post-feedback_predict',
        pred_dir_path = config.pop('pred_dir_path', 'step-*_*_predict')

    if reduce_known is None:
        reduce_known = config.pop('reduce_known', False)
    if reduce_unknown is None:
        reduce_unknown = config.pop('reduce_unknown', True)


    # Give kowl=False to avoid loading this if in config.
    if kowl is None or kowl is True:
        # Load for getting knowns and unknowns at time steps
        kowl = config.pop('kowl', None)
        if kowl is not None:
            kowl = docstr_cap(kowl, return_prog=True).environment
            kowl = [kowl.start] + kowl.steps

            # Otherwise, causes errors as is
            config['measures'].pop('Top-5 Accuracy', None)

    load_inplace_results_tree(
        config['ocms'],
        root_dir,
        data_split,
        filename=filename,
        leaf_is_dir=True,
        steps=kowl,
        reduce_known=reduce_known,
        reduce_unknown=reduce_unknown,
    )

    # TODO When calculating novelty, need to look at pre-novlety.
    #   and post-novelty contains no information of importance for novelty
    #   detection, as is... may be able to code that up to do
    #   predictor.predict_recognize(), but the known labels get updated in 100%
    #   feedback of human labels.

    # Get DataFrame of measures over increments.
    if not get_ocms and os.path.splitext(filename)[-1] == '.h5':
        return get_step_measures_ocm(
            config['ocms'],
            data_split,
            config['measures'],
            cumulative=cumulative,
        )

    # Expects to be returning the tree containing lists of DataFrames:
    return config['ocms']
