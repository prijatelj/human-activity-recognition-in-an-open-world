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


def load_ocm_step(path, regex, re_groups, steps=None, binary=False):
    ocm = OrderedConfusionMatrices.load(path)
    matched = regex.match(path)
    for re_group, dtype in re_groups.items():
        setattr(ocm, re_group, dtype(matched.group(re_group)))

    if steps:
        # Collapse unknowns if steps
        # NOTE: OCM.reduce DNE, so discard ocm, get cm

        # TODO train.label_enc is only applicable for val and test of same
        # split post-feedback. Otherwise, needs to be the last steps' train
        # label_enc.
        if ocm.step_num > 0 and ocm.pre_feedback:
            step = steps[ocm.step_num - 1]
        else:
            step = steps[ocm.step_num]

        step_num = ocm.step_num
        pre_feedback = ocm.pre_feedback
        ocm = ocm.get_conf_mat()

        unknowns = set(ocm.label_enc) - set(step.train.label_enc)

        if ocm.label_enc.unknown_key is None:
            if 'unknown' in ocm.label_enc:
                ocm.label_enc._unknown_key = 'unknown'
                if not ocm.label_enc.are_keys_sorted:
                    ocm.label_enc._update_argsorted_keys(ocm.label_enc.encoder)
            else:
                label_enc = NDE(['unknown'], unknown_key='unknown')
                label_enc.append(list(ocm.label_enc))
                ocm.label_enc = label_enc

        if unknowns:
            unknowns = np.array(list(unknowns))
            logging.debug(
                '%d unknowns at step %d',
                len(unknowns), # 0 if not unknowns else len(unknowns),
                step_num,
            )

            ocm.reduce(
                unknowns,
                'unknown', #ocm.label_enc.unknown_key,
                reduced_idx=0, #ocm.label_enc.unknown_idx,
                inplace=True,
            )

            if binary:
                # Collapse knowns if steps and binary
                ocm.reduce(
                    ['unknown'],
                    'known', #ocm.label_enc.unknown_key,
                    reduced_idx=-1, #ocm.label_enc.unknown_idx,
                    inverse=True,
                    inplace=True,
                )
        ocm.step_num = step_num
        ocm.pre_feedback = pre_feedback
    return ocm


def load_inplace_results_tree(
    tree : dict,
    root_dir='',
    data_split='test',
    filename='preds_top-cm.h5',
    leaf_is_dir=True,
    pred_dir_path='step-*_*_predict',
    steps=None,
    binary=False,
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
    binary : bool = False
        If True, and steps given, reduces known and unknown classes.
    """
    logger.debug('Begin `load_inplace_results_tree`')
    if binary and not steps:
        raise ValueError('`binary` is True and steps is None! Cannot compute')
    regex = re.compile(
        '.*step-(?P<step_num>\d+)_(?P<pre_feedback>.*)_predict.*'
    )

    get_ocm = os.path.splitext(filename)[-1] == '.h5'

    # TODO construct load_ocm_step partial such that it reduces for unks, and
    # binary

    func = partial(
        load_ocm_step,
        regex=regex,
        re_groups={'step_num': int, 'pre_feedback': lambda x: x == 'new-data'},
        steps=steps,
        binary=binary,
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
                    ocm = OrderedConfusionMatrices.load(base_path)
                    if steps:
                        # Collapse unknowns if steps
                        # NOTE: OCM.reduce DNE, so discard ocm, get cm
                        ocm = ocm.get_conf_mat()
                        step = steps[ocm.step_num]
                        unknowns = np.array(
                            set(ocm.label_enc) - set(step.train.label_enc)
                        )
                        logging.debug(
                            '%d unknowns at step %d',
                            len(unknowns),
                            ocm.step_num,
                        )

                        ocm.reduce(
                            unknowns,
                            'unknown', #ocm.label_enc.unknown_key,
                            reduced_idx=0, #ocm.label_enc.unknown_idx,
                            inplace=True,
                        )

                        if binary:
                            # Collapse knowns if steps and binary
                            ocm.reduce(
                                ['unknown'],
                                'known', #ocm.label_enc.unknown_key,
                                reduced_idx=-1, #ocm.label_enc.unknown_idx,
                                inverse=True,
                                inplace=True,
                            )
                    ptr[key] = ocm
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

                        if binary:
                            # Collapse knowns if steps and binary
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


def get_step_measures_ocm(
    tree,
    data_split,
    measures,
    cumulative=True,
    kenv=None,
    binary=False, # If True calculates
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
    return df


def load_incremental_ocms_df(
    yaml_path,
    data_split=None,
    filename=None,
    binary=None,
    pred_dir_path=None,
    kowl=None,
):
    # Load in yaml config file
    with open(yaml_path) as openf:
        config = yaml.load(openf, Loader=yaml.CLoader)
    root_dir = config.pop('root_dir', '')
    if data_split is None:
        data_split = config.pop('data_split', 'test')
    if filename is None:
        filename = config.pop('filename', 'preds_top-cm.h5')
    cumulative = config.pop('cumulative', True)

    if pred_dir_path is None:
        # 'step-*_post-feedback_predict',
        binary = config.pop('pred_dir_path', 'step-*_*_predict')

    if binary is None:
        binary = config.pop('binary', False)


    # Give kowl=False to avoid loading this if in config.
    if kowl in {None, True}:
        # Load for getting knowns and unknowns at time steps
        kowl = config.pop('kowl', None)
        if kowl is not None:
            kowl = docstr_cap(kowl, return_prog=True).environment
            kowl = [kowl.start] + kowl.steps

            # Causes errors otherwise
            config['measures'].pop('Top-5 Accuracy', None)

    load_inplace_results_tree(
        config['ocms'],
        root_dir,
        data_split,
        filename=filename,
        leaf_is_dir=True,
        steps=kowl,
    )

    # TODO When calculating novelty, need to look at pre-novlety.
    #   and post-novelty contains no information of importance for novelty
    #   detection, as is... may be able to code that up to do
    #   predictor.predict_recognize(), but the known labels get updated in 100%
    #   feedback of human labels.

    # Get DataFrame of measures over increments.
    if os.path.splitext(filename)[-1] == '.h5':
        # TODO If post-feedback, increase post-feedback step by 0.5
        return get_step_measures_ocm(
            config['ocms'],
            data_split,
            config['measures'],
            cumulative=cumulative,
        )

    # Expects to be returning the tree containing lists of DataFrames:
    return config['ocms']
