"""Code for creating increments of a KineticsUnified Dataset."""
from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from exputils.data.labels import NominalDataEncoder

import logging
logger = logging.getLogger(__name__)


def get_increments(
    n_increments,
    src_dataset,
    known_label_enc,
    seed=None,
    label_col='labels',
):
    """
    Args
    ----
    n_increments : int
        The number of increments to create
    src_dataset : KineticsUnified
        The source KineticsUnified Dataset for splitting into the incremental
        datasets.
    known_label_enc : exputils.data.labels.NominalDataEncoder
    seed : int = None
    label_col : str = 'labels'

    Returns
    -------
    list
        List of KineticsUnified Datasets that form the ordered increments.
    """
    if seed is None:
        np_gen = None
    else:
        np_gen = np.random.default_rng(seed)

    tmp_dataset = deepcopy(src_dataset)
    del src_dataset

    known_df = tmp_dataset.data
    tmp_dataset.data = None
    tmp_dataset.label_enc = None

    # Separate samples with known labels from those with unknown.
    known_mask = known_df[label_col].isin(known_label_enc)
    unknown_df = known_df[~known_mask]
    known_df = known_df[known_mask]
    label_enc = known_label_enc
    del known_label_enc

    logger.debug(
        'sum(known_mask) = %d; len(known_df) = %d',
        sum(known_mask),
        len(known_df),
    )
    logger.debug(
        'sum(~known_mask) = %d; len(unknown_df) = %d',
        sum(~known_mask),
        len(unknown_df),
    )
    logger.debug('len(known_label_enc) = %d', len(label_enc))

    # Stratified folds for known class samples, ensuring presence and balance
    # across incs
    skf = StratifiedKFold(
        n_increments,
        random_state=seed,
        shuffle=np_gen is not None,
    )
    known_incs = [
        known_df.iloc[test] for train, test in
        skf.split(known_df['sample_index'], known_df[label_col])
    ]

    # TODO randomization criteron that forces certain classes to be separate,
    # e.g. classes farther from each other in some class hierarchy should be
    # separated across the increments such that there is sufficiently novel
    # classes in each increment.

    # Randomize unique, unknown classes across n increments. Last w/ remainder
    unique, unique_inv = np.unique(unknown_df[label_col], return_inverse=True)
    n_unique = len(unique)
    logger.debug('unknowns: n_unique = %d', n_unique)

    if np_gen is not None:
        unique_perm = np.arange(n_unique)
        np_gen.shuffle(unique_perm)
        nde = NominalDataEncoder(unique_perm)
        unique = unique[unique_perm]
        unique_inv = nde.encode(unique_inv)

    classes_per_inc = np.ceil(n_unique / n_increments)
    unique_slices = np.cumsum(
        [0] + [classes_per_inc] * n_increments,
        dtype=int,
    )

    # Loop through and create the incremental KineticsUnified datasets
    increments = []
    for i, known_inc in enumerate(known_incs):
        inc_dataset = deepcopy(tmp_dataset)
        inc_uniques = unique[unique_slices[i]:unique_slices[i+1]]
        unks = unknown_df[
            unknown_df[label_col].isin(inc_uniques)
        ]
        logger.debug('increment %d: known samples = %d', i, len(known_inc))
        logger.debug('increment %d: unknown samples = %d', i, len(unks))
        inc_dataset.data = known_inc.append(unks)

        label_enc.append(unique[unique_slices[i]:unique_slices[i+1]])
        inc_dataset.label_enc = deepcopy(label_enc)

        increments.append(inc_dataset)

    return increments
