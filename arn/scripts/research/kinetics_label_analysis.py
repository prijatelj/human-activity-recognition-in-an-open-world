"""The helper code to analyze the Kinetics label CSVs"""
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd


class OrderedCounter(Counter, OrderedDict):
    pass


def get_unique_youtube_ids(df):
    """Obtains the unique youtube_ids in the given DataFrame and also finds the
    indices in the dataframe where those unique youtube_ids occur that have a
    count greater than 1 to help find samples from the same youtube video.
    """
    unique_ids = np.unique(
        df.youtube_id,
        return_counts=True,
        return_inverse=True,
    )
    return (
        unique_ids,
        np.logical_or.reduce([
            unique_ids[1] == i for i in
            np.arange(len(unique_ids[0]))[unique_ids[2] > 1]
        ]),
    )


def overlaps_where(start, end, rel_start, rel_end, inclusive=True):
    """Given time segment overlaps where relative to another time segment?"""
    assert start < end
    assert rel_start < rel_end

    if inclusive:
        start_within = start >= rel_start and start <= rel_end
        end_within = end >= rel_start and end <= rel_end
    else:
        start_within = start > rel_start and start < rel_end
        end_within = end > rel_start and end < rel_end

    if start_within and end_within:
        return 'within'
    if not start_within and end_within:
        return 'start'
    if start_within and not end_within:
        return 'end'
    return None


def overlapping_samples(df, inclusive=True):
    """Returns the youtube_ids whose durations overlap and their amount of
    overlap in seconds.

    Args
    ----
    df : pandas.DataFrame
        The dataframe whose repeated youtube ids' times are checked for overlap.
    inclusive : bool = True
        If True, then the equality used is inclusive of same value in seconds
        marker, meaning that those videos that overlap just the beginnings and
        end by 1 second are included. Otherwise, excludes these samples.
        When this is True, the same second overlaps will have a duration of 0.

    Returns
    -------
    OrderedDict({str: OrderedCounter({int: int})})
        An OrderedDict of youtube_id to a counter of the row number in the
        original DataFrame to the length of the duration in seconds. Includes a
        counter of its youtube_id to its first occurrence row index in original
        DataFrame.
    """
    overlapping_ids = OrderedDict()
    repeat = {}
    for i, row in enumerate(df.iloc):
        if row['youtube_id'] not in repeat:
            repeat[row['youtube_id']] = [
                {'idx': i, 'start': row['time_start'], 'end': row['time_end']},
            ]
            continue

        # The row's youtube_id is a repeat. For each prior id, check overlap
        for prior in repeat[row['youtube_id']]:
            overlap_state = overlaps_where(
                row['time_start'],
                row['time_end'],
                prior['start'],
                prior['end'],
                inclusive,
            )

            if overlap_state is None:
                continue
            elif overlap_state == 'within':
                duration = row['time_end'] - row['time_start']
            elif overlap_state == 'start':
                duration = row['time_end'] - prior['start']
            elif overlap_state == 'end':
                duration = prior['end'] - row['time_start']

            if row['youtube_id'] in overlapping_ids:
                overlapping_ids[row['youtube_id']][i] = duration
            else:
                overlapping_ids[row['youtube_id']] = OrderedCounter({
                    row['youtube_id']: prior['idx'],
                    i: duration
                })

        repeat[row['youtube_id']].append({
            'idx': i,
            'start': row['time_start'],
            'end': row['time_end'],
        })

    return overlapping_ids

"""
len(bad_idx)
bad_idx_euqals = bad_idx

bad_idx = []
redund = {}
for i in k7train.iloc[hmm].iloc:
    if i['youtube_id'] not in redund:
        redund[i['youtube_id']] = [{'start': i['time_start'], 'end': i['time_end']}]
        continue
    for x in redund[i['youtube_id']]:
        if (i['time_start'] > x['start'] and i['time_start'] < x['end']) \
            or (i['time_end'] > x['start'] and i['time_end'] < x['end']):
            bad_idx.append(i['youtube_id'])
    redund[i['youtube_id']] = [{'start': i['time_start'], 'end': i['time_end']}]
len(bad_idx)
len(bad_idx_euquals)
"""

# zy4M--DmT6U : hugging baby and hugging (not baby) have 3 second overlap.
# 978 samples in bad_idx_equals, 974 unique youtube_ids with overlap.
# 927 samples in bad_idx, 923 unique youtube_ids with overlap.
#   51 unique ids with samples that overlap by a second.

# Determine which of these overlapping repeats are identical if any.
def get_overlaps_of_n_seconds(df, n=10):
    """Return a list of youtube_ids whose overlap duration matches n-seconds"""
    match = []
    for key, val in df.items():
        for vkey, vval in val.items():
            if isinstance(vkey, str): continue
            if vval == n: match.append(key)
    return match

def merge_kinetics_splits(dset_id, train_path, val_path, test_path):
    return (
        pd.read_csv(train_path)
        .append(pd.read_csv(val_path))
        .append(pd.read_csv(test_path))
        .rename(columns={
            'label': f'kinetics{dset_id}',
            'split': f'kinetics{dset_id}_split',
        })
    )


def align_kinetics_csvs(k400, k600, k700_2020):
    """Align the Kinetics CSVs such that repeat samples across datasets are
    tracked.
    """
    # Align all data frames
    k4all = merge_kinetics_splits('400', k400.train, k400.val, k400.test)
    k6all = merge_kinetics_splits('600', k600.train, k600.val, k600.test)
    k7all = merge_kinetics_splits(
        '700_2020',
        k700_2020.train,
        k700_2020.val,
        k700_2020.test,
    )

    kall = pd.merge(
        k4all,
        k6all,
        on=['youtube_id', 'time_start', 'time_end'],
        how='outer',
    )
