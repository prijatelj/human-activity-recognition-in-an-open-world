"""The helper code to analyze the Kinetics label CSVs"""
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd


class OrderedCounter(Counter, OrderedDict):
    pass

def get_filename(
    df,
    uid='youtube_id',
    start='time_start',
    end='time_end',
    ext='.mp4',
    zfill=6,
):
    """Given the DataFrame, return the Kinetics filename, expecting `.mp4`.
    Args
    ----
    df : pd.DataFrame
        DataFrame with the columns `id`, `start`, and `end`.
    uid : str | object
        The DataFrame column name of the youtube id for Kinetics.
    start : str | object
        The DataFrame column name of the start time in seconds as an integer
    end : str | object
        The DataFrame column name of the end time in seconds as an integer
    ext : str = '.mp4'
        The str filename extention to use. If None, then no extention is added.
    zfill : int = 0
        The number of zeros to pad for each start and end time in the filename.

    Returns
    -------
    pd.Series
        Series of string filenames for each row in the given DataFrame `df`.
    """
    if ext is None:
        # No extention
        return df.apply(
            lambda x: f'{x[uid]}_{x[start]:0{zfill}}_{x[end]:0{zfill}}',
            axis=1,
        )
    return df.apply(
        lambda x: f'{x[uid]}_{x[start]:0{zfill}}_{x[end]:0{zfill}}{ext}',
        axis=1,
    )


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
    )


def align_kinetics_csvs(k400, k600, k700_2020):
    """Align the Kinetics CSVs such that repeat samples across datasets are
    tracked.

    Returns
    -------
    pd.DataFrame
        The DataFrame that consists of the Kinetics datasets merged by
        youtube_id, time_start, and time_end columns. This is the dataframe
        typically expected as input to the following functions as it represents
        all changes between the different Kinetics datasets.
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
    # TODO be aware of time step possibly changing across Kinetics Datasets!
    return k4all.merge(
        k6all,
        on=['youtube_id', 'time_start', 'time_end'],
        how='outer',
        suffixes=('_kinetics400', '_kinetics600'),
    ).merge(
        k7all.rename(columns={
            'label': 'label_kinetics700_2020',
            'split': 'split_kinetics700_2020',
        }),
        on=['youtube_id', 'time_start', 'time_end'],
        how='outer',
    )


def check_all_uniques(
    df,
    columns=[
        'label_kinetics400',
        'label_kinetics600',
        'label_kinetics700_2020',
    ],
    nan=np.NaN,
    nan_replace=None,
    *args,
    **kwargs,
):
    """Convenience function to get uniques across multiple columns."""
    return np.unique(
        df[columns].replace(nan, nan_replace).values,
        *args,
        **kargs,
    )

def disjoint_samples(
    df,
    columns=[
        'label_kinetics400',
        'label_kinetics600',
        'label_kinetics700_2020',
    ],
):
    """Returns the disjoint samples whose values are not in any other column.
    Used to get the disjoint unique labels in each Kinetics dataset.
    """
    # Get Boolean mask of elements that are NaN for each column
    isna = [pd.isna(df[c]).values for c in columns]
    # Use that to determine when a sample belongs to only one of the columns
    # When applied to label columns, this ignores that it may be a repeated
    # youtube_id that occurs in other datasets but with different time start
    # and end.
    return [
        df[np.logical_and.reduce(isna[:i] + [isna[i] ^ 1] + isna[i + 1:])]
        for i in range(len(columns))
    ]

# TODO Get all samples that originate from K400 and K600
# TODO Get all samples whose labels are different across Kinetics datasets
# TODO Get all samples whose splits are different across Kinetics datasets

def get_unique_pairs(df, col_1, col_2):
    """Get the unique label pairs of col_1 to col_2 with counts without NAs.
    Good for checking samples with label changes or when samples shifted to
    different splits in later versions.
    """
    no_nas = df[(pd.isna(df[col_1]) ^ 1) & (pd.isna(df[col_2]) ^ 1)]
    diff_labels = no_nas[no_nas[col_1] != no_nas[col_2]]
    return Counter(
        [(row[0], row[1]) for row in diff_labels[[col_1, col_2]].values]
    )

def get_label_series_pref_later_kinetics(df, label_csv):
    """Returns a pd.Series of the labels per sample using the combined Kinetics
    csv pd.DataFrame. This unified labeling prioritizes more recent Kinetics
    datasets, so 700_2020, then 600, then 400. The default mapping used is thus
    Kinetics 700. The samples who values have changed between Kinetics Datasets
    prefer to use the appropriate Kinetics700 labeling for that sample. If the
    sample was not in Kinetics 700, then its most recent label (Kinetics 600,
    then Kinetics 400) is used to convert to the approriate Kinetics700 label.
    This process provides the unified Kinetics labeling based on preference for
    the most recent label.

    Args
    ----
    df : pd.DataFrame
        The pandas dataframe of the combined Kinetics datasets merged such that
        youtube_id, time_start, and time_end serve as unique samples (rows).
    label_csv : str
        Path to the label csv that contains the mapping of Kinetics 400 to 600
        to 700_2020. This unified labeling is the macro default labeling to aid
        this function in ensuring that all Kinetics 400 and 600 older labels
        are updated to the most recent Kinetics700 label.

    Note
    ----
        This label scheme ignores any structural of hierarchial information
        between the labels and also ignores the possibility of soft labels
        based on a sample's different labelings across the Kinetics dataset
        releases. This process puts faith into the original designers of
        Kinetics 700_2020.
    """

    return


# TODO Unify the labeling scheme if there are labels that are just
# typos/characters off. Otherwise, make it a 1, 2, or 3 hot encoding,
# optionally weighting the dataset's labels by priority.
def get_k7_priority_labeling(df):
    """Get the pd.Series with df's index to add as a column for labels
    prioritizing K700, then K600 -> K700, then K400 -> K700. Marking those w/o
    any mapping to a K700 class to be handled manually. Where
    f'TODO_{src_class}' is the placeholder label.
    """
    # TODO if label_k700_2020, then use it.

    # TODO else if class was mapped to k7, use most frequent map
        # TODO opt. save the uid to a counter of keys being which k700 class to values of occurrences of that mapping.
        # TODO if no mapping for any class' samples to a k7 class, then handle manually.

    return


# TODO visualize label's samples across datasets w/ histogram of each dataset
# compared to each other.


# TODO Kinetics Torch Dataset(s) /  Dataloader(s) / Manager

# TODO For labels, use the latest version of the label, keeping parents and
# children as necessary. If able to determine a hierarchy from those that
# changed, then do so.

# TODO Get all Kinetics400 train, val, & test
# TODO Get subset of Kinetics600 train, val, & test, ignoring that which was
# taken already
# TODO Get subset of Kinetics700_2020 train, val, & test, ignoring that which
# was taken already

# For youtube ids and which dset and split they belong, order by the following
# to determine the precedence of when the samples will first occur:
#   Dataset order by earliest first: K400 -> K600 -> K700
#   Split order: train -> val -> test

# For the labels of samples order precedence by following:
#   Newer datasets take priority: K700 -> K600 -> K400
#   This means that if a label changed to from K400 to K700, then use the K700
#   label. This esp. applies to when labels get "corrected", such as typo fixes
#   or lowercasing the label text.
# If a hierarchy of action labels is apparent, then use it for factor analysis.
