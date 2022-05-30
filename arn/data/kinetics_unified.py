"""Dataset for the sample aligned Kinetics 400, 600, and 700_2020 datasets."""
from collections import namedtuple
from dataclasses import dataclass, InitVar
from functools import partial
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from arn.data.dataloader_utils import status_video_frame_loader
from arn.torch_utils import torch_dtype
from exputils.data.labels import NominalDataEncoder

import logging
logger = logging.getLogger(__name__)


def load_file_list(path):
    with open(path, 'r') as openf:
        contents = openf.read().splitlines()
    return contents


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


def get_kinetics_uni_dataloader(dataset, *args, **kwargs):
    """Get torch DataLoader of a KineticsUnified (subclass) dataset."""
    logger.debug('`type(dataset)` = %s', type(dataset))

    if isinstance(dataset, KineticsUnified):
        logger.debug('`len(dataset)` = %s', len(dataset))
        return torch.utils.data.DataLoader(dataset, *args, **kwargs)
    elif isinstance(dataset, torch.Tensor):
        logger.debug('`len(dataset)` = %s', len(dataset))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dataset),
            *args,
            **kwargs,
        )
    elif isinstance(dataset, (list, tuple)):
        logger.debug('dataset is list: `len(dataset)` = %s', len(dataset))
        for is_tensor in dataset:
            if not isinstance(is_tensor, torch.Tensor):
                raise TypeError(' '.join([
                    'If dataset a list or tuple, all elements must be',
                    f'torch.Tensor not `{type(is_tensor)}`',
                ]))
            logger.debug(
                'dataset is list: `len(is_tensor)` = %s',
                len(is_tensor),
            )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*dataset),
            *args,
            **kwargs,
        )
    elif not isinstance(dataset, torch.utils.data.DataLoader):
        raise TypeError(f'Unexpected dataset type: `{type(dataset)}`')

    logger.debug('`len(dataset)` = %s', len(dataset))
    return dataset


class KineticsRootDirs(object):
    """Stores the root directory paths for each Kinetics dataset.

    Attributes
    ----------
    kinetics400_dir : str = ''
    kinetics600_dir : str = ''
    kinetics700_2020_dir : str = ''
    root_dir : str = ''
    """
    def __init__(
        self,
        kinetics400_dir='',
        kinetics600_dir='',
        kinetics700_2020_dir='',
        root_dir='',
    ):
        """
        Args
        ----
        see self
        """
        if (
            root_dir == ''
            and (
                kinetics400_dir == ''
                or kinetics600_dir == ''
                or kinetics700_2020_dir == ''
            )
        ):
            raise ValueError(' '.join([
                '`root_dir` must be provided if any of kinetics dirs are',
                'empty strings ""',
            ]))
        self.kinetics400_dir = os.path.join(root_dir, kinetics400_dir)
        self.kinetics600_dir = os.path.join(root_dir, kinetics600_dir)
        self.kinetics700_2020_dir = os.path.join(
            root_dir,
            kinetics700_2020_dir,
        )

    def __len__(self):
        """Hotpatch length for indexing."""
        return 3

    def __getitem__(self, index):
        """Hotpatch getitem for indexing."""
        return np.array([
            self.kinetics400_dir,
            self.kinetics600_dir,
            self.kinetics700_2020_dir,
        ])[index]


def get_path(
    df,
    root_dirs,
    order=None,
    youtube_id='youtube_id',
    start='time_start',
    end='time_end',
    ext='.mp4',
    zfill=6,
    k700_suffix_label=True,
    split_prefix=None,
    #split_suffix=[None, None, ''],
):
    """Create filepath for every video, preference based on orderd datasets

    Args
    ----
    df : pandas.DataFrame
        The Kinetics Unified DataFrame.
    root_dirs : KineticsRootDirs | iterable | list | tuple
        Must have same length as order. If KineticsRootDirs, always ordered
        same and always 3, so have to mod it before putting in here if you want
        diferent order or less than 3.
    order : list(str)
        Determines the order of which columns are prioritized for getting
        video paths. Default is to prefer earlier Kinetics datasets, based
        on release date.
    split_prefix : list(str) = None
        The prefix to add to the beginning of the Kinetics split portion of
        the data directory.
    split_suffix : list(str)
        NOT Implemented. An attempt at generalizaiton postponed.
        The suffix to add to the end of the Kinetics split portion of the
        data directory.
    k700_suffix_label : bool = True
        When True, add the label name in lower as a directory suffix in the
        path. Otherwise do not add any suffix directories.

    Returns
    -------
    pd.Series
        A Pandas Series of the filepaths to each sample's video where the
        earlier Kinetics datasets' videos are prioritized.

    Note
    ----
    This is definitely hardcoded towards the specific Kinetics Unified csv
    where its columns are expected to be a certain way for this to
    function, e.g. the split columns all are expected to follow the
    pattern: 'split_kinetics[dset_num]'.
    """
    if order is None:
        order = [
        'split_kinetics400',
        'split_kinetics600',
        'split_kinetics700_2020',
        ]

    if split_prefix is None:
        split_prefix = ['kinetics-dataset-400-', None, None]
    elif split_prefix is False:
        split_prefix = [None, None, None]
    elif not isinstance(split_prefix, (list, tuple)):
        raise TypeError(
            'Expected split_prefix type of list of str, or None, or False, '
            f'not {type(split_prefix)}'
        )

    # Save when each sample is present in each dataset
    not_null = 1 ^ pd.isnull(df[order])

    # TODO include support for replacing videos that were corrupted in
    # earlier versions with those that are available and working in later
    # Kinetics versions.

    # TODO optionally expand samples by their pre-augmented versions.

    # Create the filepaths in order of preference of source dataset.
    df_order = []
    for i, col in enumerate(order):
        if i == 0: # Save a Kinetics sample if not null, and start init mask_or
            mask_or = not_null[col].copy()
            mask = mask_or
        elif i == 1:
            # mask_or is already set at idx 1 from idx 0. No need to update yet
            mask = (1 ^ (mask_or & not_null[col])) & not_null[col]
        else:
            # Save a Kinetics sample if not in other Kinetics (mask_or)
            mask_or |= not_null[order[i-1]]
            # AND(NAND(other_kinetics, not_null), not_null)
            mask = (1 ^ (mask_or & not_null[col])) & not_null[col]

        # Check which dataset this is and treat its path accordingly.. hardcode
        if '400' in col:
            if split_prefix[0]:
                k4_prefix = os.path.sep + split_prefix[0]
            else:
                k4_prefix = os.path.sep
            df_order.append(
                root_dirs.kinetics400_dir
                + k4_prefix
                + df[col].replace('validate', 'val')[mask],
            )
        elif '600' in col:
            if split_prefix[1]:
                k6_prefix = os.path.sep + split_prefix[1]
            else:
                k6_prefix = os.path.sep
            df_order.append(
                root_dirs.kinetics600_dir
                + k6_prefix
                + df[col][mask]
            )
        elif '700_2020' in col:
            # TODO support Kinetics700_2020 test video paths. They exist, but
            # no label directory and no known labels, resulting in NaN paths.
            # Currently no project plans to use Kinetics700_2020 testing that
            # are not already in the other Kinetics, so a future todo.
            # NOTE this may result in issues if 700_2020 is priority or the
            # others' videos are missing / corrupted, but this video does exist
            if split_prefix[2]:
                k7_prefix = os.path.sep + split_prefix[2]
            else:
                k7_prefix = os.path.sep

            #other_splits = df[col][(df[col] != 'test') & mask]
            #test_split = df[col][(df[col] == 'test') & mask]
            if k700_suffix_label:
                df_order.append(
                    root_dirs.kinetics700_2020_dir
                    + os.path.sep
                    + df[col][mask]
                    + k7_prefix
                    + df[col.replace('split', 'label')][mask]
                )
            else:
                df_order.append(
                    root_dirs.kinetics700_2020_dir
                    + k7_prefix
                    + df[col][mask]
                )

    video_filename = (
        os.path.sep
        + df[youtube_id]
        + '_'
        + df[start].astype(str).str.zfill(zfill)
        + '_'
        + df[end].astype(str).str.zfill(zfill)
        + ext
    )

    return pd.concat(df_order) + video_filename


class KineticsSplitConfig(NamedTuple):
    """Specifies the split ocnfiguration of a kinetics dataset.

    Attributes
    ----------
    train: bool = False
    validate: bool = False
    test: bool = False
    NaN: bool = False
    """
    train: bool = False
    validate: bool = False
    test: bool = False
    NaN: bool = False


class LabelConfig(NamedTuple):
    """A Label Configuration that specifies the details about the labels.  This
    configuration informs which samples are kept based on their labels and any
    types of masking of labels that occur for certain types of labels.

    Attributes
    ----------
    name : str
        The name of the label set expressed by this label configuration.
    known : list = True
        The known labels whose symbols are used as is.
    unknown : list = True
        The labels that are kept but whose symbols are masked as `unknown`.
    unlabeled : list = None
        The labels that are kept but whose symbols are masked as `None`. This
        differs from unknown because unknown may be used to represent samples
        whose labels are certain to be different from the known labels.

    Note
    ----
    Currently, known, unknown, and unlabeled may all share labels, which is
    undesirable and checking for this case would be good to avoid user error.
    Otherwise, known labels in unknown or unlabeled will be masked in
    KineticsUnified with unknown masking taking precedence over unlabeled
    masking.

    every labels : str = None
            Path to file that loads a list of str label names. If not provided
            and known is True in LabelConfig, then the label encoder is based
            on the sorted unique labels from the knowns.
    """
    name : str
    known : list = True
    unknown : list = True
    unlabeled : list = None


class KineticsUnifiedSubset(NamedTuple):
    """Specifies a data subset of the KineticsUnified data.

    Attributes
    ----------
    kinetics400 : KineticsSplitConfig = None
    kinetics600 : KineticsSplitConfig = None
    kinetics700_2020 : KineticsSplitConfig = None
    labels : LabelConfig = None
    """
    kinetics400: KineticsSplitConfig = None
    kinetics600: KineticsSplitConfig = None
    kinetics700_2020: KineticsSplitConfig = None
    labels : LabelConfig = None


def update_subset_mask(df, mask, split_config, col):
    if split_config.train:
        mask |= df[col] == 'train'
    if split_config.validate:
        mask |= df[col] == 'validate'
    if split_config.test:
        mask |= df[col] == 'test'
    if split_config.NaN:
        mask |= pd.isna(df[col])
    return mask


def subset_kinetics_unified(df, subset):
    """Provides a subset of the given DataFrame based on the subset object.
    Args
    ----
    df : pandas.DataFrame
        Kinetics Unified DataFrame
    subset : KineticsUnifiedSubset
        The subset configuration that informs what to keep from the original
        DataFrame.

    Note
    ----
    This is inefficient as it expects the DataFrame to be loaded into
    memory already, and so if this is the case, then it may be better to use
    Dask or some file reader to read the parts of the csv that are to be
    kept, although this only matters for large csvs.
    """
    if not isinstance(subset, KineticsUnifiedSubset):
        raise TypeError(' '.join([
            'Expected subset to be `KineticsUnifiedSubset`, not',
            f'{type(subset)}',
        ]))

    mask = pd.Series([False] * len(df))

    if subset.kinetics400 is not None:
        mask = update_subset_mask(
            df,
            mask,
            subset.kinetics400,
            'split_kinetics400',
        )
    if subset.kinetics600 is not None:
        mask = update_subset_mask(
            df,
            mask,
            subset.kinetics600,
            'split_kinetics600',
        )
    if subset.kinetics700_2020 is not None:
        mask = update_subset_mask(
            df,
            mask,
            subset.kinetics700_2020,
            'split_kinetics700_2020',
        )

    if subset.labels is not None:
        label_set = set()
        if subset.labels.known is True: # Use all unique labels
            label_set |= set(df[subset.labels.name].unique())
        elif subset.labels.known not in {None, False}:
            label_set |= set(subset.labels.known)

        if subset.labels.unknown is True: # Include 'unknown' label in unknowns
            label_set |= set(['unknown'])
        elif subset.labels.unknown not in {None, False}:
            label_set |= set(subset.labels.unknown)

        if subset.labels.unlabeled is not None:
            label_set |= set(subset.labels.unlabeled)

        # Update the mask to exlude all samples whose labels are not in config
        mask &= np.logical_or.reduce(
            [df[subset.labels.name] == label for label in label_set],
            axis=0,
        )

    return df[mask]


@dataclass
class KineticsUnified(torch.utils.data.Dataset):
    """The features extracted sample aligned dataset for Kinetics 400, 600, and
    700_2020.

    Attributes
    ----------
    data : pandas.DataFrame
        A DataFrame whose rows represents each sample's annotation data.
    kinetics_class_map : str | pandas.DataFrame = None
        A mapping of the unique classes in each Kinetics dataset to one
        another. May include other mappings as well. This serves the role of
        older unique `class_labels`.
    sample_dirs : KineticsRootDirs = None
    subset : KineticsUnifiedSubset = None
    unlabeled_token : str = None
    device : str = 'cpu'
    dtype : str = torch.float32
    return_index : bool = False
        If True, returns only the contents within the self.data DataFrame's
        column `labels` for the sample. Otherwise, the default, returns
        the index of the sample with the dataframe.
    log_warn_file_not_found : bool = False
        If True, logs a warning about a sample file not being found. Otherwise
        raises the typical FileNotFoundError.
    label_enc : NominalDataEncoder
        Label encoder for this dataset.
    one_hot : bool = True
        If the labels, if available, from the dataset should be given in one
        hot encodings.
    """
    annotation_path : InitVar[str]
    kinetics_class_map :  InitVar[str] = None
    sample_dirs : KineticsRootDirs = None
    subset :  InitVar[KineticsUnifiedSubset] = None
    unlabeled_token : str = None
    filepath_order : InitVar[list] = None
    reorder : InitVar[list] = None
    ext : InitVar[str] = '_feat.pt'
    device : InitVar[str] = 'cpu'
    dtype : InitVar[str] = torch.float32
    return_index : bool = False
    log_warn_file_not_found : bool = False
    blacklist : InitVar[str] = None
    whitelist : InitVar[str] = None
    one_hot : bool = True
    k700_suffix_label : InitVar[bool] = True

    def __post_init__(
        self,
        annotation_path,
        kinetics_class_map,
        subset,
        filepath_order,
        reorder,
        ext,
        device,
        dtype,
        blacklist,
        whitelist,
        k700_suffix_label,
    ):
        """
        Args
        ----
        annotation_path : str
            The filepath to the annotations for the data.
        kinetics_class_map : see self
        sample_dirs : see self
        subset : see self
        unlabeled_token : see self
        filepath_order : list = None
            The order of the pd.DataFrame columns to use for filepath priority.
        reorder : list = None
            Reordering of the `sample_dirs` indices.
        ext : str = '_feat.pt'
            The string suffix to expect at the end of feature files and
            includes the file extention within it.
        device : see self
        dtype : see self
        return_index : see self
        log_warn_file_not_found : see self
        blacklist : str = None
            Path to a file that loads a list of str filenames that are the
            files that are NOT allowed within the data, removing them if they
            exist.
        whitelist : str = None
            Path to a file that loads a list of str filenames that are the only
            ones to be allowed within the data, thus removing any others if
            they exist.
        k700_suffix_label : bool = True
            When True, add the label name in lower as a directory suffix in the
            path. Otherwise do not add any suffix directories.
        split_prefix : bool = None
            The prefix to add to the beginning of the Kinetics split portion of
            the data directory.
        """
        self.device = torch.device(device)
        self.dtype = torch_dtype(dtype)

        # Load the kinetics class map.
        if isinstance(kinetics_class_map, str):
            map_ext = os.path.splitext(kinetics_class_map)[-1]
            if map_ext == '.csv':
                self.kinetics_class_map = pd.read_csv(kinetics_class_map)
            elif map_ext == '.json':
                self.kinetics_class_map = pd.read_json(kinetics_class_map)
            else:
                raise ValueError(' '.join([
                    'Expected `kinetics_class_map` as a str to have',
                    f'extention, ".csv" or ".json", not `{map_ext}`',
                ]))
        elif isinstance(kinetics_class_map, pd.DataFrame):
            self.kinetics_class_map = kinetics_class_map
        elif kinetics_class_map is not None:
            raise TypeError(' '.join([
                '`kinetics_class_map` expected to be str or pd.DataFrame, but',
                f'recieved type: {type(kinetics_class_map)}',
            ]))

        # Read in the annotation csv, and cast known columns to str.
        self.data = pd.read_csv(
            annotation_path,
            dtype={
                'label_kinetics400': str,
                'label_kinetics600': str,
                'label_kinetics700_2020': str,
                'split_kinetics400': str,
                'split_kinetics600': str,
                'split_kinetics700_2020': str,
            },
        )

        # Using subset config, prune the undesired samples from this dataset
        # instance, leaving only a subset of all of the Kinetics unified data.
        if subset is not None:
            # Keep the parts of the dataframe specified by the subset config
            self.data = subset_kinetics_unified(self.data, subset)

            if subset.labels is not None:
                # Label Config determines the data column to use for labels.
                # This way, if any masking occurs, it does not change the
                # source the dataframe column which can then be used for eval.
                self.data[['labels']] = self.data[[subset.labels.name]]

                # Mask the unknowns and unlabeled samples.
                if subset.labels.unknown not in {None, False, True}:
                    # Mask the unknowns
                    self.data.loc[
                        np.logical_or.reduce(
                            [
                                self.data['labels'] == label
                                for label in subset.labels.unknown
                            ],
                            axis=0,
                        ),
                        'labels',
                    ] = 'unknown'

                if subset.labels.unlabeled is not None: # Mask the unlabeled
                    self.data.loc[
                        np.logical_or.reduce(
                            [
                                self.data['labels'] == label
                                for label in subset.labels.unlabeled
                            ],
                            axis=0,
                        ),
                        'labels',
                    ] = self.unlabeled_token

                if isinstance(subset.labels.known, str):
                    labels = load_file_list(labels)
                elif subset.labels.known is True: # Yes, _IS_ True.
                    labels = self.data['labels'].unique()
                    labels.sort()
                elif subset.labels.known is None:
                    labels = []
                elif not isinstance(subset.labels.known, list):
                    raise TypeError(' '.join([
                        'subset.labels.known unexpected type!',
                        f'{type(subset.labels.known)}',
                    ]))
                self.label_enc = NominalDataEncoder(
                    labels,
                    unknown_key='unknown' if subset.labels.unknown else None,
                )
            else:
                logger.warning(
                    'subset given but no labels! No changes to DataFrame',
                )
        else:
            logger.debug('Subset was None!')

        # Blacklist unique samples
        if isinstance(blacklist, str):
            blacklist = load_file_list(blacklist)
        if isinstance(blacklist, list):
            self.data = self.data[
                ~get_filename(self.data, ext=None).isin(blacklist)
            ]

        # Whitelist unique samples
        if isinstance(whitelist, str):
            whitelist = load_file_list(whitelist)
        if isinstance(whitelist, list):
            self.data = self.data[
                get_filename(self.data, ext=None).isin(whitelist)
            ]

        # Create an index column for ease of accessing labels from DataLoader
        self.data['sample_index'] = self.data.index

        # Ensure the sample path exists for ease of grabbing.
        if 'sample_path' not in self.data:
            if self.sample_dirs is None:
                raise ValueError(' '.join([
                    '`sample_path` column must be in annotation data or',
                    'sample_dirs is given to generate the video paths.',
                ]))
            if filepath_order is None:
                filepath_order = [
                    'split_kinetics400',
                    'split_kinetics600',
                    'split_kinetics700_2020',
                ]
            self.data['sample_path'] = get_path(
                self.data,
                self.sample_dirs \
                if reorder is None else self.sample_dirs[reorder],
                order=filepath_order,
                ext=ext,
                k700_suffix_label=k700_suffix_label,
                split_prefix=split_prefix,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """For the given index, load the corresponding sample labels.

        Args
        ----
        index : int

        Returns
        -------
        torch.Tensor | tuple
            A tuple whose first item is the sample feature encoding as a
            torch.Tensor, second is the sample index in the DataSet's DataFrame
            `data` to access labels etc outside of Torch computation.
        """
        # Given the index, obtain the sample's row from the DataFrame.
        sample = self.data.iloc[index]

        if self.return_index:
            return sample['sample_index']
        if self.label_enc:
            return torch.as_tensor(
                    self.label_enc.encode(
                        [sample['labels']],
                        one_hot=self.one_hot,
                    ).squeeze(),
                    dtype=self.dtype,
                )
        return sample['labels']


class KineticsUnifiedFeatures(KineticsUnified):
    """The Feature Extracted sample for Kinetics 400, 600, and 700_2020.

    Attributes
    ----------
    see KineticsUnified
    return_label : bool = False
        If True, which is the default, returns the label along with the
        input sample. The label typically is the smaple index to access
        the DataFrame row which contains all labels.
    """
    return_label : bool = False

    def __post_init__(self, *args, **kwargs):
        """Python dataclass post init

        Args
        ----
        see KineticsUnified.__post_init__
            This errors due to including the attributes as args, when only
            the args of __post_init__ are what is expected.
        """
        super().__post_init__(*args, **kwargs)

    # TODO Make efficient DataSet/DataLoader pre-loading, wherever is most
    # efficient and proper (in this DataSet class or outside).
    def __getitem__(self, index):
        """For the given index, load the corresponding sample feature encoding
        and labels.

        Args
        ----
        index : int

        Returns
        -------
        torch.Tensor | tuple
            A tuple whose first item is the sample feature encoding as a
            torch.Tensor, second is the sample index in the DataSet's DataFrame
            `data` to access labels etc outside of Torch computation.
        """
        # Given the index, obtain the sample's row from the DataFrame.
        sample = self.data.iloc[index]

        # Load from file. Hopefully, this is efficient enough.
        try:
            feature_extract = torch.load(
                sample['sample_path'],
                self.device,
            ).squeeze().to(self.dtype)
        except FileNotFoundError as e:
            if self.log_warn_file_not_found:
                logger.warning('%s: %s', e.strerror, e.filename)
                return None
            else:
                raise e

        if self.return_label:
            if self.return_index:
                return feature_extract, sample['sample_index']
            if self.label_enc:
                return feature_extract, \
                    torch.as_tensor(
                        self.label_enc.encode(
                            [sample['labels']],
                            one_hot=self.one_hot,
                        ).squeeze(),
                        dtype=self.dtype,
                    )
            return feature_extract, sample['labels']
        return feature_extract


class KineticsUnifiedVideos(KineticsUnified):
    """The video sample aligned dataset for Kinetics 400, 600, and 700_2020.

    Attributes
    ----------
    see KineticsUnified
    spatial_transform : torchvision.transforms.Compose = None
        An image transformation that is applied to every video frame. Default
        is at least a Compose consisting of ToTensor to ensure the np.array
        video frames are converted to pytorch tensors.
    video_loader : callable = status_video_frame_loader
        A callable that given a path loads the video frames from disk.
    frame_step_size : int = 5
        The step size used to select frames from the video to represent that
        video in the sample. Named gamma tau in the X3D paper.
    time_crops : int = 10
        The total temporal crops of each video. This is specifically for use in
        X3D.
    randomize_spatial_params : bool = False
        If True, randomizes the spatial transforms parameters.
    return_sample_status : bool = False
        If True, then __getitem__ returns the VideoStatus.value for the loaded
        image at the end of the returned tuple.
    """
    spatial_transform : Compose = Compose([ToTensor()])
    video_loader : callable = status_video_frame_loader
    frames : int = 300
    frame_step_size : int = 1
    time_crops : int = 1
    randomize_spatial_params : bool = False
    return_sample_status : bool = False
    ext : InitVar[str] = '.mp4'

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def __getitem__(self, index, get_label=False):
        """For the given index, load the corresponding sample video frames and
        labels.

        Returns
        -------
        tuple
            A tuple whose first item is the sample video frames as a
            torch.Tensor, second is the sample index in the DataSet's DataFrame
            `data` to access labels etc outside of Torch computation, and an
            optional third element which is the status value of loading the
            video as an int which corresponds to the enumeration
            `arn.data.dataloader_utils.VideoStatus`. The status value is only
            included if `return_sample_status` is True.
        Note
        ----
        This is built off from the X3D's dataloader.__get_item__().
        """
        # Given the index, obtain the sample's row from the DataFrame.
        sample = self.data.iloc[index]

        # Load the video
        video, status = self.video_loader(sample['sample_path'])

        # Get the representative frames of this video sample
        video = [
            video[i] for i in range(0, len(video))[::self.frame_step_size]
        ]
        # For multi-crop testing
        step = int((len(video) - self.frames)//(self.time_crops))

        # Apply spatial transform to video frames, if any
        if self.randomize_spatial_params:
            self.spatial_transform.randomize_parameters()
        video = [
            self.spatial_transform(Image.fromarray(img)) for img in video
        ]

        # Permute the video tensor such that its dimensions: T C H W -> C T H W
        video = torch.stack(video, 0).permute(1, 0, 2, 3)

        # This is related to X3D, where it wants multiple temporal crops.
        # Trim all videos in the batch to a maxium of self.frames and
        # temporal crop. . .?
        if step == 0:
            # TODO [explain with comment here] Why range time_crops?
            video = torch.stack(
                [video[:, :self.frames, ...] for i in range(self.time_crops)],
                0,
            )
        else:
            # TODO [explain with comment here]
            video = [
                video[:, i:i + self.frames, ...]
                for i in range(0, step * self.time_crops, step)
            ]

            # For every video, ensure the frames are padded with zeros.
            for i, frame in enumerate(video):
                if frame.shape[1] != self.frames:
                    # Padding in torch is absolutely bonkers, lol this pads
                    # dimension 1
                    video[i] = F.pad(
                        frame,
                        (0, 0, 0, 0, 0, self.frames - frame.shape[1]),
                        "constant",
                        0,
                    )

            video = torch.stack(video, 0)
        if get_label:
            sample_label = sample['labels']
        else:
            sample_label = sample['sample_index']

        # TODO consider returning the label token? or leave this collate_fn?
        if self.return_sample_status: # Returns the status code at end
            return video, sample_label, status.value
        return video, sample_label
