"""Dataset for the sample aligned Kinetics 400, 600, and 700_2020 datasets."""
from collections import namedtuple
from dataclasses import dataclass, InitVar
from functools import partial
import os
import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from arn.data.dataloader_utils import status_video_frame_loader


class KineticsRootDirs(object):
    """Stores the root directory paths for each Kinetics dataset."""
    def __init__(
        self,
        kinetics400_dir='',
        kinetics600_dir='',
        kinetics700_2020_dir='',
        root_dir='',
    ):
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

    def get_path(
        self,
        df,
        youtube_id='youtube_id',
        start='time_start',
        end='time_end',
        ext='.mp4',
        zfill=6,
        order=[
            'split_kinetics400',
            'split_kinetics600',
            'split_kinteics700_2020',
        ],
        #split_prefix=['kinetics-dataset-400-', None, None],
        #split_suffix=[None, None, ''],
    ):
        """Create filepath for every video, preference based on orderd datasets

        Args
        ----
        df : pd.DataFrame
            The Kinetics Unified DataFrame.
        order : list(str)
            Determines the order of which columns are prioritized for getting
            video paths. Default is to prefer earlier Kinetics datasets, based
            on release date.
        split_prefix : list(str)
            NOT Implemented. An attempt at generalizaiton postponed.
            The prefix to add to the beginning of the Kinetics split portion of
            the data directory.
        split_suffix : list(str)
            NOT Implemented. An attempt at generalizaiton postponed.
            The suffix to add to the end of the Kinetics split portion of the
            data directory.

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
        # Save when each sample is present in each dataset
        not_null = 1 ^ pd.isnull(df[order])

        # TODO include support for replacing videos that were corrupted in
        # earlier versions with those that are available and working in later
        # Kinetics versions.

        dset_num_regex = re.compile('split_kinetics(?P<dnum>.*)')

        # Create the filepaths in order of preference of source dataset.
        df_order = []
        for i, col in enumerate(order):
            dset_num = dset_num_regex.findall(col)[0]

            if i == 0:
                mask_or = not_null[col]
                mask = mask_or
            else:
                # Save a Kinetics sample if not in other Kinetics (mask_or)
                mask_or |= not_null[col]
                # AND(NAND(other_kinetics, not_null), not_null)
                mask = (1 ^ (mask_or & not_null[col])) & not_null[col]

            if dset_num == '400':
                df_order.append(
                    self.kinetics400_dir
                    + os.path.sep
                    + 'kinetics-dataset-400-'
                    + df[col].replace('validate', 'val')[mask],
                )
            elif dset_num == '600':
                df_order.append(
                    self.kinetics600_dir
                    + os.path.sep
                    + df[col][mask]
                )
            elif dset_num == '700_2020':
                df_order.append(
                    self.kinetics700_2020_dir
                    + os.path.sep
                    + df[col][mask]
                    + os.path.sep
                    + df[col.replace('split', 'label')][mask]
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
    known : list
        The known labels whose symbols are used as is.
    unknown : list
        The labels that are kept but whose symbols are masked as `unknown`.
    unlabeled : list
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
    """
    name : str
    known : list
    unknown : list = None
    unlabeled : list = None


class KineticsUnifiedSubset(NamedTuple):
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
    df : pd.DataFrame
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
        if subset.labels.known is not None:
            label_set |= set(subset.labels.known)
        if subset.labels.unknown is not None:
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
    """The dataset for the sample aligned Kinetics 400, 600, and 700_2020
    datasets.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame whose rows represents each sample's annotation data.
    kinetics_class_map : pd.DataFrame
        A mapping of the unique classes in each Kinetics dataset to one
        another. May include other mappings as well. This serves the role of
        older unique `class_labels`.
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
    sample_tuple : namedtuple
        A namedtuple of the Kinetics Unified csv column names and serves to
        contain a single sample (row) of the DataFrame with the loaded video
        frames and annotations.
    """
    annotation_path : InitVar[str]
    kinetics_class_map :  InitVar[str]
    video_dirs : KineticsRootDirs = None
    subset :  InitVar[KineticsUnifiedSubset] = None
    spatial_transform : Compose = Compose([ToTensor()])
    video_loader : callable = status_video_frame_loader
    frames : int = 300
    frame_step_size : int = 1
    time_crops : int = 1
    randomize_spatial_params : bool = False
    return_sample_status : bool = False
    corrupt_videos : list = None
    missing_videos : list = None
    unlabeled_token : str = None
    filepath_order : InitVar[list] = [
        'split_kinetics400',
        'split_kinetics600',
        'split_kinteics700_2020',
    ]

    def __post_init__(
        self,
        annotation_path,
        kinetics_class_map,
        subset,
        filepath_order,
    ):
        if isinstance(kinetics_class_map, str):
            ext = os.path.splitext(kinetics_class_map)[-1]
            if ext == '.csv':
                self.kinetics_class_map = pd.read_csv(kinetics_class_map)
            elif ext == '.json':
                self.kinetics_class_map = pd.read_json(kinetics_class_map)
            else:
                raise ValueError(' '.join([
                    'Expected `kinetics_class_map` as a str to have',
                    f'extention, ".csv" or ".json", not `{ext}`',
                ]))
        elif isinstance(kinetics_class_map, pd.DataFrame):
            self.kinetics_class_map = kinetics_class_map
        else:
            raise TypeError(' '.join([
                '`kinetics_class_map` expected to be str or pd.DataFrame, but',
                f'recieved type: {type(kinetics_class_map)}',
            ]))

        # TODO consider something like Dask for parallel reading of csvs
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
                if subset.labels.unknown is not None: # Mask the unknowns
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

        if 'video_path' not in self.data:
            if self.video_dirs is None:
                raise ValueError(' '.join([
                    '`video_path` column must be in annotation data or',
                    'video_dirs is given to generate the video paths.',
                ]))
            self.data['video_path'] = self.video_dirs.get_path(
                self.data,
                order=filepath_order,
            )

        # TODO when multiprocessing with num_workers for torch DataLoader this
        # fails to pickle.
        #self.sample_tuple = namedtuple(
        #    'kinetics_unified_sample',
        #    ['video'] + self.data.columns.tolist(),
        #)

        # Create an index column for ease of accessing labels from DataLoader
        self.data['sample_index'] = self.data.index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """For the given index, load the corresponding sample video frames and
        labels.

        Returns
        -------
        namedtuple
            A namedtuple whose fields match that of the columns of the
            DataFrame, but where the video frames are included in the beginning
            index as a torch.Tensor.
        """
        # Given the index, obtain the sample's row from the DataFrame.
        sample = self.data.iloc[index]

        # Load the video
        video, status = self.video_loader(sample['video_path'])

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

        # TODO consider returning the label token? or leave this collate_fn?
        if self.return_sample_status: # Returns the status code at end
            return video, sample['sample_index'], status.value
        return video, sample['sample_index']

    #def __del__(self):
    #    """Deconstructor to close any open files upon deletion."""
    #    self.data.close()
