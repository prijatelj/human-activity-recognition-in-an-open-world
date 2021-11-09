"""Dataset for the sample aligned Kinetics 400, 600, and 700_2020 datasets."""
from collections import namedtuple
from dataclasses import dataclass, InitVar
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arn.data.dataloder_utils import status_video_frame_loader


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
        id='youtube_id',
        start='time_start',
        end='time_end',
        ext='.mp4',
        zfill=6,
    ):
        """Create filepath for every video, preferring older versions first.
        Args
        ----
        df : pd.DataFrame
            The Kinetics Unified DataFrame.

        Returns
        -------
        pd.Series
            A Pandas Series of the filepaths to each sample's video where the
            earlier Kinetics datasets' videos are prioritized.
        """
        # Save when each sample is present in each dataset
        not_null = 1 ^ pd.isnull(df[[
            'split_kinetics400',
            'split_kinetics600',
            'split_kinetics700_2020',
        ]])

        # Save when a Kinetics 600 sample is not in Kinetics 400
        k600_not_in_400 = (1 ^ (
            not_null['split_kinetics400']
            & not_null['split_kinetics600']
        )) & not_null['split_kinetics600']

        # Save when a Kinetics 700_2020 sample is not in either prior Kinetics
        k700_not_in_others = (1 ^ (
            (not_null['split_kinetics400'] | not_null['split_kinetics600'])
            & not_null['split_kinetics700_2020']
        )) & not_null['split_kinetics700_2020']

        # TODO include support for replacing videos that were corrupted in
        # earlier versions with those that are available and working in later
        # Kinetics versions.

        return (
            self.kinetics400_dir
            + f'{os.path.sep}kinetics-dataset-400-'
            + df["split_kinetics400"][not_null['split_kinetics400']]
        ).append(
            self.kinetics600_dir
            + os.path.sep
            + df['split_kinetics600'][k600_not_in_400]
        ).append(
            self.kinetics700_2020_dir
            + os.path.sep
            + df['split_kinetics700_2020'][k700_not_in_others]
            + os.path.sep
            + df['label_kinetics700_2020'][k700_not_in_others]
        ) \
        + df[id] + '_' + df[start].astype(str).str.zfill(zfill) \
        + '_' + df[end].astype(str).str.zfill(zfill) + ext


class BadVideoSample(NamedTuple):
    index: int
    video_path: str


class KineticsSplitConfig(NamedTuple):
    train: bool = False
    validate: bool = False
    test: bool = False
    NaN: bool = False


class KineticsUnifiedSubset(NamedTuple):
    kinetics400: KineticsSplitConfig = None
    kinetics600: KineticsSplitConfig = None
    kinetics700_2020: KineticsSplitConfig = None


def update_subset_mask(df, mask, split_config, col):
    if split_config.train:
        mask |= df[df['split_kinetics400'] == 'train']
    if split_config.validate:
        mask |= df[df['split_kinetics400'] == 'validate']
    if split_config.test:
        mask |= df[df['split_kinetics400'] == 'test']
    if split_config.NaN:
        mask |= np.isna(df['split_kinetics400'])
    return mask


def subset_kinetics_unified(df, subset):
    """Provides a subset of the given DataFrame based on the subset object.
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

    mask = pd.Series(False * len(df))

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

    return df[mask]


@dataclass
class KineticsUnified(torch.utils.data):
    """The dataset for the sample aligned Kinetics 400, 600, and 700_2020
    datasets.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame whose rows represents each sample's annotation data.
    annotation_path : str
        Path to the annotations of the unified Kinetics datasets.
    kinetics_class_map_path : str
        This serves the role of older unique `class_labels`.
    annotation_view : slice = None
        The view of the annotation samples as seen by this dataset class. This
        is either a slice, list of int indices, or bool array of len(data).

        This may not be necessary in the Dataset itself?
    spatial_transform : torchvision.transforms.Compose = None
    video_loader : callable = status_video_frame_loader
    frame_step_size : int = 5
        The step size used to select frames from the video to represent that
        video in the sample. Named gamma tau in the X3D paper.
    crops : int = 10
        The total temporal crops of each video.
    randomize_spatial_params : bool = True
        If True, randomizes the spatial transforms parameters.
    sample_tuple : namedtuple
        A namedtuple of the Kinetics Unified csv column names and serves to
        contain a single sample (row) of the DataFrame with the loaded video
        frames and annotations.
    """
    annotation_path : InitVar[str]
    kinetics_class_map :  InitVar[str]
    image_dirs : KineticsRootDirs = None
    subset :  InitVar[KineticsUnifiedSubset] = None
    spatial_transform : torchvision.transforms.Compose = None
    video_loader : callable = status_video_frame_loader
    frame_step_size : int = 5
    crops : int = 10
    randomize_spatial_params : bool = True
    collect_bad_samples : InitVar[bool] = False
    corrupt_samples : list(BadVideoSample) = None
    missing_samples : list(BadVideoSample) = None

    def __post_init__(
        self,
        annotation_path,
        kinetics_class_map,
        subset,
        collect_bad_samples,
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
                f'recieved type: type(kinetics_class_map)',
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


        # Include subset feature to make this dataset instance pertain to
        # only a subset of all of the Kinetics unified data.
        if subset is not None:
            # Keep the parts of the dataframe specified by the subset config
            self.data = subset_kinetics_unified(self.data, subset)

            # TODO add label subset logic for controlling when samples from
            # certain labels get included in this dataset instance.

        if 'video_path' not in self.data:
            if self.image_dirs is None:
                raise ValueError(' '.join([
                    '`video_path` column must be in annotation data or',
                    'image_dirs is given to generate the video paths.',
                ]))
            self.data['video_path'] = self.image_dirs.get_path(self.data)

        self.sample_tuple = namedtuple(
            'kinetics_unified_sample',
            ['video'] + self.data.columns.tolist(),
        )

        if collect_bad_samples:
            self.corrupt_videos = []
            self.missing_videos = []

    def __len__(self):
        return len(self.data)

    def __get_item__(self, index):
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
        sample = self.data[index]

        # Load the video
        video, status = self.video_loader(sample['video_path'])

        if self.collect_bad_samples:
            if status == 'bad_video':
                self.corrupt_videos.append(BadVideoSample(
                    index,
                    sample['video_path'],
                ))
            elif status == 'path_dne':
                self.missing_videos.append(BadVideoSample(
                    index,
                    sample['video_path'],
                ))

        # Get the representative frames of this video sample
        video = [
            video[i] for i in range(0, len(video))[::self.frame_step_size]
        ]
        step = int((len(video) - self.frames)//(self.crops))

        # Apply spatial transform to video frames, if any
        if self.spatial_transform is not None:
            if self.randomize_spatial_params:
                self.spatial_transform.randomize_parameters()
            video = [
                self.spatial_transform(Image.fromarray(img)) for img in video
            ]

        # Permute the video tensor such that its dimensions: T C H W -> C T H W
        video = torch.stack(video, 0).permute(1, 0, 2, 3)

        # Trim all videos in the batch to a maxium of self.frames and
        # temporal crop. . .?
        if step == 0:
            # TODO [explain with comment here] Why range crops?
            video = torch.stack(
                [video[:, :self.frames, ...] for i in range(self.crops)],
                0,
            )
        else:
            # TODO [explain with comment here]
            video = [
                video[:, i:i + self.frames, ...]
                for i in range(0, step * self.crops, step)
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

        return self.sample_tuple(video, *sample)

    #def __del__(self):
    #    """Deconstructor to close any open files upon deletion."""
    #    self.data.close()
