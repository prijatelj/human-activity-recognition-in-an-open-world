"""Dataset for the sample aligned Kinetics 400, 600, and 700_2020 datasets."""
from collections import namedtuple
from dataclasses import dataclass, InitVar
import os

import pandas as pd
from PIL import Image
import torch
import torchvision
from tqdm import tqdm

from arn.data.dataloder_utils import (
    load_value_file,
    my_video_loader,
    pil_loader,
    get_default_image_loader,
    video_loader,
    get_default_video_loader,
    get_class_labels,
)


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


@dataclass
class KineticsUnified(torch.utils.data):
    """The dataset for the sample aligned Kinetics 400, 600, and 700_2020
    datasets.

    Attributes
    ----------
    image_dirs : KineticsRootDirs
    annotation_path : str
        Path to the annotations of the unified Kinetics datasets.
    kinetics_class_map_path : str
        This serves the role of older unique `class_labels`.
    annotation_view : slice = None
        The view of the annotation samples as seen by this dataset class. This
        is either a slice, list of int indices, or bool array of len(data).

        This may not be necessary in the Dataset itself?
    spatial_transform : torchvision.transforms.Compose = None
    temporal_transform : torchvision.transforms.Compose = None
    video_loader : callable = get_default_video_loader
    gamma_tau : int = 5
        Artifact naming from the X3D.
    crops : int = 10
    randomize_spatial_params : bool = True
    sample_tuple : namedtuple
        A namedtuple of the Kinetics Unified csv column names and serves to
        contain a single sample (row) of the DataFrame with the loaded video
        frames and labels.
    target_label : str = 'KineticsUnifiedPref700'
        The column which serves as the primary target label.

        This is probably not needed.
    """
    annotation_path : InitVar[str]
    kinetics_class_map :  InitVar[str]
    image_dirs : KineticsRootDirs = None
    annotation_view : torch.Tensor = None
    spatial_transform : torchvision.transforms.Compose = None
    temporal_transform : torchvision.transforms.Compose = None
    video_loader : callable = get_default_video_loader
    gamma_tau : int = 5
    crops : int = 10
    randomize_spatial_params : bool = True
    target_label : str = 'KineticsUnifiedPref700'
    collect_bad_samples : InitVar[bool] = False
    bad_samples : list(str) = None

    def __post_init__(
        self,
        annotation_path,
        kinetics_class_map,
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
            self.bad_samples = []

    def __len__(self):
        # TODO handle len of slice, sum of boolean array, len of int indices
        return len(self.data_index)

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
        # TODO given the index, obtain the sample's row from the DataFrame.

        # TODO Load the video

        # TODO Apply spatial transform to video frames, if any

        # TODO Apply temporal transform to video frames, if any

        return self.sample_tuple(video_frames, *row)

    #def __del__(self):
    #    """Deconstructor to close any open files upon deletion."""
    #    self.data.close()
