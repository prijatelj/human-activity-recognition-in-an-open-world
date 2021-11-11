"""Dataset for the sample aligned Kinetics 400, 600, and 700_2020 datasets."""
from collections import namedtuple
from dataclasses import dataclass
import os

import pandas as pd

from arn.data.kinetics_unified import KineticsUnified


@dataclass
class PARData(KineticsUnified):
    """The dataset for the PAR HAR data.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame whose rows represents each sample's annotation data.
    kinetics_class_map : pd.DataFrame
        A mapping of the unique classes in each Kinetics dataset to one
        another. May include other mappings as well. This serves the role of
        older unique `class_labels`.
    spatial_transform : torchvision.transforms.Compose = None
        An image transformation that is applied to every video frame.
    video_loader : callable = status_video_frame_loader
        A callable that given a path loads the video frames from disk.
    frame_step_size : int = 5
        The step size used to select frames from the video to represent that
        video in the sample. Named gamma tau in the X3D paper.
    time_crops : int = 10
        The total temporal crops of each video. This is specifically for use in
        X3D.
    randomize_spatial_params : bool = True
        If True, randomizes the spatial transforms parameters.
    sample_tuple : namedtuple
        A namedtuple of the Kinetics Unified csv column names and serves to
        contain a single sample (row) of the DataFrame with the loaded video
        frames and annotations.
    """
    def __post_init__(
        self,
        annotation_path,
        kinetics_class_map,
        subset,
        filepath_order,
        reorder,
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

        self.data = pd.read_csv(annotation_path)

        if 'video_path' not in self.data:
            if self.video_dirs is None:
                raise ValueError(' '.join([
                    '`video_path` column must be in annotation data or',
                    'video_dirs is given to generate the video paths.',
                ]))
            self.data['video_path'] = self.video_dirs \
                + self.data['anonymous_id']

        self.sample_tuple = namedtuple(
            'par_data_sample',
            ['video'] + self.data.columns.tolist(),
        )

        # Create an index column for ease of accessing labels from DataLoader
        self.data['sample_index'] = self.data.index
