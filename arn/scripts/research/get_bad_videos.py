"""A Script to save a list of bad videos, whether missing or corrupted."""
from enum import Enum
import os
import logging

import pandas as pd
import torch
from tqdm import tqdm

from arn.data.kinetics_unified import (
    KineticsUnified,
    KineticsUnifiedSubset,
    KineticsRootDirs,
)

from arn.data.dataloader_utils import VideoStatus

from arn.scripts.research import arg_utils
from arn.scripts.research.clip.kinetics_clip_img_encode import \
    clip_transform_image_frames

from exputils.io import create_filepath, parse_args


def script_args(parser):
    arg_utils.har_dataset_general(parser)
    arg_utils.kinetics_root_dirs(parser)
    arg_utils.single_label_config(parser)
    arg_utils.single_kinetics_unified_subset(parser)


def post_script_args(args):
    return arg_utils.post_single_kinetics_unified_subset(
        arg_utils.post_kinetics_root_dirs(args)
    )


if __name__ == '__main__':
    args = post_script_args(parse_args(custom_args=script_args))

    if not isinstance(args.bad_samples_dir, str):
        raise ValueError('Must give this script `--bad_samples_dir` filepath.')

    logging.info('args = %s', args)
    logging.info('args.kinetic_root_dirs = %s', vars(args.kinetic_root_dirs))
    logging.info('args.subset = %s', args.subset)

    # Load the KineticsUnified dataset with the given KineticsUnifiedSubset and
    # ensure that collect_bad_samples = True
    kuni = KineticsUnified(
        args.annotation_path,
        args.kinetics_class_map_path,
        args.kinetic_root_dirs,
        args.subset,
        return_sample_status=True,
        spatial_transform=clip_transform_image_frames(244),
    )

    corrupt_videos = pd.DataFrame([], columns=kuni.data.columns)
    missing_videos = pd.DataFrame([], columns=kuni.data.columns)

    logging.info('len(kuni) = %d', len(kuni))
    logging.debug('%s', kuni.data)

    dataloader = torch.utils.data.DataLoader(
        kuni,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    # TODO use `get_path()` to check the K600 and K700 repeats of K400.

    # Loop through the dataset's videos, it will save those that fail to load.
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sample_indices = batch[1].numpy()

        logging.debug(
            'batch %d: %s',
            i,
            kuni.data.loc[sample_indices, 'video_path'],
        )

        status_codes = batch[-1].numpy()

        # if status == 'bad_video':
        corrupt_videos = corrupt_videos.append(kuni.data.iloc[
            sample_indices[status_codes == VideoStatus.CORRUPT.value]
        ])
        # elif status == 'path_dne':
        missing_videos = missing_videos.append(kuni.data.iloc[
            sample_indices[status_codes == VideoStatus.MISSING.value]
        ])

    # Save the corrupt samples, if any. Log if there are any or not.
    corrupt_videos.to_csv(create_filepath(os.path.join(
        args.bad_samples_dir,
        'corrupt_videos.csv',
    )))

    # Save the missing samples, if any. Log if there are any or not.
    missing_videos.to_csv(create_filepath(os.path.join(
        args.bad_samples_dir,
        'missing_videos.csv',
    )))
