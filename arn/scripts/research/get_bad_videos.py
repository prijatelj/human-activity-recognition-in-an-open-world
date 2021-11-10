"""A Script to save a list of bad videos, whether missing or corrupted."""
import os
import logging

import torch
from tqdm import tqdm

from arn.data.kinetics_unified import (
    KineticsUnified,
    KineticsUnifiedSubset,
    KineticsRootDirs,
)

from arn.scripts.research import arg_utils
from arn.scripts.research.clip.kinetics_clup_img_encode import \
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

def main():
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
        collect_bad_samples=True,
        spatial_transform=clip_transform_image_frames(244),
    )

    logging.info('len(kuni) = %d', len(kuni))
    logging.debug('%s', kuni.data)

    #"""
    dataloader = torch.utils.data.DataLoader(
        kuni,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )
    #"""

    # TODO use `get_path()` to check the K600 and K700 repeats of K400.

    # Loop through the dataset's videos, it will save those that fail to load.
    for i, (vid, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        logging.debug('%d: %s', i, labels['video_path'])

    # Save the corrupt samples, if any. Log if there are any or not.
    kuni.data.iloc[[i.index for i in kuni.corrupt_videos]] \
        .to_csv(create_filepath(os.path.join(
            args.bad_samples_dir,
            'corrupt_videos.csv',
        )))

    # Save the missing samples, if any. Log if there are any or not.
    kuni.data.iloc[[i.index for i in kuni.missing_videos]] \
        .to_csv(create_filepath(os.path.join(
            args.bad_samples_dir,
            'missing_videos.csv',
        )))


if __name__ == '__main__':
    main()
