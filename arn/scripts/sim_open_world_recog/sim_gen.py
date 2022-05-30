"""Generates the Simulated Datasets using Gaussian distributions for each
class.
"""
import os
import math

import numpy as np
import pandas as pd
import ray
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath

from arn.data.kinetics_unified import get_filename

import logging
logger = logging.getLogger(__name__)


class SimClassifyGaussians(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to. The Gaussian distributions are labeled their index which
    starts at zero at the top most Gaussian centered at [0, 1] and labels the
    rest that follow clockwise around the unit circle.

    Attributes
    ----------
    mvns : list
        List of torch.distirbutions.multivariate_normal.MultivariateNormal.
        The source distributions per class.
    label_enc :  NominalDataEncoder = None
        The label encoder if the labels are some different symbol than the
        index of the mvns.
    """
    def __init__(self, locs=None, scales=0.2, labels=None, seed=0):
        """
        Args
        ----
        locs : list = None
            TODO docstr support: list(float) | list(list(float)) = None
            Defaults to 4 gaussian locations =
                [[1, 0], [0, 1], [-1, 0], [0, -1]]
        scales : float = 0.2
            TODO docstr support: float | list(float) = 0.2
            The scales of the gaussians in the mixture.
        labels : list = None
        seed : int = 0
        """
        if seed is not None:
            # Set seed: Seems I cannot carry an individual RNG easily...
            torch.manual_seed(seed)

        # Create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs = [[0, 1], [1, 0],  [0, -1], [-1, 0]]

        if not isinstance(scales, list):
            scales = [scales] * len(locs)

        self.mvns = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.Tensor(loc),
                torch.eye(2) * scales[i],
            )
            for i, loc in enumerate(locs)
        ]

        if labels:
            self.label_enc = NominalDataEncoder(labels)
        else:
            self.label_enc = None

    def eq_sample_n(self, num, randperm=True):
        if randperm:
            idx = torch.randperm(num * len(self.mvns))
            return (
                torch.cat([mvn.sample_n(num) for mvn in self.mvns])[idx],
                torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten()[idx],
            )
        return (
            torch.cat([mvn.sample_n(num) for mvn in self.mvns]),
            torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten(),
        )


def gen_inc_sim_dataset(
    start_sim,
    locs,
    scales,
    incs_per_new_class,
    eq_samples_per_inc,
    dataset_id,
    seed=0,
):
    sim_set = SimClassifyGaussians(
        locs,
        scales=scales,
        labels=[f'k{dataset_id}-{i}' for i in range(1, len(locs)+1)],
        seed=seed,
    )
    start_sim.label_enc.append(list(sim_set.label_enc))

    inc_samples = []
    inc_labels = []
    inc_split = []
    inc_youtube_id = []
    inc_time_end = []
    inc_time_start = []

    n_prior_classes = len(start_sim.mvns)

    # Loop through new classes
    for i in range(1, len(locs) + 1):
        # Loop through incrments per new classes
        for j in range(incs_per_new_class):
            # Loop through the splits
            for split_idx, split in enumerate(['train', 'validate', 'test']):
                samples, labels = start_sim.eq_sample_n(eq_samples_per_inc)
                inc_samples.append(samples)
                inc_labels.append(labels)

                # This Dsets prior seen class' samples
                for k in range(0, i - 1):
                    samples = sim_set.mvns[k].sample_n(eq_samples_per_inc)
                    inc_samples.append(samples)
                    inc_labels.append(torch.Tensor(
                        [n_prior_classes + k] * eq_samples_per_inc
                    ))

                # New class' samples
                #if j == 1:
                samples = sim_set.mvns[i - 1].sample_n(eq_samples_per_inc * i)
                inc_labels.append(torch.Tensor(
                    [n_prior_classes + i - 1] * (eq_samples_per_inc * i)
                ))
                #else:
                #    samples = sim_set.mvns[i - 1].sample_n(eq_samples_per_inc)
                #    inc_labels.append(torch.Tensor(
                #        [n_prior_classes + i - 1] * (eq_samples_per_inc)
                #    ))
                inc_samples.append(samples)

                num_samples = eq_samples_per_inc * (n_prior_classes + i + i-1)
                inc_time_end += list(range(
                    len(inc_time_end),
                    len(inc_time_end) + num_samples
                ))
                inc_split += [split] * num_samples
                inc_time_start += [split_idx] * num_samples
                inc_youtube_id += [f'k{dataset_id}-{split}'] * num_samples
    inc_samples = torch.concat(inc_samples)
    inc_labels = torch.concat(inc_labels)

    df = get_kinetics_unified_df(
        start_sim.label_enc.decode(inc_labels.to(int).cpu().numpy()),
        inc_split,
        inc_youtube_id,
        inc_time_end,
        inc_time_start,
        dataset_id,
    )
    return sim_set, df, inc_samples


def get_kinetics_unified_df(
    inc_labels,
    inc_split,
    inc_youtube_id,
    inc_time_end,
    inc_time_start,
    dataset_id,
):
    assert dataset_id in {'400', '600', '700_2020'}
    df = pd.DataFrame({
        f'label_kinetics{dataset_id}': inc_labels, #[
        #    f'k{dataset_id}-{i}' for i in inc_labels
        #],
        f'split_kinetics{dataset_id}': inc_split,
        'youtube_id': inc_youtube_id,
        'time_end': inc_time_end,
        'time_start': inc_time_start,
    })
    if dataset_id != '400':
        df['label_kinetics400'] = None
        df['split_kinetics400'] = None
    if dataset_id != '600':
        df['label_kinetics600'] = None
        df['split_kinetics600'] = None
    if dataset_id != '700_2020':
        df['label_kinetics700_2020'] = None
        df['split_kinetics700_2020'] = None
    return df[[
        'label_kinetics400',
        'label_kinetics600',
        'label_kinetics700_2020',
        'time_end',
        'time_start',
        'youtube_id',
        'split_kinetics400',
        'split_kinetics600',
        'split_kinetics700_2020',
    ]]


@ray.remote
def save_sample(sample_tensor, dir_name, filename):
    torch.save(sample_tensor, os.path.join(dir_name, filename))


def gen_sim_dataset(root_dir):
    """Create the simulated data samples and save to disk as .pt files. Save
    the labels in the format of Kinetics Unified csv. The datasets [1,3]
    correspnd to Kinetics 400, 600, and 700_2020, respectively. There are
    train (0), validation (1), and test splits (2), which will have the same
    number of samples each. This covers the Kinetics Unified columns for label
    and split.  The columns for youtube_id, time_start, and time_end
    respectively are filled with the dataset string name, split number from [0,
    2] matching order [train, val, test], and the sample index within the
    datasplit. The these column values will not be unique per sample (row), but
    the resulting filepath will be unique as per kinetics sample identifiers.
    """
    logger.info('First dataset: Starting Increment')

    scale = 1 / 20
    start_sim = SimClassifyGaussians(
        scales=scale,
        labels=[f'k400-{i}' for i in range(1, 5)],
    )
    samples_per_class = 1000

    start_inc_samples = []
    start_inc_labels = []
    start_inc_youtube_id = []
    start_inc_split = []
    start_inc_time_end = []
    start_inc_time_start = []
    for split_idx, split in enumerate(['train', 'validate', 'test']):
        samples, labels = start_sim.eq_sample_n(samples_per_class)
        start_inc_samples.append(samples)
        start_inc_labels.append(labels)
        start_inc_youtube_id += [f'k400-{split}'] * samples_per_class * 4
        start_inc_time_end += list(np.arange(
            len(start_inc_time_end),
            len(start_inc_time_end) + samples_per_class * 4
        ))
        start_inc_split += [split] * 4 * samples_per_class
        start_inc_time_start += [split_idx] * 4 * samples_per_class
    start_inc_samples = torch.concat(start_inc_samples)
    start_inc_labels = torch.concat(start_inc_labels)

    # Format the DataFrame in Kinetics Unified format.
    df = get_kinetics_unified_df(
        start_sim.label_enc.decode(start_inc_labels.to(int).cpu().numpy()),
        start_inc_split,
        start_inc_youtube_id,
        start_inc_time_end,
        start_inc_time_start,
        dataset_id='400',
    )

    # Save the sample points to their own filepath
    dir_name = create_filepath(os.path.join(root_dir, 'sim_k400/'))
    save_dset = []
    for i, filename in enumerate(get_filename(df, ext='_feat.pt')):
        save_dset.append(
            save_sample.remote(start_inc_samples[i], dir_name, filename)
        )
    ray.get(save_dset)

    logger.info('Second dataset: Incremental Open World Recognition.')
    # Generate the 2nd dataset's samples
    new_class_locs = [
        # NE and SW
        [math.sqrt(2) / 2.0, math.sqrt(2) / 2.0],
        [- math.sqrt(2) / 2.0, - math.sqrt(2) / 2.0],
        # Pos, Neg Quadrant
        [math.sqrt(3) / 2.0, -0.5],
        [0.5, - math.sqrt(3) / 2.0],
        # Neg, Pos Quadrant
        [- math.sqrt(3) / 2.0, 0.5],
        [-0.5, math.sqrt(3) / 2.0],
        # Pos, Pos Quadrant
        [0.5, math.sqrt(3) / 2.0],
        [math.sqrt(3) / 2.0, 0.5],
        # Neg, Neg Quadrant
        [-0.5, - math.sqrt(3) / 2.0],
        [- math.sqrt(3) / 2.0, -0.5],
    ]
    incs_per_new_class = 4
    eq_samples_per_inc = 25

    new_sim, k6df, k6_sim_samples = gen_inc_sim_dataset(
        start_sim,
        new_class_locs,
        scale,
        incs_per_new_class,
        eq_samples_per_inc,
        dataset_id='600',
        seed=1,
    )

    # Update df with new labels
    df = df.append(k6df)

    # Save the sample points to their own filepath
    dir_name = create_filepath(os.path.join(root_dir, 'sim_k600/'))
    save_dset = []
    for i, filename in enumerate(get_filename(k6df, ext='_feat.pt')):
        save_dset.append(
            save_sample.remote(k6_sim_samples[i], dir_name, filename)
        )
    ray.get(save_dset)

    logger.info('Third dataset: Incremental Open World Recognition.')
    # Generate the 3rd dataset's samples
    in_btwn_5_11 = 3.5 * math.pi / 12
    new_class_locs = [
        [0, 2],
        [2, 0],
        [0, -2],
        [-2, 0],
        # radii = 2:  NE, SE, SW, NW
        [math.sqrt(2), math.sqrt(2)],
        [math.sqrt(2), - math.sqrt(2)],
        [- math.sqrt(2), - math.sqrt(2)],
        [- math.sqrt(2), math.sqrt(2)],
        # radii = 3: N, and in_btwn_5_11
        [0, 3],
        [3 * math.cos(in_btwn_5_11), 3 * math.sin(in_btwn_5_11)],
    ]

    # Combine new sim mvns and LabelEncoder within start_sim.
    start_sim.mvns += new_sim.mvns
    #start_sim.label_enc.append(list(new_sim.label_enc))

    new_sim, k7df, k7_sim_samples = gen_inc_sim_dataset(
        start_sim,
        new_class_locs,
        scale,
        incs_per_new_class,
        eq_samples_per_inc,
        dataset_id='700_2020',
        seed=2,
    )

    # Update df with new labels
    df = df.append(k7df)
    df.to_csv(os.path.join(root_dir, 'sim_kunified.csv'), index=False)

    # Save the sample points to their own filepath
    dir_name = create_filepath(os.path.join(root_dir, 'sim_k700/'))
    save_dset = []
    for i, filename in enumerate(get_filename(k7df, ext='_feat.pt')):
        save_dset.append(
            save_sample.remote(k7_sim_samples[i], dir_name, filename)
        )
    ray.get(save_dset)


if __name__ == '__main__':
    import sys
    ray.init(num_cpus=15, num_gpus=1)
    gen_sim_dataset(sys.argv[1])
