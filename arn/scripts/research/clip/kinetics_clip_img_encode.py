"""Script that encodes and saves the given Kinetics images use CLIP."""
import argparse
import copy
import logging
import os
import pdb
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

import clip
from arn.data.kinetics import Kinetics
import arn.x3d as resnet_x3d
from torchsummary import summary
from transforms.spatial_transforms_old import (
    CenterCrop,
    CenterCropScaled,
    Compose,
    MultiScaleRandomCrop,
    MultiScaleRandomCropMultigrid,
    Normalize,
    RandomHorizontalFlip,
    ToTensor,
)
from transforms.target_transforms import ClassLabel
from transforms.temporal_transforms import TemporalRandomCrop
from utils.apmeter import APMeter

import exputils

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-task', default='class', type=str)
parser.add_argument('-config', default="config.txt", type=str)
parser.add_argument('-id', default="", type=str)
KINETICS_CLASS_LABELS = 'data/kinetics400_labels.txt'
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
ID = args.id
# set_batch_size
BS = 1 # 6
BS_UPSCALE = 2
INIT_LR = 0.0002 * BS_UPSCALE
GPUS = 1

X3D_VERSION = 'M'

with open(args.config, "r") as f:
    raw_lines = f.readlines()
lines = []
for x in raw_lines:
    if x[0] != "#":
        lines.append(x)
KINETICS_VAL_ROOT = lines[0][:-1]
KINETICS_VAL_ANNO = lines[1][:-1]
# KINETICS_VAL = lines[2][:-1]
model_save_path = lines[3][:-1]
save_txt_dir = lines[4][:-1]


def clip_transform_image_frames(
    n_px,
    means=(0.48145466, 0.4578275, 0.40821073),
    stds=(0.26862954, 0.26130258, 0.27577711),
):
    """Transforms video frames from Kinetics output to CLIP expectation."""
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize(means, stds),
    ])


def get_kinetics_dataloader(
    init_lr=INIT_LR,
    max_epochs=1,
    root=KINETICS_VAL_ROOT,
    anno=KINETICS_VAL_ANNO,
    val_anno=KINETICS_VAL_ANNO,
    batch_size=BS*BS_UPSCALE,
    task="class",
    transform=None,
    spatial=None,
    randomize_spatial_params=True,
    frames=300,
    gamma_tau=1,
    crops=1,
):
    """Instantiates and returns the Kinetics Dataloader given the params."""
    # TODO all this boiler plate need put into a Kinetics Dataloader wrapper
    # class or some pipeline object.

    if spatial is None:
        raise ValueError('`spatial` is None. Spatial transform must be given')

    validation_transforms = {
        'spatial':  spatial,
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }

    val_dataset = Kinetics(
        root,
        KINETICS_VAL_ANNO,
        KINETICS_CLASS_LABELS,
        'val',
        spatial_transform = validation_transforms['spatial'],
        temporal_transform = validation_transforms['temporal'],
        target_transform = validation_transforms['target'],
        sample_duration=frames,
        gamma_tau=gamma_tau,
        crops=crops,
        randomize_spatial_params=randomize_spatial_params,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )

    #num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    #cur_iterations = steps * num_steps_per_update
    # iterations_per_epoch = len(dataset)//batch_size
    #val_iterations_per_epoch = len(val_dataset)//(batch_size//2)
    # max_steps = iterations_per_epoch * max_epochs

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    print('val',len(datasets['val']))
    # print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')

    return dataloaders


# TODO ray parallelize the encoding process of images for speed.
def main(output_dir, device='cuda', model_path='ViT-B/32', *args, **kwargs):
    KINETICS_MEAN = [110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255]
    KINETICS_STD = [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255]

    # TODO Load in CLIP Pre-trained
    model, preprocess = clip.load(model_path, device)

    # Preprocess images: Go from Kinetics form to CLIP preprocess expectation
    # 224 input_resolution (based on clip_model.visual.input_resolution
    # CLIP process is to resize the PIL Image to this resolution using BICUBIC
    # interpolation, then CenterCrop, convert to RGB, and Normalize
    # Kinetics dataloader outputs clips as (Channels, Timeframe, Height, Width)
    # CLIP expects them as (B, C, H, W), so B for T
    spatial = clip_transform_image_frames(
            model.visual.input_resolution,
            KINETICS_MEAN,
            KINETICS_STD,
    )

    # Extract useful vars from kwargs
    if 'frames' in kwargs:
        frames = kwargs['frames']
    else:
        frames = 300

    # Get Kinetics Dataloader for specified data
    dataloader = get_kinetics_dataloader(
        spatial=spatial,
        randomize_spatial_params=False,
        *args,
        **kwargs,
    )['val']

    output_dir = exputils.io.create_filepath(output_dir)

    # TODO Calculate the text encoding of every Kinetics class

    # Encode the images
    with torch.no_grad():
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloaders))
        for i, (inputs, labels) in bar:
            # Make video frames Batch, Time, Channels, Height, Width, again.
            inputs = inputs.permute(0, 2, 1, 2, 3)
            shape = inputs.shape

            # Flatten batch and time
            # Encode images and Reconstruct Batch and Time
            image_encs = model.encode_image(
                inputs.flatten(0, 1),
            ).reshape(shape)

            # TODO Save Zero-Shot predictions while you're at it
            # TODO Cosine similarity
            # CLIP paper states the mean of the predictions was used for K700

            # Pick the top 5 most similar labels for the image
            image_enc /= image_enc.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_inc @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

    # TODO Save the encoded images
    raise NotImplementedError()


if __name__ == '__main__':
    targets = ["/media/sgrieggs/pageparsing/kinetics-dataset-400-test/"]

    for x in targets:
        main(root=x)
