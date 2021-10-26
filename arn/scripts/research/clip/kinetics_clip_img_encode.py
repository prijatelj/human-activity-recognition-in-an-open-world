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
    shuffle=True,
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
        shuffle=shuffle,
        num_workers=12,
        pin_memory=True,
    )

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    print('val',len(datasets['val']))
    print('datasets created')

    return datasets, dataloaders


def main(
    image_path=None,
    label_path=None,
    pred_path=None,
    device='cuda',
    model_path='ViT-B/32',
    load_encoded_labels=True,
    *args,
    **kwargs,
):
    if image_path is None and label_path is None and pred_path is None:
        raise ValueError(
            '`image_path`, `label_path`, and `pred_path` are all None',
        )

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
    dataset, dataloader = get_kinetics_dataloader(
        spatial=spatial,
        randomize_spatial_params=False,
        *args,
        **kwargs,
    )
    dataset = dataset['val']
    dataloader = dataloader['val']

    # TODO the CLIP Zeroshot and feature repr needs to be a module used by
    # baseline. Save or reuse whatever boilerplate is around CLIP encoding here

    if load_encoded_labels and label_path:
        encoded_labels = torch.load(label_path)
    else:
        encoded_labels = None

    encoded_images = None
    preds = None

    # Encode the images
    with torch.no_grad():
        # Calculate & save, or load the label text CLIP features
        if encoded_labels is None and (label_path or pred_path):
            # Get the unique text labels and sort them
            text_labels = sorted({d['label'] for d in dataset.data})

            # Encode the Labels
            label_encs = model.encode_text(text_labels)

            # Save the encoded text labels.
            torch.save(exputils.io.create_filepath(label_path), encoded_labels)

        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in bar:
            if image_path or pred_path:
                # Make video frames Batch, Time, Channels, Height, Width,
                # again.
                inputs = inputs.permute(0, 2, 1, 2, 3)
                shape = inputs.shape

                # Flatten batch and time
                # Encode images and Reconstruct Batch and Time
                image_encs = model.encode_image(
                    inputs.flatten(0, 1),
                ).reshape(shape)

                if image_path:
                    # Store the encoded images
                    if encoded_images is None:
                        encoded_images = image_encs
                    else:
                        encoded_images = torch.stack(
                            (encoded_images, image_encs),
                        )

            if pred_path:
                # Calculate Zero-Shot predictions (Cosine Similarity * 100)
                # CLIP paper states the mean of the predictions was used for
                # K700

                # TODO calculate the label similarity per frame thru softmax
                similarity = (
                    100.0
                    * (
                        image_encs / image_encs.norm(dim=-1, keepdim=True)
                    ) @ (
                        encoded_labels
                        / encoded_labels.norm(dim=-1, keepdim=True)
                    ).T
                ).softmax(dim=-1)

                # TODO Save the averaging of the resulting prob vector
                if preds is None:
                    preds = similarity
                else:
                    preds = torch.stack((preds, similarity))

    if image_path:
        # Save the encoded images
        torch.save(exputils.io.create_filepath(image_path), encoded_images)
    if pred_path:
        # Save the encoded images
        torch.save(exputils.io.create_filepath(pred_path), preds)


if __name__ == '__main__':
    targets = ["/media/sgrieggs/pageparsing/kinetics-dataset-400-test/"]

    # TODO Pay attention to main param defaults and set shuffle to False

    for x in targets:
        main(root=x)
