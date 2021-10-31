"""Script that encodes and saves the given Kinetics images use CLIP."""
import argparse
import logging
import os
import warnings

import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
)
#from torchvision.transforms.functional.InterpolationMode import BICUBIC
import tqdm

import clip
from arn.data.kinetics import Kinetics
from arn.transforms.target_transforms import ClassLabel

import exputils


def clip_transform_image_frames(
    n_px,
    means=(0.48145466, 0.4578275, 0.40821073),
    stds=(0.26862954, 0.26130258, 0.27577711),
):
    """Transforms video frames from Kinetics output to CLIP expectation."""
    return Compose([
        Resize(n_px, interpolation='BICUBIC'),
        CenterCrop(n_px),
        Normalize(means, stds),
    ])


def get_kinetics_dataloader(
    root,
    anno,
    class_labels,
    max_epochs=1,
    batch_size=1,
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
        #'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }

    val_dataset = Kinetics(
        root,
        anno,
        class_labels,
        'val',
        spatial_transform = validation_transforms['spatial'],
        #temporal_transform = validation_transforms['temporal'],
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


def text_zeroshot_encoding(model, label_texts, templates):
    """Given the classes and the provided templates, the CLIP text encoding for
    each class using all templtes for each class is returned.

    Args
    ----
    model : clip.model.CLIP
        CLIP model used for text encoding the given labels with their templates
    label_texts : list(str)
        The labels' text to be encoded with the given text templates
    templates : list(str)
        The text templates that every label text is inserted into and used to
        obtain the CLIP text encoding representation of that label text.

    Returns
    -------
    torch.Tensor
        The CLIP model's text encoding of each label text using the given
        templates in order. Shape =
    """
    with torch.no_grad():
        zeroshot_weights = []
        for label_text in label_texts:
            # Place the class label text inside each template text and tokenize
            texts = clip.tokenize(
                [template.format(label_text.lower()) for template in templates]
            ).cuda()

            # CLIP Encode the text, normalize dividing by L1 norm
            label_embeddings = model.encode_text(texts)
            label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

            # Obtain normalized mean as label encoding, again divide by L1 norm
            label_embedding = label_embeddings.mean(dim=0)
            label_embedding /= label_embedding.norm()

            zeroshot_weights.append(label_embedding)

    return torch.stack(zeroshot_weights, dim=1).cuda()


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
        shuffle=False,
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

    # Calculate & save, or load the label text CLIP features
    if encoded_labels is None and (label_path or pred_path):
        # Get the unique text labels and sort them
        label_texts = sorted({d['label'] for d in dataset.data})

        # Encode the Labels
        encoded_labels = text_zeroshot_encoding(label_texts, templates)

        # Save the encoded text labels.
        torch.save(exputils.io.create_filepath(label_path), encoded_labels)

    # Encode the images, optionally get preds
    with torch.no_grad():
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
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default='0', type=str)
    parser.add_argument('-task', default='class', type=str)
    parser.add_argument('-config', default="config.txt", type=str)
    parser.add_argument('-id', default="", type=str)

    # Must give these args:
    parser.add_argument('kinetics_root', help='Root of images.')
    parser.add_argument('kinetics_anno', help='Path to annotations.')
    parser.add_argument(
        'kinetics_class_labels',
        help='Path to Kinetics class labels txt ',
        #default='data/kinetics400_labels.txt',
    )
    parser.add_argument('model_path', help='Provide path to clip model.')

    parser.add_argument('--image_path', default=None)
    parser.add_argument(
        '--label_path',
        default=None,
        help='Path to encoded labels',
    )
    parser.add_argument('--pred_path', default=None)

    args = parser.parse_args()

    # set_batch_size
    BS = 1 # 6
    BS_UPSCALE = 2 # TODO What is this for and why???? X3D artifiact?

    #targets = ["/media/sgrieggs/pageparsing/kinetics-dataset-400-test/"]

    # TODO Pay attention to main param defaults and set shuffle to False

    for x in targets:
        main(
            image_path=args.image_path,
            label_path=args.label_path,
            pred_path=args.pred_path,
            device='cuda',
            model_path=args.model_path,
            root=args.kinetics_root,
            anno=args.kinetics_anno,
            class_labels=args.kinetics_class_labels,
        )
