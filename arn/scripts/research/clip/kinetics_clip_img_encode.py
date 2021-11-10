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
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode
import tqdm

import clip
from arn.data.kinetics import Kinetics
from arn.transforms.target_transforms import ClassLabel
from arn.data.kinetics import Kinetics
from arn.data.kinetics_unified import Kinetics_Unified

import exputils


def clip_transform_image_frames(
    n_px,
    means=(0.48145466, 0.4578275, 0.40821073),
    stds=(0.26862954, 0.26130258, 0.27577711),
):
    """Transforms video frames from Kinetics output to CLIP expectation."""
    return Compose([
        Resize(n_px, interpolation=InterpolationMode('bicubic')),
        CenterCrop(n_px),
        ToTensor(),
        Normalize(means, stds),
    ])


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
    root,
    anno,
    class_labels,
    image_path=None,
    label_path=None,
    pred_path=None,
    device='cuda',
    model_path='ViT-B/32',
    load_encoded_labels=True,
    load_encoded_images=False,
    model_repr_dim=512,
    templates=None,
    batch_size=1,
    gamma_tau=1,
    crops=1,
    frames=300,
    unified_split=None,
    feature_extract_outpath=None,
    start_idx=None,
    end_idx=None,
):
    if image_path is None and label_path is None and pred_path is None:
        raise ValueError(
            '`image_path`, `label_path`, and `pred_path` are all None',
        )

    # Load in CLIP Pre-trained
    model, preprocess = clip.load(model_path, device)

    # Preprocess images: Go from Kinetics form to CLIP preprocess expectation
    # 224 input_resolution (based on clip_model.visual.input_resolution
    # CLIP process is to resize the PIL Image to this resolution using BICUBIC
    # interpolation, then CenterCrop, convert to RGB, and Normalize
    # Kinetics dataloader outputs clips as (Channels, Timeframe, Height, Width)
    # CLIP expects them as (B, C, H, W), so B for T
    spatial = clip_transform_image_frames(
        model.visual.input_resolution,
        means=[110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255],
        stds=[38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255],
    )

    # TODO make it easier to batch this process cuz it can be a pain when not
    # enough VRAM.
    #if start_idx:
    #    val_dataset.data
    #if end_idx:
    #    val_dataset = val_dataset[:]

    # Get Kinetics Dataloader for specified data
    if unified_split is None:
        # Use old single kinetics dataset loader for 400 and 600
        dataset = Kinetics(
            root,
            anno,
            class_labels,
            'val',
            spatial_transform = spatial,
            target_transform = ClassLabel(),
            sample_duration=frames,
            gamma_tau=gamma_tau,
            crops=crops,
            randomize_spatial_params=False,
        )
    elif unified_split.lower() == 'par': # Use PAR dataloader
        raise NotImplementedError()
    elif isinstance(unified_split, str): # Use unified kinetics dataloader
        dataset = Kinetics_Unified(
            root,
            anno,
            class_labels,
            unified_split,
            spatial_transform = spatial,
            target_transform = ClassLabel(),
            sample_duration=frames,
            gamma_tau=gamma_tau,
            crops=crops,
            randomize_spatial_params=False,
            outpath=feature_extract_outpath,
        )
    else:
        raise TypeError(
            'Unexpected type for `unified_split`: {type(unified_split)}',
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    print('dataset samples:', len(dataset))
    print('datasets created')

    # TODO the CLIP Zeroshot and feature repr needs to be a module used by
    # baseline. Save or reuse whatever boilerplate is around CLIP encoding here

    if load_encoded_labels and label_path:
        encoded_labels = torch.load(label_path).type(torch.float32)
    else:
        encoded_labels = None

    if load_encoded_images and image_path:
        encoded_images = torch.load(image_path)
        video_paths = None
    else:
        encoded_images = None
        video_paths = []

    preds = None

    # Calculate & save, or load the label text CLIP features
    if encoded_labels is None and (label_path or pred_path):
        # Get the unique text labels and sort them
        label_texts = sorted({d['label'] for d in dataset.data})

        # Encode the Labels
        encoded_labels = text_zeroshot_encoding(
            model,
            label_texts,
            templates,
        ).type(torch.float32)

        # Save the encoded text labels.
        torch.save(encoded_labels, exputils.io.create_filepath(label_path))

    # Encode the images, optionally get preds
    with torch.no_grad():
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

        # Prealocate encoded images, memory efficient
        if encoded_images is None and (image_path or pred_path):
            encoded_images = torch.empty([
                len(bar),
                frames,
                model_repr_dim,
            ])

            # Get the image encodings
            for i, (images, labels) in bar:
                if image_path or pred_path:
                    # Make video frames Batch, Time, Channels, Height, Width,
                    # again. Just transpose the two dims:
                    image_encs = model.encode_image(
                        images.squeeze().transpose(0,1).cuda()
                    )

                    # The following is for batch size greater than 1:
                    #images = images.transpose(0, 2, 1, 2, 3)
                    #shape = images.shape
                    # Flatten batch and time
                    # Encode images and Reconstruct Batch and Time
                    #image_encs = model.encode_image(
                    #    images.flatten(0, 1),
                    #).reshape(shape)

                    video_paths.append(dataset.data[i]['video'])

                    if image_path:
                        # Store the encoded images
                        encoded_images[i] = image_encs
        elif not isinstance(encoded_images, torch.Tensor) and (pred_path):
            # No encoded images despite pred path existing (preds to get)
            raise TypeError(' '.join([
                'No encoded images despite predictions to get and save at',
                'pred_path',
            ]))

        if image_path and video_paths is not None:
            # Save the encoded images
            torch.save(encoded_images, exputils.io.create_filepath(image_path))
            with open(
                exputils.io.create_filepath(
                    f'{os.path.splitext(image_path)[0]}_video_paths.txt',
                ),
                'w',
            ) as openf:
                openf.write('\n'.join(video_paths))

        if pred_path:
            # Calculate Zero-Shot predictions (Cosine Similarity * 100)
            # CLIP paper states the mean of the predictions was used for
            # K700
            shape = encoded_images.shape

            # Flatten samples and frames into one dim, but save original shape
            encoded_images = encoded_images.flatten(0, 1).to('cuda')

            # Calculate the label similarity per frame thru softmax
            # The similarity they used is essentially the "unnormalized"
            # Cosine Similarity and is proportional to Cosine Similrity.
            # Simply mat mul the 2 vectors because if you are taking the
            # max then the normalization is irrelevant as a scaling factor.
            preds = (
                100.0
                * (
                    encoded_images / encoded_images.norm(dim=-1, keepdim=True)
                ) @ (
                    encoded_labels
                    / encoded_labels.norm(dim=-1, keepdim=True)
                ).T
            ).softmax(dim=-1).reshape(
                [shape[0], shape[1], encoded_labels.shape[0]],
            )

    if pred_path:
        # Save the encoded images
        torch.save(preds, exputils.io.create_filepath(pred_path))


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
    parser.add_argument(
        '--load_encoded_images',
        action='store_true',
        help='If given, expects image_path is a Tensor of image encodings.',
    )
    parser.add_argument(
        '--encode_labels',
        action='store_false',
        help='If given, expects label_path is a Tensor of image encodings.',
        dest='load_encoded_labels',
    )

    parser.add_argument(
        '--unified_split',
        default=None,
        help='If given, the dataloader is Kinetics Unified, unless "par".',
    )

    # TODO for partial/batch completion of this process when VRAM not enough
    parser.add_argument(
        '--start_idx',
        default=None,
        type=int,
        help='The inclusive index of data to start process at',
    )
    parser.add_argument(
        '--end_idx',
        default=None,
        type=int,
        help='The exclusive index of data to end process at',
    )

    args = parser.parse_args()

    # TODO What is this for and why???? X3D artifiact?

    # TODO Pay attention to main param defaults and set shuffle to False
    main(
        image_path=args.image_path,
        label_path=args.label_path,
        pred_path=args.pred_path,
        device='cuda',
        model_path=args.model_path,
        root=args.kinetics_root,
        anno=args.kinetics_anno,
        class_labels=args.kinetics_class_labels,
        load_encoded_labels=args.load_encoded_labels,
        load_encoded_images=args.load_encoded_images,
        unified_split=args.unified_split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
