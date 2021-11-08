"""Helper functions for getting the dataloaders"""
import torch

from arn.data.kinetics import Kinetics
from arn.transforms.target_transforms import ClassLabel


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


    # TODO make it easier to batch this process cuz it can be a pain when not
    # enough VRAM.
    #if start_idx:
    #    val_dataset.data
    #if end_idx:
    #    val_dataset = val_dataset[:]

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


#TODO def get_unified_kinetics_dataloader():


def get_par_dataloader():
    return
