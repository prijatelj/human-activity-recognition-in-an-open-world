"""Any torch utils used across multiple files for predictors."""
import torch

from arn.data.kinetics_unified import KineticsUnifiedFeatures


def torch_dtype(dtype):
    """Given dtype as str or torch dtype, perform the casting, and checks."""
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
        if isinstance(dtype, torch.dtype):
            return dtype
        else:
            raise TypeError('Expected torch.dtype for dtype not: {dtype}')
    else:
        raise TypeError('Expected torch.dtype for dtype not: {dtype}')


def get_kinetics_uni_dataloader(dataset, *args, **kwargs):
    """Get torch DataLoader of a KineticsUnifiedFeatures (subclass) dataset."""
    if isinstance(dataset, KineticsUnifiedFeatures):
        return torch.utils.data.DataLoader(dataset, *args, **kwargs)
    elif not isinstance(dataset, torch.utils.data.dataloader):
        raise TypeError(f'Unexpected dataset type: `{type(dataset)}`')
    return dataset
