"""Any torch utils used across multiple files for predictors."""
import torch

def torch_dtype(dtype):
    """Given dtype as str or torch dtype, perform the casting, and checks."""
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
        if isinstance(dtype, torch.dtype):
            return dtype
        else:
            raise TypeError('Expected torch.dtype for dtype not: {dtype}')
    else:
        raise TypeError('Expected torch.dtype for dtype not: {dtype}')
