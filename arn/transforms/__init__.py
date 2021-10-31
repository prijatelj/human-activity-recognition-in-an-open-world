from importlib import import_module

__all__ = [
    'spatial_transforms',
    'temporal_transforms',
    'target_transforms',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', 'arn.transforms')
