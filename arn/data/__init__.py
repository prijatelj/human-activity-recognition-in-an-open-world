from importlib import import_module

__all__ = [
    'kinetics',
    'kinetics_combined',
    'par_data',
    'dataloader_utils',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', 'arn.data')
