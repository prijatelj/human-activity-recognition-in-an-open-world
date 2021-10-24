from importlib import import_module
__all__ = [
    'generics',
    'augmentation',
    'feature_extraction',
    'owhar',
    # Add new files to be imported here
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', 'arn.models')
