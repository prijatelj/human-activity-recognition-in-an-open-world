from importlib import import_module
__all__ = [
    # Add new files to be imported here
    'generics',
    'augmentation',
    'feature_extraction',
    'owhar',
    'feedback',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', __name__)
del import_module, module
