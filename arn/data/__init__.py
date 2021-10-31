from importlib import import_module

__all__ = [
    'kinetics',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', 'arn.data')
