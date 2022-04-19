from importlib import import_module
__version__='0.1.0'

__all__ = [
    'data',
    'models',
    'transforms',
    'torch_utils',
    #'visuals',
    #'utils',
    # TODO really should rely on external install for X3D and TimeSformer if we
    # can so we don't carry their code in the repo, to be shown in containerize
    #'timesformer',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', __name__)
del import_module, module
