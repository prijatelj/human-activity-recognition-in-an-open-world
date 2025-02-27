from importlib import import_module
__version__='0.2.0rc1'

__all__ = [
    'data',
    'models',
    'transforms',
    'torch_utils',
    'scripts',
    #'visuals',
    #'utils',
    # TODO really should rely on external install for X3D and TimeSformer if we
    # can so we don't carry their code in the repo, to be shown in containerize
    #'timesformer',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', __name__)
del import_module, module
