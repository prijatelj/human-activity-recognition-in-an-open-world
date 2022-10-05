from importlib import import_module

__all__ = [
    'docstr_workarounds',
    'dataloader_utils',
    'kinetics_unified',
    'kinetics_owl',
    'par',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', __name__)
del import_module, module
