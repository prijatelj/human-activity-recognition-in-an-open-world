## Experiments

Examples and reproducible experiments are located here for performing research.
The scripts here use the package(s) provided in this repo and depict their use.

### Future Warning

In the future, experiments may be moved to inside the package by default and a flag added to setup.py to allow optional installation of the experiments package.
As it is setup now, two packages are needed to be installed: the repo's actual package(s) and the experiments package.

For the above stated functionality and to have it such that the experiments code is not installed without providing an arg to specify its installation, such as extras_require in setuptools, the experiments would be a 2nd package anyway.
Due to experiments must be a separate package either way, it remains external.
If experiments is desired to be separated from the main project repo, this may occur, but for simplicity the template will keep them togther.
