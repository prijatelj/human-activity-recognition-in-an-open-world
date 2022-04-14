# ARN package Setup.py
from setuptools import setup, find_packages
import re

def get_property(prop, project):
    """Gets the given property by name in the project's first init file."""
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + '/__init__.py').read()
    )
    return result.group(1)

long_description = ''
with open('README.md', 'r') as f:
    long_description = f.read()

project_name = 'arn'

install_requires = ''
with open(f'requirements/{project_name}_docker.txt', 'r') as f:
    install_requires = f.read()

# TODO if possible, install the python deps done in Dockerfile here so it can
# just be pip installed: vast, CLIP, X3D, TimeSformer. This will simplify the
# Dockerfile and make it way easier to install this in general via pip or conda

setup(
    name=project_name,
    version=get_property('__version__', project_name),
    author='Derek S. Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[f'{project_name}.{pkg}' for pkg in find_packages(project_name)],
    description=' '.join(
        'Human Activity Recognition with novelty through machine learning',
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/prijatelj/{project_name}',
    install_requires=install_requires,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Add the Script Interfce that provides `packagename submodule` similart to
    # `git pull` or `git push` commands for git and unify them under one alias
    # TODO implement this ofc, but also consider making it optional?
    #scripts
    #entry_points={
    #   `arn=arn.scripts:script_cli`
    #},
)
