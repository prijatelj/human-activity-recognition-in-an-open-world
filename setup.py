from setuptools import setup

long_description = ''
with open('README.md', 'r') as f:
    long_description = f.read()

project_name = 'arn'

install_requires = ''
with open('requirements/{project_name}.txt', 'r') as f:
    install_requires = f.read()

setup(
    name=project_name,
    version=get_property('__version__', project_name),
    author='Derek S. Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[
        project_name,
        f'{project_name}.data',
        f'{project_name}.models',
        #f'{project_name}.visuals',
    ],
    #scripts
    description=' '.join(
        'Human Activity Recognition with novelty through machine learning',
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/prijatelj/{project_name}',
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    #tests_require=['pytest'],
    #setup_requires=['pylint', 'pytest-runner'],
)
