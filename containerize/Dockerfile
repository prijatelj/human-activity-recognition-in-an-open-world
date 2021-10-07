FROM nvcr.io/nvidia/pytorch:19.02-py3

# Set a working directory for the app
WORKDIR /tmp/container_workspaces/docker/arn/

# Copy all the files to the container
COPY . .

# TODO create the virtual environment (Conda?)

# TODO intall dependencies

# TODO install EVM
pip install timm



# TODO install X3D


# TODO install CLIP

# TODO install other_model


# Install this repo's main package and its dependencies
python setup.py install

# Install this repo's experiments package and its dependencies
python experiments/setup.py install
