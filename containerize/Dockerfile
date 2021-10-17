FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set a working directory for the app
WORKDIR /tmp/container_workspaces/docker/arn/

# Copy all the files to the container
COPY . .

# TODO intall dependencies

# TODO install Prijatelj's public fork of `vast` for the EVM and FINCH
pip install git+https://github.com/prijatelj/vast.git

# TODO install CLIP (Necessary or just add to this repo?)

# Install this repo's main package and its dependencies
python setup.py install

# Install this repo's experiments package and its dependencies
python experiments/setup.py install
