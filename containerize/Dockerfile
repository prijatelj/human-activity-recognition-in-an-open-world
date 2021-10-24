FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set a working directory for the app
WORKDIR /tmp/container_workspaces/docker/arn/

# Copy all the files to the container
COPY . .

# Intall dependencies
# install Prijatelj's public fork of `vast` for the EVM and FINCH
#pip install git+https://github.com/prijatelj/vast.git
RUN mkdir deps
WORKDIR deps

git clone --recurse-submodules https://github.com/prijatelj/vast
WORKDIR vast
python setup.py install

WORKDIR ../../

# Install CLIP
# pytorch, torchvision, and cudatoolkit deps are handled from NVIDIA container
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Install this repo's main package and its dependencies
python setup.py install

# Install this repo's experiments package and its dependencies
python experiments/setup.py install


# Weight Installation can be handled after basic build is finished.
# Should just give instuctions in README and tell user to mount the directory
# containing those weights when running the Docker container.
# TODO download CLIP weights
#python experiments/download_CLIP_weights.py

# TODO download X3D weights
#python experiments/download_X3D_weights.py

# TODO download TimeSformer weights
#python experiments/download_TimeSformer_weights.py
