## Containerize with Docker and Singularity

This directory contains the configuration files to containerize this project using, first Docker, and then Singularity.
The Docker container depends on NVIDIA provided container.

### Install the Docker Image
Download the Docker image from Docker Hub or build the image from the source repository.

### Download from Docker Hub
```
docker pull . . . [Available after publication]
```

### Build from Source Repository
When your active director is the root of this repository, run the following command to build the docker container:
```
docker build -t arn:latest -f containerize/Dockerfile .
```

If you run into an error where you cannot connect to the internet, for example, downloading the git repositories or pip intsalling does not work,
you can give the docker build full network access as host.
```
docker build --network host -t arn:latest -f containerize/Dockerfile .
```

#### Install the Singularity Image

If you prefer to use singularity, this work ran the singularity container by creating it from the docker image.
You must obtain the Docker image as specified above either through downloading it from docker hub or building it from the source.
After you have the Docker image, you run the following command to obtain the singularity image named as `arn_latest.sif` using the docker-daemon:
```
sudo singularity build arn_latest.sif docker-daemon://arn:latest
```
Note that this may take a while.

### Run the Docker Image
To run the Docker container:
```
docker run --gpus all -it --rm nvcr.io/nvidia/arn:latest
```

If you want to have access to a local directory while in the container for ease of accessing and saving files between the container instance and the actual machine use the following commmand:

Mounts approach, where the data are located indirectories that are contained under this repository:
```
docker run \
    --mount type=bind,src=$HOME/absolute_path_to/arn/,dst=/tmp/docker/arn \
    --gpus all \
    -it \
    --rm \
    arn:latest
```

### Run the Singularity Image

```
sudo singularity run arn_latest.sif
```

Mounts approach:
The paper used mounts for the model weights and input data along with a mount for the output results, such as new model states, predictions, etc.
```
sudo singularity run --bind $HOME/absolute_path_to/arn:/tmp/arn arn_latest.sif
```
Once inside the container, you will need to move to that path `cd /tmp/arn`
