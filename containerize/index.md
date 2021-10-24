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
```
docker build . . . TODO --network host
```

If you run into the follow error being unable to connect to a network:
```
```
You can give the docker build full network access as host.
```
docker build . . . TODO --network host
```

#### Install the Singularity Image

If you prefer to use singularity, this work ran the singularity container by creating it from the docker image.
You must obtain the Docker image as specified above either through downloading it from docker hub or building it from the source.
After you have the Docker image, you run the following command to obtain the singularity image:
```
singularity
```

### Run the Docker Image
To run the Docker container:
```
docker run --gpus all -it --rm -v local_dir:/tmp/docker/torch21 nvcr.io/nvidia/pytorch:21.09-py3
```

If you want to have access to a local directory while in the container for ease of accessing and saving files between the container instance and the actual machine use the following commmand:

Mounts approach:

### Run the Singularity Image
