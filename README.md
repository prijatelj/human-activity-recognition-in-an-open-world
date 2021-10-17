## Formalization and Analysis of Human Activity Recognition with Novelty

Human Activity Recognition (HAR) as a classification task is the task of mapping videos to their corresponding activity label.
Novelty and open world learning in this domain is akin to the open world learning versions of image classification or any traditional classification task.
The original paper is available at [TODO URL].

### Installation

This project is containerized using Docker and by extension, Singularity.
The Docker image is available at Docker Hub and installable via:
```
docker  . . . [TODO: INSTALL FROM DOCKER HUB URL ONCE PUBLISHED]
```

To build this package with Docker, simply run
```
docker build . . . TODO
```

The original paper's research was run in a Singularity container by converting the above Docker image into a Singularity container using the following process.
```
singularity . . . TODO
```

See `containerize/Dockerfile` for the details on how to install this package and its dependencies from source.

### Reproducing the Experimental Results

After installation, the final model states may be downloaded and used in order to reproduce our evaluation results and to be used immediately.
These model states are available for download at: [TODO URL ONCE PUBLISHED]

#### Reproducing the Training and Experiments

To reproduce the training and experiments the initial pre-training model states need downloaded and the following sequence of commands are required to reproduce the experimental process followed in the paper.

TODO pre-training model states links

##### Experiment 1

TODO

##### Experiment 2

TODO

##### Experiment 3

TODO


### License

The license for this repository is incldued in `LICENSE.txt`.

### Citation

If you use our work, please cite us using the following bibtex:
```
TODO ONCE PUBLISHED
```


### Acknowledgements

This project was funded by . . .

Thank you to . . .


### Activity Recognition Label Specifity

Label specificity means how specific and detailed the labels are to the actual given data to provide more information related to the task.
AR can have more specific labels, just as image classification tasks may have, e.g. object detection/recognitoin with 3D objects overlayed the 2d image providing at least the 3d info mapped to the pixels in the 2d image, if not providing the distance from the camera of that image.
AR label specificty can be as follows:

- Single class label per video
    - seems to be known in literature as simply Activity Recognition
    - Note some datasets focus specifically on human-based AR
    - Datasets:
- Single class label per frame in video
    - Knownin literature as ???
    - Datasets:
- Multiple classes per frame in video w/ Bounding Boxes
    - Knownin literature as ???
    - Datasets:
- Fully simulated / near perfectly specified labels
    - All 2D frames have their corresponding 3D mapping.
        Each object / area is mapped with a task relation by some probability [0, 1], i.e. how much that portion of 3D space pertains to a task.
    - Knownin literature as ???
    - Datasets:

Note that AR relates to Next-Frame Prediction.
PredNet (cite?) is a model known to perform the task of Next-Frame Prediction.

### Repository Directory Structure

Each directory will have an index.md file with information about the contents of that directory.

+ setup.py
    - basic template to be filled out for the package that will be worked on
+ Dockerfile
    - at least an example of how to setup a docker file
+ requirements.txt
    - any base requirements
+ tests/
    - index.md: basic test structure goes here for convenience (e.g. pytest)
+ arn/
    - Package place holder. This contains common abstract classes.
+ experiments/
    - index.md: Where examples and experiments using the package will be stored
    - data/
        + abstract data class that indicates typical ways to load a dataset.
        + all data related code goes here, like preprocessing etc.
    - research/
        + The research experiments that use the package
