## Formalization and Analysis of Human Activity Recognition with Novelty

Human Activity Recognition (HAR) as a classification task is the task of mapping videos to their corresponding activity label.
Novelty and open set/world learning in this domain is akin to image classification or any traditional classification task.

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

### Structure

Each directory will have an index.md file to serve as a starting point.

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
