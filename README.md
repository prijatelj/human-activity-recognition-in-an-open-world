## Human Activity Recognition in the Open World

Human Activity Recognition (HAR) as a classification task is the task of mapping videos to their corresponding activity label.
Novelty and open world learning in this domain is akin to the open world learning versions of image classification or any traditional classification task.

- The published paper is available in [JAIR Vol. 81 (2024)](https://doi.org/10.1613/jair.1.14476)
- An updated arxiv version is also available at https://arxiv.org/abs/2212.12141 .

### Installation

This project's submission container uses Docker and Apptainer.
For our submission version we had to convert to Apptainer/Singularity due to our local computer cluster, and is named `huma-activity-recog-submission.sif`.
This submission version is hosted temporarily for [download on this google drive](https://drive.google.com/file/d/1wMySr7muXp33yB3QC-WKA6Mq4G7iWYtt/view?usp=sharing).

A future Docker container will be provided at Dockerhub or at another permanent hosting service.

#### Containerize Details

See `containerize/index.md` for instructions on building the Docker or Singularity container.

See `containerize/Dockerfile` for the details on how to install this package and its dependencies from source.
The Dockerfile is the build process.

### Reproducing the Experimental Results

After installation, the final model states may be downloaded and used in order to reproduce our evaluation results and to be used immediately.
These model states are available for download at: [TODO URL ONCE PUBLISHED]

#### Reproducing the Training and Experiments

To reproduce the training and experiments the initial pre-training model states need downloaded and the following sequence of commands are required to reproduce the experimental process followed in the paper.

[TODO pre-training model states links]

##### Experiment 1: Analysis of HAR Model Robustness to Nuissance Novelty

Experiment 1 examines the performance of three HAR models on Kinetics 400 versus Kinetics 400 with visually augmented images.
The models examined are: X3D, TimeSformer, and CLIP.
The three models are all using their provided Kinetics 400 weights, or their own pre-training in the case of CLIP.

The augmentations used includes the following 7:
- Blur
- Flip
- Invert Color
- Noise
- Perspective
- Rotation
- Color Jitter

These augmentations correspond to the code under `[arn/. . . TODO add location(s)]`.
Each augmentation type was applied to all images separately, and the performance of the pre-trained models on unaugmented images versus augmented images.
This experiment captures the performance difference resulting from each augmentation, which serves as a source of nuissance novelty to these pre-trained HAR models.

TODO to compute and save the augmentations do the following:

TODO to save the pre-trained models' predictions on the augmented data do the following:

TODO to to evaluate the predictions do the following:

##### Experiment 2: Benchmark Analysis of Open World HAR Predictors

TODO specify how to perform this experiment

###### Factor Analysis

With experiment 2 results saved, the performance of the different predictors at each step of Open World HAR learning may be obsereved, especially with regards to when different types of novelty occurs.
This section performs the in-depth analysis that breaks down how the predictors perform on Open World HAR, Novelty Detection, and Novelty Recognition when encountering specific types of novelty.

TODO specify how to perform this analysis


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

### License

The license for this repository is incldued in `LICENSE.txt`.

### Citation

If you use our work, please cite us using the following bibtex:
```
@article{prijateljHumanActivityRecognition2024,
	title = {Human {Activity} {Recognition} in an {Open} {World}},
	volume = {81},
	copyright = {Copyright (c) 2024 Journal of Artificial Intelligence Research},
	issn = {1076-9757},
	url = {https://www.jair.org/index.php/jair/article/view/14476},
	doi = {10.1613/jair.1.14476},
	language = {en},
	journal = {Journal of Artificial Intelligence Research},
	author = {Prijatelj, Derek S. and Grieggs, Samuel and Huang, Jin and Du, Dawei and Shringi, Ameya and Funk, Christopher and Kaufman, Adam and Robertson, Eric and Scheirer, Walter J.},
	month = dec,
	year = {2024},
	keywords = {neural networks, machine learning, perception, reasoning about actions and change},
	pages = {935--971},
}
```

### Acknowledgements

This research was sponsored in part by the National Science Foundation (NSF) grant CAREER-
1942151 and by the Defense Advanced Research Projects Agency (DARPA) and the Army Re-
search Office (ARO) under multiple contracts/agreements including HR001120C0055, W911NF-
20-2-0005, W911NF-20-2-0004, HQ0034-19-D-0001, W911NF2020009. The views contained in
this document are those of the authors and should not be interpreted as representing the official
policies, either expressed or implied, of the DARPA or ARO, or the U.S. Government.
