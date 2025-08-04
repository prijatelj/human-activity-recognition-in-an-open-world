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

The paper provides the important details for reproduction, and this repository provides the specific code and scripts.

The Kinetics 400, 600, 700-2020 videos can be downloaded from [https://github.com/cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).
See the paper for details on this and other sources.

After installation, the final model states may be downloaded and used in order to reproduce our evaluation results and to be used immediately.
These model states are available for download at:

- X3D
    + [Features](https://drive.google.com/file/d/1Ky4_GY0D3XrltoxQ1s8FdTspeHUHzFbN/view?usp=sharing)
    + [Model Weights](https://drive.google.com/file/d/1gYoGujPB4rOUl9r6qM-CCTK_69Qbm1VJ/view?usp=sharing)
- TimeSformer
    + [Feature Extraction](https://drive.google.com/file/d/1hKBXOeC_86N_n3hn5Cb8e9i8TMIbymGS/view?usp=sharing)
    + [Model Weights](https://drive.google.com/file/d/1JVenNIqPOsOZCpCzEMf95k-5SJA_CVrp/view?usp=sharing)


#### Repository Directory Structure

The primary directories are:
`tree -d 1 ./arn/`
arn/
├── data
├── models
│   └── novelty_recog
├── scripts
└── transforms

#### Reproducing the Training and Experiments

The directory at `./arn/scripts/` contains the scripts used to manage the data, perform the experiments, and visualize the results.

`tree -d 2 arn/scripts/`
arn/scripts/
├── data_management
│   ├── clip
│   ├── crc
│   │   └── par
│   └── par
├── exp1
│   ├── configs
│   └── visuals
├── exp2
│   ├── configs
│   │   └── visual_transforms
│   ├── crc
│   │   └── met10
│   └── visuals
├── sim_open_world_recog
│   ├── configs
│   ├── crc
│   └── visuals
└── visuals

The configuration files for each experiment and predictor pair are contained in `configs/`.
These configs are used by the CLI tool `docstr` and examples of scripts used to run these experiments are contained within the `crc/` directories.
`docstr` loads the configs as the defaults, which then may be overridden using CLI flags and specifying the values, as seen in the various scripts.
There are scripts for visuals per experiment and in general.

To reproduce the training and experiments the initial pre-training model states need downloaded and the following sequence of commands are required to reproduce the experimental process followed in the paper.


##### Experiment 1: Analysis of HAR Model Robustness to Nuisance Novelty

This is the experiment in Section 5.4 and Appendix C's Table 3 and Figure 16. of the paper.
This corresponds to directory `./arn/scripts/exp1/`.
Please refer to the paper for an overview of the experiment and key details.

Experiment 1 examines the performance of three HAR models on Kinetics 400 versus Kinetics 400 with visually augmented images.
The models examined are: X3D and TimeSformer.
The models all use their provided Kinetics 400 weights.

The augmentations used includes the following 7:
- Blur
- Flip
- Invert Color
- Noise
- Perspective
- Rotation
- Color Jitter

These augmentations correspond to the code under `./arn/augment_videos.py` and `./arn/transforms/`.
Each augmentation type was applied to all images separately, and the performance of the pre-trained models on unaugmented images versus augmented images was recorded.
This experiment captures the performance difference resulting from each augmentation, which serves as a source of nuisance novelty to these pre-trained HAR models.

The augmentations were applied to each video and saved.

The pre-trained models' predictions on the augmented data were then recorded and evaluated following the crc scripts and configs.

##### Experiment 2: Benchmark Analysis of Open World HAR Predictors

This is the primary KOWL experiment in Section 5.2 of the paper and corresponds to directory `./arn/scripts/exp2/`.
Please refer to the paper for an overview of the experiment and key details.

`./arn/scripts/exp2/configs/` contains the default configurations, which are fairly exhaustive.
`./arn/scripts/exp2/crc/` contains the CRC scripts used to execute the experiments per predictor.
You can apply flags to change the configuration of the experiment or predictor for that experiment as desired.

###### Factor Analysis

With experiment 2 results saved, the performance of the different predictors at each step of Open World HAR learning may be observed, especially with regards to when different types of novelty occurs.
This section performs the in-depth analysis that breaks down how the predictors perform on Open World HAR, Novelty Detection, and Novelty Recognition when encountering specific types of novelty.

### License

The MIT license for this repository is included in `LICENSE.txt`.

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
