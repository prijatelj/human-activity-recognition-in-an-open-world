## Experiment 1

All scripts pertaining to running and reproducing the paper's experiment 1 are located here.
Experiment 1 consists of assessing the performance of known classifiers for video Human Activity Recognition on the Kinetics400 dataset.
The code includes:
- scripts for obtaining the feature encdoings from the different feature representation models, which are the logit layers of the Kinetics400 classifiers before the linear-softmax layer for classification.
- the configuration files for running the Kinetics400 experiments.
    - trains the fine tune model on the given feature representation which serves as the starting incrment for the open world recognition task across the Kinetics datasets.
    - evaluated the trained models on the augmented videos which assess the feature representation models by themselves as well as the different fine tune models.
        - This depicts how the state-of-the-art classifiers perform on unseen augmentations to the videos.
