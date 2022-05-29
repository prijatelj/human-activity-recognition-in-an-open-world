## Simulated Experiments

This directory contains
all scripts that simulate both experiment 1 and 2 by generating points in a 2 dimensional space that are assigned to separable classes.
The classes' distributions are all 2D Gaussian distributions.
The generated points are saved as formatted for `arn.data.kinetics_unified.KineticsUnifiedFeatures` for test running the pipeline and aiding in debugging.
The 2D points serve as the feature represention of samples of each class.

### Simulated Experiment 1

The starting increment and experiment 1 are simulated using 4 2d Gaussians placed on the unit cirlce in the cardinal directions.
These 4 Gaussians each represent their own class.
Each Gaussian has a standard deviation of 1/10.
Each have 1k samples each for each data split of train, val, and test.
The predictors perform the classification after being trained on the data, as per usual closed-set classification.

### Simulated Experiment 2

The simulated data for experiment 2 matches that of the paper's experiments on the Kinetics data.
There are 2 sets of Gaussians to stand in for the Kinetics600 and Kinetics700-2020 datasets, where the pair of sets consist of 10 classes (Gaussians) each and where one new class is added in each increment (10 increments per dataset).
All Gaussians in experiment 2 have a standard deviation of 1/10.

The first 10 increments consists of adding a singular new class per increment.
The number of sampled points per known class at each increment is 100 per class.
The new class introduced in its respective increment (i) for the dataset is samples `i*100`, where i is an integer in the inclusive range [1,10].
This means at the end of each dataset, every new class will have 1k samples.
This pattern is repeated for each of the 2 datasets.

The first dataset of Gaussians are all located on the unit circle in between the existing 4 class' distributions.

The second dataset of Gaussians are one the 2 times the unit circle's radius and located at the cardinal 8 cardinal directions.
The final two classes are on a circle with 3 times the unit circle's radius, one at North and one in between the angles of the 5th and 11th class (see the code or visuals to understand its location).
