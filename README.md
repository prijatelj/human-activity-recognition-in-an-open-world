## Open Set/World Recogntion: Fraud Detection

Fraud detection in this case is the task of detecting a change in the distribution.
The dataset is specifically online reviews such as Amazon and the objective is to determine when there is a change in the distirbution of known writer styles such that it may be determined when a change in writing occurs.

### Structure

Each directory will have an index.md file to serve as a starting point.

+ setup.py
    - basic template to be filled out for the package that will be worked on
+ dockerfile
    - at least an example of how to setup a docker file
+ requirements.txt
    - any base requirements
+ tests/
    - index.md: basic test structure goes here for convenience (e.g. pytest)
+ fraud_detection/
    - Package place holder. This contains common abstract classes.
+ experiments/
    - index.md: Where examples and experiments using the package will be stored
    - data/
        + abstract data class that indicates typical ways to load a dataset.
        + all data related code goes here, like preprocessing etc.
    - research/
        + The research experiments that use the package
