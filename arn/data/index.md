## Fraud Detection: Data

This is general purpose code for loading and managing data for fraud detection such that the models may recieve the data in some standardized way.
If the models need to do specific modification to the data then their pipeline code handles that expecting this standardized input.

There will probably be a [dataset]_loader.py file per dataset and a single data_pipeline.py file that loads them.
