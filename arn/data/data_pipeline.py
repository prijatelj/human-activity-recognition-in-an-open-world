"""The data pipeline code. Given the dataset info and id, selects the proper
dataloader and loads the data from it in a standardized format for the models.
"""

# TODO load data from dataloader(s)

# TODO optionally save data, in case of changes

# TODO DataLoader abstraction
"""
Every sample is the following:
   NamedTuple(
        'sample_path'
        'video'/'image'
        'labels'
        'aug'
    )
    sample_path : str
        The path to the sample data, whether that is image or video.
    image/video : torch.Tensor
        The actual image of a frame from the video OR the video
    labels : dict | None
        A dict of the name of the label set to its specific label value,
        otherwise none if no labels available. The label sets' label values
        will be None if no label value of this label set available for that
        sample.
        Possible label sets:
            - par_activity
            - k400_activity
            - k700_activity
            - perspective
            - location
            - relation1
            - relation2
            - ava
        `ava` corresponds to the annotations from KineticsAVA to provide more
        information on the features in that video for factor analysis.
    aug : OrderedDict
        The name of the augmentation applied to a dict of parameter names to
        their values, which is the **kwargs able to be used in the original
        function to get the exact augmentation.
    feedback: list
        In PAR's feedback, a list of the top 5 closest Kinetics700 classes are
        given, where we are uncertain if order matters to tell which is
        closest.  We intend to also check when the correct class if given back
        and when given top [3, 5, 10] closest PAR or Kinetics700 classes are
        given back.
"""

# NOTE, it may be better to have every thing be accessible in the above way,
# but still be sliceable at each point.
