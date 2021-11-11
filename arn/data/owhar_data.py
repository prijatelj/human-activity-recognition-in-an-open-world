"""Incremental Open World Learning."""


class OWHARData(object):
    """The Open World Human Activity Recognitoin Data

    datasets : list(KineticsUnified)
        A list of datasets where each index corresponds to a dataset used at
        that increment.
    dataloaders : list(DataLoader)
        A list of DataLoaders where each index corresponds to a dataset used at
        that increment and is aligned with `datasets`.
    """
    def __init__(self):
        raise NotImplementedError()

    # TODO?
    #def __getitem__(self):
    #    return

    def get_increment(self):
        raise NotImplementedError()

        # TODO For every increment up to this point, we need the training,
        # validation, and testing data.
        #   train -> input, labels, other annotations
        #   val -> input, labels, other annotations
        #   test -> input, labels, other annotations

        return


class OWHARExperiment(object):
    """The Open World Human Activity Recognition experiment pipeline including
    data and the predictor to use this data. This class contains the exectution
    or parts for data. This contains the code to perform the incremental open
    world learning experiment with training and evaluation.

    Attributes
    ----------
    train_data : OWHARData
        The data to be used for the training of the OWHAR experiment.
    val_data : OWHARData
        The data to be used for the OWHAR experiment.

        OR does OWHARData handle all of this?
    test_data : OWHARData
        The data to be used for the OWHAR experiment.
    predictor : OpenWorldHumanActivityRecognizer
        The predictor to be trained and evaluated throughout the open world
        learning process.
    evaluator :
        An object that manages the evaluation of the predictor at each
        increment. The results of those evaluation measurements are managed by
        this OWHARExperiment class, including saving to disk (Or should it be
        te Evaluator's job?).
    """
    def __init__(self):
        raise NotImplementedError()

    def increment_fit(self, index):
        """The fitting process for this increment."""
        raise NotImplementedError()

    def increment_eval(self, index, eval_data_id):
        """The evaluation process for this increment given eval data id.."""
        raise NotImplementedError()

        # TODO run data through the predictor
        #preds, detects = self.predictor.predict(data)
        # TODO Changes in predictor state need to be saved!
        #   This should be fine-tuning model states.
        #   This should be fine-tuning model states.

        # TODO update the measures
        #self.evalutor.measure_recog(data_classes, preds)
        #self.evalutor.measure_detect(novelty_data_labels, detects)

        # TODO evaluator needs to save measures recorded for this increment!

    def increment(self, index):
        """Perform the incremental learning as the increment indicated by index
        """
        raise NotImplementedError()

    #def __del__(self):
    #    """Deconstructor to handle closing and deletion of all objects nicely.
    #    """
    #    raise NotImplementedError()
