"""Trains the downstream models on the pre-computed features of some model.
This includes the X3D and TimeSformer models as the feature representations.
The downstream models include fine-tuning as a block of dense, fully connected
layers or the Extreme Value Machine. This is for experiment 1, train on the
original Kinetics 400 videos (feature extracted in this case) and evaluate on
the original validation and test datasets along with augmentaitons to show the
difference in performance given nuisance, reprsentation novelty (never befor
seen augmented versions of the images).
"""

#@docstr.meta.splat_expansion
def train(**kwargs):
    """
    Args
    ----
    dataloader : KineticsUnifiedFeatures
    predictor : OWHARPredictor
        An OWHARPredictor whose model after feature representation is to be
        trained. Checkpoint saving may be specified through OWHARPredictor's
        initialization.
    save_model : str
        Filesystem location to save the resulting trained model.
    """
    raise NotImplementedError()

def eval():
    # TODO perhaps different file.
    raise NotImplementedError()
