"""Feedback models, specifically CLIP label encoding."""
import torch

import clip


class CLIPFeedback(object):
    """Feedback Interpretter using CLIP zeroshot.

    Attributes
    ----------
    clip : CLIP
        CLIP model used for encoding both text and images.
    preprocess :
        CLIP text preprocess.
    k600_label_encs : torch.Tensor
        CLIP pre-processed label encodings of Kinetics600
    par_label_encs : torch.Tensor
        CLIP pre-processed label encodings of PAR.
    """
    def __init__(
        self,
        model_path,
        k600_label_encs,
        par_label_encs,
    ):
        # Load in CLIP Pre-trained
        self.clip, self.preprocess = clip.load(model_path, device)

        # NOTE this may be the only part of the Predictor that does not need be
        # a torch.nn.Module, but in the end


    def interpret_feedback(
        self,
        class_text,
        feature_repr=None,
        task_repr=None,
        videos=None,
    ):
    """Interpret the given feedback with the sample feature and task repr,
    returning a soft label vector per sample for the knowns + prob of unknown.

    Args
    ----
    class_text : list(list(str)) | np.ndarray(str) | pd.DataFrame
        A matrix of text strings where rows are samples and columns are the
        number of classes given back sa feedback, which is assumed to be 5 due
        to protocol with PAR.
    feature_repr : torch.Tensor = None
        The feature representation of the input samples.
    task_repr : torch.Tensor = None
        The fine-tuning or task representation of the samples.
    videos : torch.Tensor
        The actual input videos, aligned to the class text, feature_repr, and
        task_repr, etc.

    Returns
    -------
    torch.Tensor
    """

    return
