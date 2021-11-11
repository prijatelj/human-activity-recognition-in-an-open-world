"""Feedback models, specifically CLIP label encoding."""
import numpy as np
import torch

import clip

from exputils.data.labels import NominalDataEncoder


class CLIPFeedback(object):
    """Feedback Interpretter using CLIP zeroshot.

    Attributes
    ----------
    clip : CLIP
        CLIP model used for encoding both text and images.
    preprocess :
        CLIP text preprocess.
    feedback_knowns : NominalDataEncoder
        Bidirectional map of feedback known texts to index. This indexing is
        consistent across feedback_label_encs and feedback_similarity rows.
        This serves as the unique known feedback labels.
    predictor_knowns : NominalDataEncoder
        Bidirectional map of predictor known texts to index. This indexing is
        consistent across predictor_label_encs and feedback_similarity columns.
        This serves as the unique known predictor labels.
    feedback_label_encs : torch.Tensor
        CLIP pre-processed label encodings of PAR.
    feedback_similarity : np.ndarray | torch.Tensor?
        A Matrix of rows being known feedback label texts and columns being the
        known predictor labels. The cosine similarity matrix is the cosine
        similarity of these labels to
    """
    def __init__(
        self,
        model_path,
        k600_label_encs,
        par_label_encs,
    ):
        # Load in CLIP Pre-trained for encoding new feedback label text
        self.clip, self.preprocess = clip.load(model_path, device)

        self.feedback_known = set()

        # TODO Cosine similarity matrix of (feedback labels) X (predictor
        # knowns) to one another when in CLIP encoded space.

        # TODO label maps for feedback label text to idx, and predictor label
        # text to idx

    def interpret_feedback(
        self,
        label_text,
        feature_repr=None,
        task_repr=None,
        videos=None,
    ):
    """Interpret the given feedback with the sample feature and task repr,
    returning a soft label vector per sample for the knowns + prob of unknown.

    Args
    ----
    label_text : list(list(str)) | np.ndarray(str)
        A matrix of text strings where rows are samples and columns are the
        number of classes given back sa feedback, which is assumed to be 5 due
        to protocol with PAR.
    feature_repr : torch.Tensor = None
        The feature representation of the input samples.
    task_repr : torch.Tensor = None
        The fine-tuning or task representation of the samples.
    videos : torch.Tensor
        The actual input videos, aligned to the label text, feature_repr, and
        task_repr, etc.

    Returns
    -------
    torch.Tensor
    """
    if isinstance(label_text, np.ndarray):
        np.array(label_text)

    # TODO Update running set of seen given labels. (OrderedDict keys)
    new_feedback_labels = np.unique(label_text)
    # TODO get the first occurrence index of the new feedback labels in
    # label_text to save the resulting text encoding.

    # TODO Sort each row of feedback labels lexically.

    # TODO CLIP Encode the given text labels with preserved structure for
    # getting indivudal feedback label encodings to be saved.
    label_encs = text_zeroshot_encoding(self.clip, label_text, self.templates)

    # TODO Obtain a CLIP encoding per row of feedback labels? Averaging?

    # TODO Obtain a Cosine Similarity of each Feedback Label to

    return interpretted_feedback
