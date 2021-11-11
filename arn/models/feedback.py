"""Feedback models, specifically CLIP label encoding."""
import numpy as np
import torch

import clip

from exputils.data.labels import NominalDataEncoder


class CLIPFeedbackInterpreter(object):
    """Feedback Interpretter using CLIP zeroshot.

    Attributes
    ----------
    clip : CLIP
        CLIP model used for encoding both text and images.
    feedback_known_map : NominalDataEncoder
        Bidirectional map of feedback known texts to index. This indexing is
        consistent across feedback_label_encs and similarity rows.
        This serves as the unique known feedback labels.
    pred_known_map : NominalDataEncoder
        Bidirectional map of predictor known texts to index. This indexing is
        consistent across predictor_label_encs and similarity columns.
        This serves as the unique known predictor labels.
    feedback_label_encs : torch.Tensor
        CLIP processed label encodings of feedback label text.
    pred_label_encs : torch.Tensor
        CLIP processed label encodings of predictor's known label text.
    similarity : np.ndarray | torch.Tensor?
        A Matrix of rows being known feedback label texts and columns being the
        known predictor labels. This matrix's elements are the unnormalized
        cosine similarities of the feedback labels to predictor labels. This is
        unnormalized because the normalization process is dependent upon the
        number of predictor classes and better to preserve the raw cosine
        similarities and normalize them on demand, rather than have to
        re-normalize without introducing error.
    """
    def __init__(
        self,
        clip_path,
        pred_labels,
        pred_label_encs=None,
        feedback_labels=None,
        feedback_label_encs=None,
        similarity=None,
    ):
        """Initialize and load the CLIPFeedbackInterpreter
        Args
        ----
        clip_path : str
            Path to the clip model
        pred_labels : str | list(str) | NominalDataEncoder
            Path to the predictor's known labels or the known labels themselves.
        pred_label_encs : torch.Tensor = None,
            Path to the predictor's known label encodings or the encoding
            themselves.
        feedback_labels : str | list(str) | NominalDataEncoder = None
            Path to the known feedback labels or the labels themselves.
        feedback_label_encs : torch.Tensor = None
            Path to the known feedback label encodings or the encoding
            themselves.
        similarity : str | torch.Tensor =None
            Path to the similarity martrix to be loaded or the similarity
            matrix.
        """
        # Load in CLIP Pre-trained for encoding new feedback label text
        self.clip, self.preprocess = clip.load(model_path, device)

        self.feedback_known = set()

        # TODO Cosine similarity matrix of (feedback labels) X (predictor
        # knowns) to one another when in CLIP encoded space.

        # TODO label maps for feedback label text to idx, and predictor label
        # text to idx

    def clip_encode_text(self, label_text):
        """Return the clip encoding of the label text preserving shape.
        Args
        ----
        label_text : list(list(str)) | np.ndarray(str)
            A matrix of text strings where rows are samples and columns are the
            number of classes given back sa feedback, which is assumed to be 5
            due to protocol with PAR.

        Returns
        -------
        torch.Tensor
            A Tensor of the label encodings with the CLIP model such that the
            first 2 dimenions match that of the input `label_text` matrix with
            the 3rd dimension being the dimension of the clip encoding.
        """
        raise NotImplementedError('Needs updated for predictor knowns'.)
        # TODO this does not need to preserve shape! This can simply encode the
        # given text labels, which is only used whenever new feedback or
        # predictor labels are given. The feedback and pred encs are then
        # updated with that CLIP encoding and the shape preserving is handled
        # by putting the similarity vectors in the correct spot of label_text,
        # handled by get_similrity()

        with torch.no_grad():
            zeroshot_weights = []
            for label_text in label_texts:
                # Place the class label text inside each template text and
                # tokenize
                texts = clip.tokenize([
                    template.format(label_text.lower())
                    for template in templates
                ]).cuda()

                # CLIP Encode the text, normalize dividing by L1 norm
                label_embeddings = model.encode_text(texts)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

                # Get label encoding as normalized mean again divide by L1 norm
                label_embedding = label_embeddings.mean(dim=0)
                label_embedding /= label_embedding.norm()

                zeroshot_weights.append(label_embedding)

        return torch.stack(zeroshot_weights, dim=1)

    def get_similarity(self, label_text):
        """Return the similarity vectors"""
        # TODO convert labels to idx, then fill in each idx w/ corresponding
        # similrity matrix row.
        return self.similarity[self.feedback_known_map.encode(label_text)]

    def update_known_preds(self, new_pred_label):
        """Given new known pred, update the existing similarity matrix, etc."""
        # TODO this gets interesting because how do we textually encode a new
        # pred label? Simplest case is to perhaps, after determinig, there is a
        # new pred label to be assigned, map that label to all other feedback
        # labels by a similarity. This would be based on all prior samples of
        # this new pred label, and so could simply be the mean of all samples'
        # similarities to a feedback label per feedback label, thus a separate
        # mean per element in the new pred label's column.

        # TODO, but then how do we calculate new feedback labels to this new
        # pred label that lacks text? Basically take the similarity of the new
        # feedback label to all existing feedback labels weighted by the
        # similarity to the pred label.

    def update_known_feedback(self, new_feedback_label):
        """Update state with the new known feedback label text."""
        # Update index encoding of known feedback labels.
        self.feedback_known_map.append(new_feedback_labels)

        # Get and save clip encoding of new feedback labels
        self.feedback_label_encs.append(
            self.clip_encode_text(new_feedback_labels)
        )

        # TODO Update the similarity of this new feedback to predictor's knowns
        # TODO Get cosine similarity of each new feedback labels to predictor's
        # TODO Save these cosine similarities to the cosine similarity matrix


    def interpret_feedback(
        self,
        label_text,
        unknown_last_dim=True,
    ):
        """Interprets the given feedback of multiple label texts per sample
        returning a soft label vector per sample for the with dimensions =
        knowns + unknown.

        Args
        ----
        label_text : list(list(str)) | np.ndarray(str)
            A matrix of text strings where rows are samples and columns are the
            number of classes given back sa feedback, which is assumed to be 5
            due to protocol with PAR.

        Returns
        -------
        np.ndarray | torch.Tensor?
            The probability vectors of predictor known classes and an unknown
            class for each sample, which is a matrix of shape (samples,
            predictor_knowns + 1). This indicates the predicted probablity that
            each feedback sample corresponds to which predictor known class or
            none of them (the unknown). Note that unknown probability is as the
            last element of each vector row.
        """
        if isinstance(label_text, np.ndarray):
            np.array(label_text)

        # Check for new feedback labels and update feedback state
        self.update_known_preds(np.unique(label_text))

        # With similarity updated, get the probability vectors for known labels
        # TODO Get the normalized cosine similarity to predictor's known labels
        sims = self.get_similarity(label_text)

        # TODO Get the probablity of none of the predictor's known labels
        # (unknown)

        # NOTE follows ONE  method of getting probs, expecting probs to have
        # columns be [0,1], but rows not to sum to 1.
        # Find probability of unknown as 1 - max 1-vs-Rest and concat
        if unknown_last_dim:
            probs = torch.cat((probs, 1 - torch.max(probs, 1, True).values), 1)
        else:
            probs = torch.cat((1 - torch.max(probs, 1, True).values, probs), 1)

        # Get a normalized probability vector keeping raitos of values.
        return probs / probs.sum(1, True)
        #return pred_known_or_unknown_probs

    def novelty_recog(
        self,
        feedback_probs,
        feature_repr=None,
        task_repr=None,
        videos=None,
        preds=None,
        predictor_belief=None,
    ):
        """Determine if given the interpretted feedback is a new predictor
        label from the current set of known predictor labels. This is this
        interpreter's novelty recognition decision per sample given available
        predictor informaiton and its own state.

        Args
        ----
        feedback_probs : torch.Tensor
            A matrix of shape (samples, probs_known_labels + 1) which is the
            interpretted feedback of samples as found from
            `interpret_feedback()`.
        feature_repr : torch.Tensor = None
            The feature representation of the input samples.
        task_repr : torch.Tensor = None
            The fine-tuning or task representation of the samples.
        videos : torch.Tensor
            The actual input videos, aligned to the label text, feature_repr,
            and task_repr, etc.
        preds : torch.Tensor
            The predictor's predictions for the sample.
        pred_belief : torch.Tensor = None
            The predictor's belief or certainty in the predictions for the
            sample.

        Returns
        -------
        str
            The class decided upon for the given samples when interpretting
            feedback and the predictor's novelty detection and recognition.
        """
        raise NotImplementedError()
        # TODO ??? Obtain a CLIP encoding per row of feedback labels? Mean?
        #   TODO Sort each row of feedback labels lexically.
        #   Gives cenroid, may be useful in detecting unknown classes

        return
