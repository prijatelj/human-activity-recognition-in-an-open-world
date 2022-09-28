"""Feedback models, specifically CLIP label encoding."""
import os

import numpy as np
import pandas as pd
import torch
import math
import clip

from exputils.data.labels import NominalDataEncoder


def calc_similarity(first, second, scalar=1.0):
    """Calculate the unnormalized cosine similarity scaled by some factor."""
    return (
        scalar
        * (
            first / first.norm(dim=-1, keepdim=True)
        ) @ (
            second / second.norm(dim=-1, keepdim=True)
        ).T
    )


class CLIPFeedbackInterpreter(object):
    """Feedback Interpretter using CLIP zeroshot.

    Attributes
    ----------
    clip : CLIP
        CLIP model used for encoding both text and images.
    clip_templates : list(str)
        The text templates that every label text is inserted into and used to
        obtain the CLIP text encoding representation of that label text.
    pred_known_map : NominalDataEncoder
        Bidirectional map of predictor known texts to index. This indexing is
        consistent across predictor_label_encs and similarity columns.
        This serves as the unique known predictor labels.
    pred_label_encs : torch.Tensor
        CLIP processed label encodings of predictor's known label text.
    feedback_known_map : NominalDataEncoder
        Bidirectional map of feedback known texts to index. This indexing is
        consistent across feedback_label_encs and similarity rows.
        This serves as the unique known feedback labels.
    feedback_label_encs : torch.Tensor
        CLIP processed label encodings of feedback label text.
    similarity : np.ndarray | torch.Tensor?
        A Matrix of rows being known feedback label texts and columns being the
        known predictor labels. This matrix's elements are the unnormalized
        cosine similarities of the feedback labels to predictor labels. This is
        unnormalized because the normalization process is dependent upon the
        number of predictor classes and better to preserve the raw cosine
        similarities and normalize them on demand, rather than have to
        re-normalize without introducing error.
    new_pred_method : str = 'weighted_centroid'
        The method of representation of new predictor labels for this
        interpreter since no text represents this label. Default and currently
        only method implemented is the weighted similarity score of every
        feedback label to every other feedback label. The weighting is based on
        how many times each feedback label was used to represent the samples of
        this new predictor label.
    new_pred_start_idx : int
        The integer index in pred_known_map for the first new prediction label.
        Subtract the new prediction encodings by this value to get their index
        in `new_pred_repr`.
    new_pred_repr : torch.Tensor
        A tensor of shape (number_of_new_prediction_labels, feedback_label)
        that stores the counts of feedback labels being related to a new
        prediction label. This is used for getting the similarity score of each
        feedback label to a new prediction label that lacks text representation
        by calculating the frequency and performing a weighted arithmetic mean
        on the feature encodings, or a weighted centroid.
    """
    def __init__(
        self,
        clip_path,
        clip_templates,
        pred_known_map,
        pred_label_encs=None,
        feedback_known_map=None,
        feedback_label_encs=None,
        similarity=None,
        new_pred_method='weighted_centroid',
        device='cpu',
        float_dtype=torch.float32,
    ):
        """Initialize and load the CLIPFeedbackInterpreter
        Args
        ----
        clip_path : str
            Path to the clip model
        clip_templates : str | list(str)
            Path to the clip templates file or the actual list of str templates
        pred_known_map : str | list(str) | NominalDataEncoder
            Path to the predictor's known labels or the known labels themselves
        pred_label_encs : torch.Tensor = None,
            Path to the predictor's known label encodings or the encoding
            themselves.
        feedback_known_map : str | list(str) | NominalDataEncoder = None
            Path to the known feedback labels or the labels themselves.
        feedback_label_encs : torch.Tensor = None
            Path to the known feedback label encodings or the encoding
            themselves.
        similarity : str | torch.Tensor = None
            Path to the similarity martrix to be loaded or the similarity
            matrix.
        float_dtype : torch.Type = torch.float32
            The default torch type for to use for the floats.
        """
        # Load in CLIP Pre-trained for encoding new feedback label text
        self.clip = clip.load(clip_path, device)[0]
        self.device = device
        self.float_dtype = float_dtype

        if isinstance(clip_templates, str):
            with open(clip_templates) as openf:
                self.clip_templates = openf.read().splitlines()
        else: # NOTE I hope you set to correct types and all that.
            self.clip_templates = clip_templates

        # Predictor label map of text to idx
        if isinstance(pred_known_map, str):
            ext = os.path.splitext(pred_known_map)[-1]
            if ext in {'.csv', '.json'}:
                loader = pd.read_csv if ext == '.csv' else pd.read_json

                # Load DataFrame, expect to get unique lowers & some PascalCase
                self.pred_known_map = (
                    pd.Series(loader(pred_known_map)['par_class'].unique())
                    .str.replace(r'(?<![ ^])(?=[A-Z])', ' ')
                    .str.replace('  ', ' ') # May not be necessary.
                    .str.strip()
                    .str.lower()
                )

                # rm 'unknown'
                self.pred_known_map = NominalDataEncoder(sorted(
                    self.pred_known_map[
                        self.pred_known_map != 'unknown'
                    ].tolist()
                ))
            else:
                self.pred_known_map = NominalDataEncoder.load(pred_known_map)

        else: # NOTE Type checking goes here to die.
            self.pred_known_map = pred_known_map

        if isinstance(pred_label_encs, str):
            self.pred_label_encs = torch.load(pred_label_encs).type(
                self.float_dtype,
            )
        elif pred_label_encs is None and self.pred_known_map is not None:
            self.pred_label_encs = self.clip_encode_text(
                list(self.pred_known_map.encoder)
            ).type(self.float_dtype)
        else: # NOTE Type checking is dead.
            self.pred_label_encs = pred_label_encs.type(self.float_dtype)

        # Label map for feedback label text to idx
        if isinstance(feedback_known_map, str):
            self.feedback_known_map = NominalDataEncoder.load(feedback_known_map)
        else: # NOTE Type checking ist tot.
            self.feedback_known_map = feedback_known_map

        if isinstance(feedback_label_encs, str):
            self.feedback_label_encs = torch.load(feedback_label_encs).type(
                self.float_dtype
            )
        elif (
            feedback_label_encs is None
            and self.feedback_known_map is not None
        ):
            self.feedback_label_encs = self.clip_encode_text(
                list(self.feedback_known_map.encoder)
            ).type(self.float_dtype)
        else: # NOTE Wow. Almost like someone should try to auto this...oh wait
            self.feedback_label_encs = feedback_label_encs.to(self.float_dtype)

        # Cosine similarity matrix of (feedback labels) X (predictor knowns) to
        # one another when in CLIP encoded space.
        if isinstance(similarity, str):
            self.similarity = torch.load(similarity)
        elif similarity is None and self.feedback_label_encs is not None:
            # Calculate the similarity matrix now using clip.
            self.similarity = calc_similarity(
                self.feedback_label_encs,
                self.pred_label_encs,
            )
        else: # NOTE Much redundancy wow If only I could finish my side project
            self.similarity = similarity

        if new_pred_method != 'weighted_centroid':
            raise NotImplementedError(
                'Only `weighted_centroid` is implemented.',
            )
        self.new_pred_method = new_pred_method
        self.new_pred_repr = None
        self.new_pred_start_idx = None

    def clip_encode_text(self, label_texts):
        """Return the clip encoding of the label text preserving shape.
        Args
        ----
        label_texts : list(str)
            The labels' text to be encoded with the given text templates.

        Returns
        -------
        torch.Tensor
            The CLIP model's text encoding of each label text using the given
            templates in order.
        """
        # NOTE this does not need to preserve shape! This can simply encode the
        # given text labels, which is only used whenever new feedback or
        # predictor labels are given. The feedback and pred encs are then
        # updated with that CLIP encoding and the shape preserving is handled
        # by putting the similarity vectors in the correct spot of label_text,
        # handled by get_similrity()

        # TODO Support pytorch self.device

        with torch.no_grad():
            zeroshot_weights = []
            for label_text in label_texts:
                # Place the class label text inside each template text and
                # tokenize
                texts = clip.tokenize([
                    template.format(label_text.lower())
                    for template in self.clip_templates
                ]).to(self.device)

                # CLIP Encode the text, normalize dividing by L1 norm
                label_embeddings = self.clip.encode_text(texts)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

                # Get label encoding as normalized mean again divide by L1 norm
                label_embedding = label_embeddings.mean(dim=0)
                label_embedding /= label_embedding.norm()

                zeroshot_weights.append(label_embedding)

        return torch.stack(zeroshot_weights, dim=1).to(self.device).T

    def get_similarity(self, label_text):
        """Return the similarity vectors of feedback text to pred label text.

        Args
        ----
        label_text : list(list(str)) | np.ndarray(str)
            A matrix of text strings where rows are samples and columns are the
            number of classes given back sa feedback, which is assumed to be 5
            due to protocol with PAR.
        """
        # Convert labels to idx, then fill in each idx w/ corresponding sim.
        # Get the similarity of label text to the known predictor labels.
        #return self.similarity[self.feedback_known_map.encode(label_text)]
        # TODO fix exputils encoder upstream to allow ndarrays, not just vecs
        return self.similarity[[np.searchsorted(
            self.feedback_known_map.encoder,
            label_text,
        )]]

    def update_known_preds(
        self,
        new_pred_label,
        annotations,
        #features,
        videos=None,
    ):
        """Given new known pred, update the existing similarity matrix, etc.

        Args
        ----
        new_pred_label : str
            The str identifier of the new pred label, BUT is not used in
            similarity scores to feedback labels. This is added to the encoding
            map of pred labels to index, but the similarity calculations have
            to be different; something other than direct clip encoding
            comparison of texts.

            Also, this allows for updating prior counts to feedback text of an
            prediction label not represented by text.
        annotations : list(list(str))
            A list of length number of samples where each element is a list of
            feedback annotaitons given for that sample.
        features : torch.Tensor
            A tensor of shape (samples, feature_repr_dim) that contains the
            samples decided as part of this new predictor label. This will be
            the sets of feedback labels per sample. This could include more
            information.
        videos : torch.Tensor = None
            ??? Perhaps...

        Note
        ----
        At this point, novelty has been detected, and samples have been
        recognized as a distinct unknown class from the known classes in the
        feature representation space. All these decisions have been made, only
        thing left is to represent this new class in a way that allows us to
        translate future feedback into a probability of it being this new
        class.

        A simple top 1 assessment of this method after testing shows it does
        not return the new pred labels for the exact annotation label text
        given to specify that new pred label, which means that the weighted
        centroid in CLIP space may not be work with cosine similarity.

        The 30th and 31st, which were added pred labels, do occur as 2nd or 3rd
        in the top 5 similar.
        """
        # TODO this gets interesting because how do we textually encode a new
        # pred label? Simplest case is to perhaps, after determinig, there is a
        # new pred label to be assigned, map that label to all other feedback
        # labels by a similarity. This would be based on all prior samples of
        # this new pred label, and so could simply be the mean of all samples'
        # similarities to a feedback label per feedback label, thus a separate
        # mean per element in the new pred label's column.

        # annotations is simply prior feedback labels per sample (label text)

        # TODO, but then how do we calculate new feedback labels to this new
        # pred label that lacks text? Basically take the similarity of the new
        # feedback label to all existing feedback labels weighted by the
        # similarity to the pred label.

        if self.new_pred_method != 'weighted_centroid':
            raise NotImplementedError(
                'update_known_preds() expects weighted centroid method'
            )

        # Save the running counts of each to later get frequency as weights
        uniques, counts = torch.unique(
            torch.Tensor(np.searchsorted(
                self.feedback_known_map.encoder,
                annotations,
            )).type(torch.long),
            sorted=True,
            return_counts=True,
        )
        uniques = uniques.to(self.device)
        counts = counts.type(self.float_dtype).to(self.device)

        # Put the counts into correct index for each feedback label
        new_pred_counts = torch.zeros(
            [len(self.feedback_known_map.encoder)]
        ).to(self.device)
        new_pred_counts[uniques] = counts

        if self.new_pred_repr is None:
            # First new predictor label, so create it
            self.new_pred_repr = new_pred_counts
            self.new_pred_start_idx = len(self.pred_known_map.encoder)
            self.pred_known_map.append(new_pred_label)

            # Update the similarity matrix's new pred protion.
            self.similarity = torch.cat(
                [
                    self.similarity,
                    calc_similarity(
                        self.feedback_label_encs,
                        (
                            (self.new_pred_repr @ self.feedback_label_encs)
                            / self.new_pred_repr.sum()
                        )
                    ).reshape(-1, 1),
                ],
                dim=1,
            )
        elif new_pred_label in self.pred_known_map.encoder:
            # Pre-existing no-text predictor label (unknown), update counts
            pred_idx = self.pred_known_map.encoder[new_pred_label]
            self.new_pred_repr[pred_idx - self.new_pred_start_idx] += \
                new_pred_counts

            pred_counts = self.new_pred_repr[[
                pred_idx - self.new_pred_start_idx
            ]]

            # Update the similarity matrix's new pred protion.
            self.similarity[:, pred_idx] = calc_similarity(
                self.feedback_label_encs,
                (
                    (pred_counts @ self.feedback_label_encs)
                    / pred_counts.sum(1, keepdims=True)
                ),
            ).squeeze()
        else:
            # New no-text predictor label (unknown), add counts and sims
            self.new_pred_repr = torch.stack(
                [self.new_pred_repr, new_pred_counts],
            )
            new_pred_counts = new_pred_counts.reshape(-1, 1)
            self.pred_known_map.append(new_pred_label)

            # Update the similarity matrix's new pred protion.
            self.similarity = torch.cat(
                [
                    self.similarity,
                    calc_similarity(
                        self.feedback_label_encs,
                        (
                            (self.new_pred_repr[-1] @ self.feedback_label_encs)
                            / self.new_pred_repr[-1].sum()
                        )
                    ).unsqueeze(1),
                ],
                dim=1,
            )

    def update_known_feedback(self, new_feedback_labels):
        """Update state with the new known feedback label text."""
        # Update index encoding of known feedback labels.
        self.feedback_known_map.append(new_feedback_labels)

        # Get and save clip encoding of new feedback labels
        new_encs = self.clip_encode_text(new_feedback_labels)
        self.feedback_label_encs.append(new_encs)

        # Update the similarity of this new feedback to predictor's knowns
        self.similarity.append(calc_similarity(new_encs, self.pred_label_encs))

    def interpret(self, label_text, dim=-1, unknown_last_dim=None):
        """Interprets the given feedback of multiple label texts per sample
        returning a soft label vector per sample for the with dimensions =
        knowns OR if `unknown_last_dim` given True or False the dimensions =
        known + 1 with the 1 representing the general unknown class located at
        the last or first dimension, respectively.

        Args
        ----
        label_text : list(list(str)) | np.ndarray(str)
            A matrix of text strings where rows are samples and columns are the
            number of classes given back sa feedback, which is assumed to be 5
            due to protocol with PAR.
        unknown_last_dim : bool = None
            If not provided (default), the probability vectors are of length
            predictor known labels. If True, then the unknown prob is
            calculated as the 1 - max probability of the knowns and is
            appeneded to the end of the vector. If False, calculated the same
            way, but pre-appeneded to the beginning of the probability vectors.

        Returns
        -------
        torch.Tensor
            The probability vectors of predictor known classes and an,
            optionally, unknown class for each sample, which is a matrix of
            shape (samples, predictor_knowns + 1). This indicates the predicted
            probablity that each feedback sample corresponds to which predictor
            known class or none of them (the unknown). Note that unknown
            probability is as the last element of each vector row.

        Note
        ----
        The use of unknown probability being added in this way is the same as
        how the ExtremeValueMachine however it was okay to do in the extreme
        value machine because all of the source probabilities came from one
        versus rest classifiers that were all trained on the same data, so they
        were dependent upon each other. I am uncertain if cosine similarity can
        be used in the same way and am witnessing a lot of "uncertain"
        probability vectors where all the elements are low values.

        For delivery of feedback, it probably would be best to just use the
        known probs w/o unknown and then let the predictor leverage other info
        to determine novelty recognition.
        """
        if isinstance(label_text, np.ndarray):
            np.array(label_text)

        # Check for new feedback labels and update feedback state
        uniques = np.unique(label_text)
        if set(uniques) - set(self.feedback_known_map.encoder):
            self.update_known_feedback(uniques)

        # With similarity updated, get the probability vectors for known labels
        # Get the normalized cosine similarity to predictor's known labels
        sims = self.get_similarity(label_text)
        sims = sims / torch.Tensor([math.pi]).to(sims.device)

        # After dividing by pi, columns be [0,1], but rows not to sum to 1.
        if unknown_last_dim is None:
            return sims / sims.sum(dim, True)

        # Find probability of unknown as 1 - max and concat
        if unknown_last_dim:
            sims = torch.cat((sims, 1 - torch.max(sims, dim, True).values), 1)
        else:
            sims = torch.cat((1 - torch.max(sims, dim, True).values, sims), 1)

        # Get a normalized probability vector keeping raitos of values.
        return sims / sims.sum(dim, True)

    def save(self, filepath):
        """Save the state of this CLIPFeedbackInterpreter."""
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        """Load the CLIPFeedbackInterpreter from the given filepath."""
        raise NotImplementedError()

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
            `interpret()`.
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
        raise NotImplementedError('NoveltyRecognizer should handle this.')
        # TODO ??? Obtain a CLIP encoding per row of feedback labels? Mean?
        #   TODO Sort each row of feedback labels lexically.
        #   Gives cenroid, may be useful in detecting unknown classes

        # NOTE this may make more sense to be done by the NoveltyRecognizer
        # which then calls this objects update_known_preds()

        return
