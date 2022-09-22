"""Gaussian Mixture Model per class where the partitions are fit by FINCH.
To find the log_prob, weighted sum of all gaussians in the mixture, which is
weighted by their mixture probabilities.
"""

# Functions first

def recognize_fit(
    class_name,
    features,
    counter=0,
    threshold_method='credible_ellipse',
    allowed_error=1e-5,
    max_likely_gmms=False,
    **kwargs,
):
    """For a single class' features, fit a Gaussian Mixture Model to it using
    the clusters found by FINCH.

    Args
    ----
    class_name : str
        The name for this class. Used to construct the labels in the resulting
        label encoder as the prefix to all the labels:
            `f'{class_name}_{component_idx}'`
    features: np.ndarray | torch.Tensor
        2 dimensional float tensor of shape (samples, feature_repr_dims)
        that are the features deemed to be corresponding to the class.
    threshold_method : str 'credible_ellipse'
        The method used to find the thresholds per component, defaulting to
        use the credible ellipse per gaussian given the accepted_error (alpha).
    accepted_error : float = 1e-5
        The accepted error (alpha) to be used to find the corresponding 1-alpha
        credible ellipse per gaussian component.
    **kwargs : dict
        Key word arguments for FINCH for detecting clusters.

    Returns
    -------
    NominalDataEncoder, list, list
        The label encoder for this class' components, a list of the components'
        MultivariateNormal distributions and a list of the components'
        thresholds.
    """
    raise NotImplementedError


#class GMMFINCH(OWHARecognizer):
#    """Gaussian Mixture Model per class using FINCH to find the components."""
