import torch
import numpy as np
import dca as DCA


def decoderscores(x_mean, x_vars, x, threshold=1e-6, debug=False):
        """
        Compute pairwise log-likelihoods log p(x_i | z_j) over a batch, where
        p(x | z) is a factorized Gaussian with mean x_mean and variance x_vars.

        The returned matrix scores has shape [batch_size, batch_size], with
        scores[i, j] = log p(x_i | z_j).

        Parameters
        ----------
        x_mean : torch.Tensor of shape [batch_size, dim_x]
            Decoder means for each latent sample z_j in the batch.
        x_vars : torch.Tensor of shape [batch_size, dim_x]
            Decoder variances for each latent sample z_j in the batch.
        x : Tensor of shape [batch_size, dim_x]
            The input samples for which to compute the scores.
        threshold : float, optional
            A small value to ensure numerical stability when computing the scores. Default is 1e-6.
        debug : bool, optional
            If True, enters debug mode after computing the scores. Default is False.
        """
        batch_size = x.shape[0]
        x_mean_tiled = torch.tile(x_mean[None, :], (batch_size, 1, 1))
        x_vars_tiled = torch.tile(x_vars[None, :], (batch_size, 1, 1))
        # robust computation
        x_vars_tiled[x_vars_tiled < threshold] += threshold
        x_tiled = torch.tile(x[:, None], (1, batch_size, 1))
        scores = torch.sum(-0.5*(torch.log(x_vars_tiled) + (x_tiled - x_mean_tiled)**2/x_vars_tiled), axis=-1)
        if debug:
            print("Decoder scores shape:", scores.shape)
            print("Decoder scores sample:", scores[0, :5])
        return scores


def DCA_init(X, T, d, n_init=1, rng_or_seed=None):
    """
    Initialize a Dynamical Components Analysis (DCA) projection matrix.

    Parameters
    ----------
    X : ndarray
        Input data of shape [n_samples, n_features].
    T : int
        Temporal window length used by DCA.
    d : int
        Number of dynamical components to extract.
    n_init : int, optional
        Number of random initializations for fitting the projection. Default is 1.
    rng_or_seed : int or numpy.random.Generator, optional
        Random seed or RNG passed to the DCA optimizer.

    Returns
    -------
    V_dca : ndarray
        Learned DCA projection matrix of shape [n_features, d].
    """
    opt = DCA.DynamicalComponentsAnalysis(T=T, rng_or_seed=rng_or_seed)
    opt.estimate_data_statistics(X)
    opt.fit_projection(d=d, n_init=n_init)
    V_dca = opt.coef_
    return V_dca


def Polynomial_expand(x):
    """
    Compute a second-order polynomial feature expansion.

    For an input with last dimension feature_dim, this returns all original
    features followed by all pairwise products x_i * x_j with i <= j,
    stacked along the last axis.

    Parameters
    ----------
    x : ndarray
        Input array of shape [..., feature_dim].

    Returns
    -------
    ndarray
        Polynomially expanded features of shape [..., expanded_dim].
    """
    res = list()
    feature_dim = x.shape[-1]
    for i in range(feature_dim):
        res.append(x[..., i])
        for j in range(i, feature_dim):
            res.append(x[..., i] * x[..., j])
    return np.stack(res, axis=-1)
    