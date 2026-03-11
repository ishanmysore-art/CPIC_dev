import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import torch.nn as nn


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


def _extract_conv_layers(module):
    """
    Extract all Conv2d and Conv1d layers from a module (in order).
    """
    layers = []
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            layers.append(m)
    return layers


def _visualize_kernel_layer_1d(layer, layer_idx, save_dir, type='mean'):
    """
    Visualize kernels for a single Conv1d layer.
    """    
    kernels = layer.weight.detach().clone().cpu()
    # average over in_channels: (out_ch, kernel_size)
    kernels = kernels.mean(dim=1).numpy()
    kernels = kernels - kernels.min()
    if kernels.max() != 0:
        kernels = np.abs(kernels / kernels.max())

    # Treat each 1D kernel as a small patch: (out_ch, k) -> (out_ch, 1, height, k)
    out_ch, k = kernels.shape
    bar_height = k  # use kernel size so each bar is roughly square
    # (out_ch, k) -> (out_ch, bar_height, k) by repeating along height
    patches = np.tile(kernels[:, np.newaxis, :], (1, bar_height, 1))
    # (out_ch, bar_height, k) -> (out_ch, 1, bar_height, k) for make_grid
    patches = torch.from_numpy(patches).unsqueeze(1)
    filter_img = torchvision.utils.make_grid(patches, nrow=8, padding=2)
    filter_img_2d = filter_img[0].numpy()

    plt.figure()
    plt.imshow(filter_img_2d, cmap='gist_gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{type} Layer {layer_idx} - {kernels.shape[0]} filters')
    plt.savefig(os.path.join(save_dir, f'{type}_kernel_layer_{layer_idx}.png'))
    plt.close()


def _visualize_kernel_layer(layer, layer_idx, save_dir, type='mean'):
    """
    Visualize kernels for a single Conv2d layer.

    Parameters
    ----------
    layer : nn.Conv2d
        Convolutional layer to visualize
    layer_idx : int
        Index of the layer
    save_dir : str, optional
        Directory to save the visualized kernels
    type : str, optional
        Type of layer to visualize ('mean' or 'std')
    """
    kernels = layer.weight.detach().clone().cpu()
    kernels = kernels.mean(dim=1, keepdim=True)
    
    print(kernels.size())
    kernels = kernels - kernels.min()
    if kernels.max() != 0:
        kernels = torch.abs(kernels / kernels.max())
    
    filter_img = torchvision.utils.make_grid(kernels, nrow=8)
    # Take first channel only to get 2D array for colormap
    filter_img_2d = filter_img[0, :, :]
    plt.imshow(filter_img_2d, cmap='gist_gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{type} Layer {layer_idx} - {kernels.shape[0]} filters')
    plt.savefig(os.path.join(save_dir, f'{type}_kernel_layer_{layer_idx}.png'))
    plt.close()


def visualize_conv_kernels(model, save_dir=None):
    """
    Visualize convolutional kernels for a given model.

    Parameters
    ----------
    model : nn.Module
        Model to visualize convolutional kernels for
    save_dir : str, optional
        Directory to save the visualized kernels
    """
    encoder = model.encoder
    
    encoder_type = getattr(encoder, "encoder_type", None)
    is_linear = getattr(encoder, "linear_encoder", False)
    if encoder_type not in ("conv_spatial", "conv_spatiotemporal", "conv_temporal") or is_linear:
        raise ValueError(
            f"Encoder is not a conv encoder or is using linear encoding. "
            f"encoder_type: {encoder_type}, linear_encoder: {is_linear}"
        )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    mean_layers = _extract_conv_layers(encoder._mean)
    for idx, layer in enumerate(mean_layers):
        if isinstance(layer, nn.Conv1d):
            _visualize_kernel_layer_1d(layer, idx, save_dir, 'mean')
        else:
            _visualize_kernel_layer(layer, idx, save_dir, 'mean')
    if not encoder.deterministic:
        std_layers = _extract_conv_layers(encoder._logvars)
        for idx, layer in enumerate(std_layers):
            if isinstance(layer, nn.Conv1d):
                _visualize_kernel_layer_1d(layer, idx, save_dir, 'std')
            else:
                _visualize_kernel_layer(layer, idx, save_dir, 'std')


def DCA_init(X, T, d, n_init=1, rng_or_seed=None):
    """
    Initialize a Dynamical Components Analysis (DCA) projection matrix.

    Requires the optional dependency: install with ``pip install cpic[dca]``.

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
    try:
        import dca as DCA
    except ImportError:
        raise ImportError(
            "DCA_init requires the DynamicalComponentsAnalysis package. "
            "Install it with: pip install cpic[dca]"
        ) from None
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
    