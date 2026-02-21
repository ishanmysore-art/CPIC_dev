import torch
import numpy as np
import dca as DCA


def decoderscores(x_mean, x_vars, x, threshold=1e-6, debug=False):
        """
        :param x_mean: batch_size x x_dim
        :param x_vars: batch_size x x_dim
        :param x: batch_size x x_dim
        :return: scores batch_size x batch_size
        """
        batch_size = x.shape[0]
        x_mean_tiled = torch.tile(x_mean[None, :], (batch_size, 1, 1))
        x_vars_tiled = torch.tile(x_vars[None, :], (batch_size, 1, 1))
        # robust computation
        x_vars_tiled[x_vars_tiled < threshold] += threshold
        x_tiled = torch.tile(x[:, None], (1, batch_size, 1))
        scores = torch.sum(-0.5*(torch.log(x_vars_tiled) + (x_tiled - x_mean_tiled)**2/x_vars_tiled), axis=-1)
        if debug:
            import pdb; pdb.set_trace()
        return scores


def DCA_init(X, T, d, n_init=1, rng_or_seed=None):
    opt = DCA.DynamicalComponentsAnalysis(T=T, rng_or_seed=rng_or_seed)
    opt.estimate_data_statistics(X)
    opt.fit_projection(d=d, n_init=n_init)
    V_dca = opt.coef_
    return V_dca


def Polynomial_expand(x):
    res = list()
    feature_dim = x.shape[-1]
    for i in range(feature_dim):
        res.append(x[..., i])
        for j in range(i, feature_dim):
            res.append(x[..., i] * x[..., j])
    return np.stack(res, axis=-1)
    