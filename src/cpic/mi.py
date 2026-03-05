import torch
import numpy as np
from .utils.math import reduce_logmeanexp_nodiag, KL_between_normals
from .utils.helpers import decoderscores


def tuba_lower_bound(scores, log_baseline=None, device="cuda:0"):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    joint_term = torch.mean(torch.diag(scores))
    marg_term = torch.exp(reduce_logmeanexp_nodiag(scores, device=device))
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores, device='cuda:0'):
    # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
    return tuba_lower_bound(scores - 1., device=device)


def mine_lower_bound(scores, device='cuda:0'):
    # equivalent to: tuba_lower_bound(scores)
    return tuba_lower_bound(scores, device=device)


def infonce_upper_bound(scores, device='cuda:0'):
    '''Bound from Van Den Oord and al. (2018)
    scores are either known log conditional distribution log p(y|x) or critic function f(x,y).
    '''
    mi = torch.mean(torch.diag(scores) - reduce_logmeanexp_nodiag(scores, dim=1, device=device))
    return mi


def infonce_lower_bound(scores):
    '''Bound from Van Den Oord and al. (2018)'''
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores,dim=1))
    k = scores.size()[0]
    mi = np.log(k) + nll
    return mi


def vub_upper_bound(mean, vars, device='cuda:0'):
    batch_size = mean.size()[0]
    mean = mean.reshape(batch_size, -1)
    vars = vars.reshape(batch_size, -1)
    dimY = mean.size()[1]
    prior_Y_distr = torch.zeros(batch_size, dimY).to(device), torch.ones(batch_size, dimY).to(device)
    encoder_Y_distr = mean, vars
    return torch.mean(KL_between_normals(encoder_Y_distr, prior_Y_distr))


def estimate_mutual_information(estimator, x, y, critic_fn=None, baseline_fn=None, decoder=None, device='cuda:0', debug=False, *args, **kwargs):
    """
    Estimate mutual information using variational lower or upper bounds.

    Parameters
    ----------
    estimator : str
        Which estimator to use. One of
        {'nwj', 'infonce_lower', 'infonce_upper', 'tuba', 'mine', 'vub'}.
    x : torch.Tensor
        Input tensor of shape [batch_size, dim_x].
    y : torch.Tensor
        Input tensor of shape [batch_size, dim_y].
    critic_fn : callable, optional
        Function taking `(x, y)` and returning a score matrix of shape
        [batch_size, batch_size]; used by all estimators except 'vub' when
        `decoder` is provided.
    baseline_fn : callable, optional
        Function taking `y` and returning a baseline of shape
        [batch_size] or [batch_size, 1]; used by the 'tuba' estimator.
    decoder : callable, optional
        Decoder mapping `x` to `(mean, vars)`; when provided, it is used with
        `decoderscores` for likelihood-based estimators and with 'vub' to form
        the variational upper bound.
    device : str, optional
        Device on which to perform tensor operations (e.g. 'cuda:0' or 'cpu').
    debug : bool, optional
        If True, enters debug mode after computing the estimate.

    Returns
    -------
    mi : torch.Tensor
        Scalar tensor containing the estimated mutual information bound.
    """
    if critic_fn is not None:
        scores = critic_fn(x, y)
    if decoder is not None:
        decoded_mean, decoded_vars = decoder(x)
        batch_size = decoded_mean.shape[0]
        decoded_mean_reshaped = decoded_mean.reshape(batch_size, -1)
        decoded_vars_reshaped = decoded_vars.reshape(batch_size, -1)
        scores = decoderscores(decoded_mean_reshaped, decoded_vars_reshaped, y)
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))

    match estimator:
        case "infonce_lower":
            mi = infonce_lower_bound(scores)
        case "infonce_upper":
            mi = infonce_upper_bound(scores, device=device)
        case "vub":
            mi = vub_upper_bound(decoded_mean, decoded_vars, device=device)
        case "nwj":
            mi = nwj_lower_bound(scores, device=device)
        case "mine":
            mi = mine_lower_bound(scores, device=device)
        case "tuba":
            mi = tuba_lower_bound(scores, log_baseline, device=device)
        case _:
            raise ValueError(f"Unknown estimator: {estimator}")
    if debug:
        import pdb; pdb.set_trace()
        decoderscores(decoded_mean_reshaped, decoded_vars_reshaped, y, debug=debug)
    return mi
    