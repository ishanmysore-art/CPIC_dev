import torch
import numpy as np


def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


def reduce_logmeanexp_nodiag(x, dim=[0,1], device="cuda:0"):
    batch_size = x.size()[0]
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)
    if dim == [0,1]:
        num_elem = batch_size * (batch_size - 1.)
        return logsumexp - torch.log(torch.tensor(num_elem).to(device))
    elif dim == 1:
        return logsumexp - torch.log(torch.tensor(batch_size - 1.).to(device))
    else:
        raise Exception("Sorry, this function is not implemented.")
        