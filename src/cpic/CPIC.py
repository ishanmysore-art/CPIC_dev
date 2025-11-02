import torch
from torch import nn
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from cpic.utils import StructuredEncoder, CRITICS, BASELINES, estimate_mutual_information


class CPIC(nn.Module):
    """Compressed Predictive Information Coding.

    Extract latent representations from time-series data X that maximizes the complexity, as defined by 
    Predictive Information (PI). CPIC selectively projects the past (input) X into a low dimensional space 
    that is predictive about the compressed data projected from the future (output) y. The key insight 
    of our framework is to learn representations by balancing the minimization of compression complexity
    with maximization of the predictive information in the latent space. We derive tractable
    variational bounds of the CPIC loss by leveraging bounds on mutual information. The
    CPIC loss induces the latent space to capture information that is maximally predictive
    of the future of the data from the past.

    Parameters
    ----------
    xdim : int
        Dimensionality of the input data.
    ydim : int
        Dimensionality of the output data.
    mi_params : str
        Parameters for mutual information estimation. A dictionary with keys:
    critic_params : int
        Parameters for critic function. A dictionary with keys:
    baseline_params : int
        Parameters for baseline function. A dictionary with keys:
    T : int, optional
        Length of the time window to compute PI. The default is 4.
    beta : float, optional
        Weight for compression term I_compress. The default is 1e-3.
    beta1 : float, optional
        Weight for predictive term I_predictive. The default is 1.
    beta2 : float, optional
        Weight for I_YX term. The default is 1.
    hidden_dim : int, optional
        Hidden dimension for encoder and critic networks. The default is 256.
    deterministic : bool, optional
        Whether to use deterministic encoder. The default is False.
    linear_encoding : bool, optional
        Whether to use linear encoder. The default is True.
    init_weights : ndarray, optional
        Initial weights for linear encoder. The default is None.
    device : str, optional
        Device to use. The default is 'cuda:0'.
    critic_params_YX : dict, optional
        Parameters for critic function for I_YX. A dictionary with keys:
    predictive_space : str, optional
        Predictive space, either 'latent' or 'observation'. The default is 'latent'.
    regularization_weight : float, optional
        Weight for regularization term. The default is 0.

    Attributes
    ----------
    encoder : nn.Module
        Encoder network.
    critic : nn.Module
        Critic network for I_predictive.
    baseline : nn.Module
        Baseline network for I_predictive.
    critic_YX : nn.Module
        Critic network for I_YX.
    mi_params : dict
        Parameters for mutual information estimation.
    device : str
        Device to use.
    regularization_weight : float
        Weight for regularization term.
    """
    def __init__(self, xdim, ydim, mi_params, critic_params, baseline_params, T=4, beta=1e-3, beta1=1, beta2=1, hidden_dim=256,
                 deterministic=False, linear_encoding=True, init_weights=None, device='cuda:0', critic_params_YX=None, predictive_space="latent",
                 regularization_weight=0):
        super(CPIC, self).__init__()

        self.predictive_space = predictive_space
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.xdim = xdim
        self.ydim = ydim
        self.T = T
        self.deterministic = deterministic
        self.linear_encoding = linear_encoding
        self.encoder = StructuredEncoder(input_dim=xdim, output_dim=ydim, hidden_dim=hidden_dim, T=self.T, deterministic=deterministic, device=device, linear_encoding=linear_encoding)
        self.encoder.to(device)
        # initialize critic and baseline for I_compress, I_predictive
        if init_weights is not None:
            self.encoder._mean.weight = torch.nn.parameter.Parameter(torch.from_numpy(init_weights.T).to(self.encoder._mean.weight.dtype).to(device))
        self.critic = CRITICS[mi_params.get('critic', 'concat')](**critic_params)
        self.critic.to(device)
        if mi_params.get('baseline', 'constant') == "constant":
            self.baseline = BASELINES[mi_params.get('baseline', 'constant')]()
        else:
            self.baseline = BASELINES[mi_params.get('baseline', 'constant')](input_dim=self.T * self.ydim, **baseline_params)
            self.baseline.to(device)
        # initialize critic for I_YX
        if self.beta2 > 0:
            self.critic_YX = CRITICS[mi_params.get('critic', 'concat')](**critic_params_YX)
            self.critic_YX.to(device)
        self.mi_params = mi_params
        self.device=device
        self.regularization_weight=regularization_weight

    def forward(self, x_past, x_future, debug=False):
        batch_size = x_past.shape[0]
        encoded_past_mean, encoded_past_vars = self.encoder(x_past)
        encoded_past = encoded_past_mean + torch.sqrt(encoded_past_vars) * \
                       torch.randn(*encoded_past_mean.size()).to(self.device)
        encoded_past_reshaped = encoded_past.reshape(batch_size, -1)
        encoded_future_mean, encoded_future_vars = self.encoder(x_future)
        encoded_future = encoded_future_mean + torch.sqrt(encoded_future_vars) * \
                       torch.randn(*encoded_future_mean.size()).to(self.device)
        encoded_future_reshaped = encoded_future.reshape(batch_size, -1)
        future_reshaped = x_future.reshape(batch_size, -1)

        if self.deterministic:
            I_compress_bound = torch.tensor([0]).to(self.device)
        else:
            I_compress_bound = estimate_mutual_information(self.mi_params['estimator_compress'], x_past,
                                                           encoded_past_reshaped, decoder=self.encoder, device=self.device)
        if self.predictive_space == "latent":
            I_predictive_bound = estimate_mutual_information(self.mi_params['estimator_predictive'],
                                                             encoded_past_reshaped,
                                                             encoded_future_reshaped, critic_fn=self.critic,
                                                             baseline_fn=self.baseline, device=self.device)
        elif self.predictive_space == "observation":
            I_predictive_bound = estimate_mutual_information(self.mi_params['estimator_predictive'],
                                                             encoded_past_reshaped,
                                                             future_reshaped, critic_fn=self.critic,
                                                             baseline_fn=self.baseline, device=self.device)
        else:
            raise ValueError('The predictive space is not specified.')

        if self.beta2 > 0:
            I_YX_bound = estimate_mutual_information("infonce_lower", encoded_past_reshaped,
                                                 future_reshaped, critic_fn=self.critic_YX, device=self.device)
            L = self.beta * I_compress_bound - self.beta1 * I_predictive_bound - self.beta2 * I_YX_bound
        else:
            L = self.beta * I_compress_bound - self.beta1 * I_predictive_bound

        if self.regularization_weight > 0:
            weight = self.encoder._mean.weight
            L = L + self.regularization_weight * torch.sum(torch.abs(weight)) / torch.norm(weight)
            # L = L + self.regularization_weight * torch.norm(weight, p='nuc') / torch.norm(weight)
            print(torch.sum(torch.abs(weight)) / torch.norm(weight))
            print(weight)
        # print(debug)
        if debug:
            estimate_mutual_information(self.mi_params['estimator_compress'], x_past, encoded_past_reshaped,
                                        decoder=self.encoder, device=self.device, debug=debug)
        return L, I_compress_bound, I_predictive_bound

    def encode(self, x):
        encoded_mean = self.encoder.get_mean(x)
        return encoded_mean

    def fit(self, X, T, encoder='linear', estimator='nongaussian'):
        # TO-DO
        pass

    def transform(self, X):
        # TO-DO
        pass
    
    def fit_transform(self, X, T, encoder='linear', estimator='nongaussian'):
        # TO-DO
        pass

    def score(self, X, Y):
        # TO-DO (score based on mutual information between X and Y for compresion complexity and predictive information)
        pass


def train_CPIC(beta, xdim, ydim, mi_params, critic_params, baseline_params, num_epochs, train_loader, T=4, signiture=22,
               deterministic=False, linear_encoding=True, init_weights=None, num_early_stop=0, device="cuda:0", lr=1e-4, beta1=1, beta2=0,
               critic_params_YX=None, predictive_space="latent", regularization_weight=0, return_mutual_information=False):
    model = CPIC(xdim, ydim, mi_params, critic_params, baseline_params, T=T, beta=beta, beta1=beta1, beta2=beta2,
                 deterministic=deterministic, linear_encoding=linear_encoding, init_weights=init_weights, device=device, critic_params_YX=critic_params_YX,
                 predictive_space=predictive_space, regularization_weight=regularization_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if init_weights is not None:
        do_init = True
        opt_init = torch.optim.Adam(list(model.critic.parameters()), lr=lr)
    else:
        do_init = False

    writer = SummaryWriter(log_dir="tensor_logs/{}".format(signiture))

    if num_early_stop > 0:
        curr_loss = np.infty

    # torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm.tqdm(range(num_epochs)):
        loss_by_epoch = []
        I_compress_bound_by_epoch = []
        I_predictive_bound_by_epoch = []

        for x_past_batch, x_future_batch in train_loader:
            x_past_batch = x_past_batch.to(torch.float).to(device)
            x_future_batch = x_future_batch.to(torch.float).to(device)
            loss, I_compress_bound, I_predictive_bound = model(x_past_batch, x_future_batch)
            # if torch.isnan(loss):
            #     model(x_past_batch, x_future_batch, debug=True)
            loss.backward()
            # check if gradients are nan
            grad_bool = True
            for name, param in model.named_parameters():
                if not torch.isfinite(param.grad).all():
                    print(epoch, name, torch.isfinite(param.grad).all())
                    grad_bool = False
                    break
            if not grad_bool:
                break
            if do_init and epoch < (num_epochs/4):
                opt_init.step()
                opt_init.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

            I_compress_bound_by_epoch.append(I_compress_bound.item())
            I_predictive_bound_by_epoch.append(I_predictive_bound.item())
            loss_by_epoch.append(loss.item())

        if num_early_stop > 0 and (epoch+1) % num_early_stop == 0:
            if np.mean(loss_by_epoch) < curr_loss:
                curr_loss = np.mean(loss_by_epoch)
            else:
                break

        writer.add_scalar("loss", np.mean(loss_by_epoch), global_step=epoch)
        writer.add_scalar("I_compress", np.mean(I_compress_bound_by_epoch), global_step=epoch)
        writer.add_scalar("I_predictive", np.mean(I_predictive_bound_by_epoch), global_step=epoch)

        print('epoch', epoch, 'loss', np.mean(loss_by_epoch), 'I_compress_bound', np.mean(I_compress_bound_by_epoch),
              'I_predictive_bound', np.mean(I_predictive_bound_by_epoch))

    if return_mutual_information:
        return model, np.mean(I_compress_bound_by_epoch), np.mean(I_predictive_bound_by_epoch)
    else:
        return model, np.mean(loss_by_epoch)
    