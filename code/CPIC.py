import torch
from torch import nn
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import dca as DCA


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


def mlp(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None):
    if activation == 'relu':
        activation_f = nn.ReLU()
    if T is None:
        layers = [nn.Linear(input_dim, hidden_dim), activation_f]
    else:
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(T), activation_f]
    for _ in range(n_layers):
        if T is None:
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_f]
        else:
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(T), activation_f]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


def conv_encoder(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None, kernel_size=3, stride=1, padding=1):
    '''
    1D convolutional encoder
    input_dim: input dim (num features)
    hidden_dim: hidden dim
    output_dim: output dim
    n_layers: number of layers
    activation: activation function (relu by default)
    T: time window
    kernel_size: conv kernel size
    stride: conv stride
    padding: conv padding
    '''
    if activation == 'relu':
        activation_f = nn.ReLU()
    
    layers = []
    
    # first conv layer
    layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding))
    if T is not None:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(activation_f)
    
    # additional conv layers, specified by n_layers
    for _ in range(n_layers):
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding))
        if T is not None:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_f)
    
    # global average pooling to reduce temporal dimension
    layers.append(nn.AdaptiveAvgPool1d(1))
    
    # final linear layer to get desired output dimension
    layers.append(nn.Flatten())
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)


class Zeros(nn.Module):
    def __init__(self, device="cuda:0"):
        super(Zeros, self).__init__()
        self.device = device

    def forward(self, output_dim):
        return torch.zeros(output_dim).to(device=self.device)


class SeparableCritic(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, embed_dim, n_layers=1, activation='relu', **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._g = mlp(x_dim, hidden_dim, embed_dim, n_layers, activation)
        self._h = mlp(y_dim, hidden_dim, embed_dim, n_layers, activation)

    def forward(self, x, y):
        x = x.view(-1, self.x_dim)
        y = y.view(-1, self.y_dim)
        x_h = self._h(x)  # Batchsize x 32
        y_g = self._g(y)  # Batchsize x 32
        scores = torch.matmul(x_h, torch.transpose(y_g, 0, 1)) #Each element i,j is a scalar in R. f(x, y)
        return scores


class ConcatCritic(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, n_layers=1, activation='relu', **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(x_dim+y_dim, hidden_dim, 1, n_layers, activation)

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Tile all possible combinations of x and y
        x_tiled = torch.tile(x[None, :],  (batch_size, 1, 1))
        y_tiled = torch.tile(y[:, None],  (1, batch_size, 1))
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.transpose(torch.reshape(scores, [batch_size, batch_size]), 1, 0)


class UnnormalizedBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, activation='relu', **extra_kwargs):
        super(UnnormalizedBaseline, self).__init__()
        # output is scalar score
        self.input_dim = input_dim
        self._f = mlp(input_dim, hidden_dim, 1, n_layers, activation)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        scores = self._f(x)
        return scores


class StructuredEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T=4, n_layers=1, activation='relu', deterministic=False, device="cuda:0", linear_encoding=True, 
                encoder_type='mlp', conv_kernel_size=3, conv_stride=1, conv_padding=1):
        super(StructuredEncoder, self).__init__()
        self.deterministic = deterministic
        self.encoder_type = encoder_type
        
        if linear_encoding:
            self._mean = nn.Linear(input_dim, output_dim)
        else:
            if encoder_type == 'mlp':
                self._mean = mlp(input_dim, hidden_dim, output_dim, n_layers, activation, T=None)
            elif encoder_type == 'conv':
                self._mean = conv_encoder(input_dim, hidden_dim, output_dim, n_layers, activation, T=None, 
                            kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'mlp' or 'conv'")
                
        if deterministic:
            self._logvars = Zeros(device=device)
        else:
            if encoder_type == 'mlp':
                self._logvars = mlp(input_dim, hidden_dim, output_dim, n_layers, activation, T=T)
            elif encoder_type == 'conv':
                self._logvars = conv_encoder(input_dim, hidden_dim, output_dim, n_layers, activation, T=T, 
                                kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'mlp' or 'conv'")

    def forward(self, x):
        # handle input shape for conv vs mlp
        if self.encoder_type == 'conv' and not self.linear_encoding:
            # for conv, input should be (batch, features, time)
            if len(x.shape) == 3:  # (batch, time, features)
                x_processed = x.transpose(1, 2)  # (batch, features, time)
            else:  # need to add time dim
                x_processed = x.unsqueeze(-1)  # (batch, features, 1)
        else:
            # for mlp or linear encoding, no shape transformation needed
            x_processed = x
            
        encoded_mean = self._mean(x_processed)
        if self.deterministic:
            encoded_vars = self._logvars(encoded_mean.shape)
        else:
            encoded_vars = torch.exp(self._logvars(x_processed))
            # encoded_vars = nn.functional.softplus(self._logvars(x_processed))
        return encoded_mean, encoded_vars

    def get_logvars(self, x):
        # handle input shape for conv vs mlp
        if self.encoder_type == 'conv' and not self.linear_encoding:
            if len(x.shape) == 3:  # (batch, time, features)
                x_processed = x.transpose(1, 2)  # (batch, features, time)
            else:  # need to add time dim
                x_processed = x.unsqueeze(-1)  # (batch, features, 1)
        else:
            # for mlp or linear, no shape transformation needed
            x_processed = x
        return self._logvars(x_processed)

    def get_mean(self, x):
        # handle input shape for conv vs mlp
        if self.encoder_type == 'conv' and not self.linear_encoding:
            if len(x.shape) == 3:  # (batch, time, features)
                x_processed = x.transpose(1, 2)  # (batch, features, time)
            else:  # (batch, features) - need to add time dim
                x_processed = x.unsqueeze(-1)  # (batch, features, 1)
        else:
            # For mlp or linear, no shape transformation needed
            x_processed = x
        return self._mean(x_processed)


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


CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic
}


BASELINES= {
    'constant': lambda: None,
    'unnormalized': UnnormalizedBaseline
}


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


def infonec_upper_bound(scores, device='cuda:0'):
    '''Bound from Van Den Oord and al. (2018)
    scores are either known log conditional distribution log p(y|x) or critic function f(x,y).
    '''
    mi = torch.mean(torch.diag(scores) - reduce_logmeanexp_nodiag(scores, dim=1, device=device))
    return mi


def infonce_lower_bound(scores):
    '''Bound from Van Den Oord and al. (2018)'''
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores,dim=1))
    k =scores.size()[0]
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
    Estimate variational lower/upper bounds on mutual information.
    :param estimator: string specifying estimator, one of: 'nwj', 'infonce_lower', 'infonce_upper', 'tuba', 'mine' and 'vub'
    :param x: [batch_size, dim_x] Tensor
    :param y: [batch_size, dim_y] Tensor
    :param critic_fn: callable that takes x and y as input and outputs critic scores
          output shape is a [batch_size, batch_size] matrix
    :param baseline_fn (optional): callable that takes y as input
          outputs a [batch_size]  or [batch_size, 1] vector
    :return: scalar estimate of mutual information
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
    if estimator == 'infonce_lower':
        mi = infonce_lower_bound(scores)
    elif estimator == "infonce_upper":
        mi = infonec_upper_bound(scores, device=device)
    elif estimator == "vub":
        mi = vub_upper_bound(decoded_mean, decoded_vars, device=device)
    elif estimator == "nwj":
        mi = nwj_lower_bound(scores, device=device)
    elif estimator == "mine":
        mi = mine_lower_bound(scores, device=device)
    elif estimator == "tuba":
        mi = tuba_lower_bound(scores, log_baseline, device=device)
    if debug:
        import pdb; pdb.set_trace()
        decoderscores(decoded_mean_reshaped, decoded_vars_reshaped, y, debug=debug)
    return mi


class CPIC(nn.Module):
    def __init__(self, xdim, ydim, mi_params, critic_params, baseline_params, T=4, beta=1e-3, beta1=1, beta2=1, hidden_dim=256,
                 deterministic=False, linear_encoding=True, init_weights=None, device='cuda:0', critic_params_YX=None, predictive_space="latent",
                 regularization_weight=0, encoder_type='mlp', conv_kernel_size=3, conv_stride=1, conv_padding=1):
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
        self.encoder = StructuredEncoder(input_dim=xdim, output_dim=ydim, hidden_dim=hidden_dim, T=self.T, deterministic=deterministic, device=device, linear_encoding=linear_encoding, 
                                        encoder_type=encoder_type, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, conv_padding=conv_padding)
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


def DCA_init(X, T, d, n_init=1, rng_or_seed=None):
    opt = DCA.DynamicalComponentsAnalysis(T=T, rng_or_seed=rng_or_seed)
    opt.estimate_data_statistics(X)
    opt.fit_projection(d=d, n_init=n_init)
    V_dca = opt.coef_
    return V_dca


def train_CPIC(beta, xdim, ydim, mi_params, critic_params, baseline_params, num_epochs, train_loader, T=4, signiture=22,
               deterministic=False, linear_encoding=True, init_weights=None, num_early_stop=0, device="cuda:0", lr=1e-4, beta1=1, beta2=0,
               critic_params_YX=None, predictive_space="latent", regularization_weight=0, return_mutual_information=False, encoder_type='mlp',
               conv_kernel_size=3, conv_stride=1, conv_padding=1):
    model = CPIC(xdim, ydim, mi_params, critic_params, baseline_params, T=T, beta=beta, beta1=beta1, beta2=beta2,
                 deterministic=deterministic, linear_encoding=linear_encoding, init_weights=init_weights, device=device, critic_params_YX=critic_params_YX,
                 predictive_space=predictive_space, regularization_weight=regularization_weight, encoder_type=encoder_type,
                 conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, conv_padding=conv_padding)
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


def Polynomial_expand(x):
    res = list()
    feature_dim = x.shape[-1]
    for i in range(feature_dim):
        res.append(x[..., i])
        for j in range(i, feature_dim):
            res.append(x[..., i] * x[..., j])
    return np.stack(res, axis=-1)


class PastFutureDataset(Dataset):
    def __init__(self, ts_list, window_size):
        """
        :param ts: a list of time series T_i x N
        :param window_size:
        """

        # if standardization:
        #     ts = (ts.T/ts.std(axis=1)).T
        past_ts = []
        future_ts = []
        for ts in ts_list:
            T, N = ts.shape
            for i in range(T-2*window_size):
                past_ts.append(ts[i:(i+window_size)])
                future_ts.append(ts[(i+window_size):(i+2*window_size)])
            self.past_ts = np.stack(past_ts)
            self.future_ts = np.stack(future_ts)

    def __len__(self):
        return len(self.past_ts)

    def __getitem__(self, idx):
        return self.past_ts[idx], self.future_ts[idx]


if __name__ == "__main__":
    pass