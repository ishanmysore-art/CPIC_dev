import torch
from torch import nn
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import os
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

   
def conv_encoder(input_dim, hidden_dim, output_dim, n_hidden_layers=0, activation='relu', T=None, kernel_size=3, stride=1, padding=1):
    '''
    2D convolutional encoder that convolves over features
        input_dim: input feature dimension (xdim)
        hidden_dim: num channels in hidden layers & output layer
        output_dim: output feature dimension (ydim)
        n_hidden_layers: number of hidden layers (there are 2 conv layers + 1 linear layer by default)
        activation: activation function (relu by default)
        T: time window    
        kernel_size: conv kernel size (applied over feature dimension with kernel_height)
        stride: conv stride
        padding: conv padding
    '''
    # input shape: (batch, 1, input_dim, length) - reshape_for_conv will handle this
    if activation == 'relu':    
        activation_f = nn.ReLU()
    
    # helper function to compute output dimension of a conv layer
    # will let us figure out the final size of input to the linear layer
    def conv_output_dim(input_dim, kernel_size, stride, padding):
        return (input_dim + (2 * padding) - kernel_size) // stride + 1
    
    conv_layers = []

    # first conv layer: in_channels=1, out_channels=hidden_dim
    # kernel_size=(kernel_size, 1) to convolve over feature dimension with kernel_height
    conv_layers.append(nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
    final_num_features = conv_output_dim(input_dim, kernel_size, stride, padding)

    # batchnorm regardless of T to ensure stability
    conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
    conv_layers.append(activation_f)
    
    for _ in range(n_hidden_layers):
        conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
        final_num_features = conv_output_dim(final_num_features, kernel_size, stride, padding)

        # batchnorm regardless of T to ensure stability
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)
    
    conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
    final_num_features = conv_output_dim(final_num_features, kernel_size, stride, padding)

    # dimension of the flattened output of the conv layers (before the linear layer)
    flattened_dim = hidden_dim * final_num_features
    if flattened_dim < 0:
        raise ValueError(f"flattened_dim is negative: {flattened_dim}")

    class ConvEncoder(nn.Module):
        def __init__(self, conv_layers, flattened_dim, output_dim):
            super().__init__()
            self.conv_seq = nn.Sequential(*conv_layers)
            self.flattened_dim = flattened_dim
            self.output_dim = output_dim
            self.linear = nn.Linear(self.flattened_dim, self.output_dim)

        def forward(self, x):
            output = self.conv_seq(x)

            if output.shape[2] * output.shape[1] != self.flattened_dim:
                raise ValueError(f"output.shape[2] * output.shape[1] != flattened_dim: {output.shape[2] * output.shape[1]} != {self.flattened_dim}")

            output = output.permute(0, 3, 1, 2)
            output = torch.flatten(output, start_dim=2)
            output = self.linear(output)

            # output shape: (batch, T, output_dim)
            # if T is 1, change output to (batch, output_dim)
            if output.shape[1] == 1:
                output = output.squeeze(1)
                
            return output

        ''' commented out things that were needed for mean pooling implementation. can be used later for testing & comparing with mean pooling
        //TODO: perhaps also add a flag to ConvEncoder class to toggle between flattening/mean pooling

        self.final_num_features = final_num_features

        mean pool over the channel dim
        output = torch.mean(output, dim=1) # now (batch, final_num_features, T)
        output = output.permute(0, 2, 1) # now (batch, T, final_num_features)
        '''
    
    return ConvEncoder(conv_layers, flattened_dim, output_dim)

def _extract_conv_layers(module):
    """extract all Conv2d layers from a module."""
    layers = []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            layers.append(m)
    return layers

def _visualize_kernel_layer(layer, layer_idx, save_dir, type='mean'):
    """visualize kernels for a single Conv2d layer."""
    kernels = layer.weight.detach().clone().cpu()
    kernels = kernels.mean(dim =1, keepdim=True)
    
    print(kernels.size())
    kernels = kernels - kernels.min()
    if kernels.max() != 0:
        kernels = torch.abs(kernels / kernels.max())
    
    filter_img = torchvision.utils.make_grid(kernels, nrow=8)
    # Take first channel only to get 2D array for colormap
    filter_img_2d = filter_img[0, :, :]
    plt.imshow(filter_img_2d, cmap='gist_gray')
    plt.colorbar()
    
    plt.title(f'{type} Layer {layer_idx} - {kernels.shape[0]} filters')
    
    plt.savefig(os.path.join(save_dir, f'{type}_kernel_layer_{layer_idx}.png'))
    plt.close()

def visualize_conv_kernels(model, save_dir=None):
    encoder = model.encoder
    if encoder.encoder_type != 'conv' or encoder.linear_encoding:
        raise ValueError(f"Encoder is not a conv encoder or is using linear encoding. encoder_type: {encoder.encoder_type}, linear_encoding: {encoder.linear_encoding}")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # visualize mean encoder layers
    mean_layers = _extract_conv_layers(encoder._mean)
    for idx, layer in enumerate(mean_layers):
        _visualize_kernel_layer(layer, idx, save_dir, 'mean')
    
    # visualize std encoder layers
    if not encoder.deterministic:
        std_layers = _extract_conv_layers(encoder._logvars)
        for idx, layer in enumerate(std_layers):
            _visualize_kernel_layer(layer, idx, save_dir, 'std')


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
    def __init__(
            self, 
            input_dim, hidden_dim, output_dim,
            T=4,
            device="cuda:0", 
            deterministic=False,
            linear_encoder=False,
            nonlinear_encoder_type="mlp",
            n_layers=1, activation='relu', 
            conv_kernel_size=3, conv_stride=1, conv_padding=1,
            ):
        super(StructuredEncoder, self).__init__()
        self.deterministic = deterministic
        self.linear_encoder = linear_encoder
        self.nonlinear_encoder_type = nonlinear_encoder_type

        if linear_encoder:
            self._mean = nn.Linear(input_dim, output_dim)
        else:
            if nonlinear_encoder_type == "mlp":
                self._mean = mlp(input_dim, hidden_dim, output_dim, n_layers, activation, T=None)
            elif nonlinear_encoder_type == "conv":
                self._mean = conv_encoder(input_dim, hidden_dim, output_dim, n_layers, activation, T=None,
                            kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            else:
                raise ValueError(f"Unknown nonlinear_encoder_type: {nonlinear_encoder_type}. Must be 'mlp' or 'conv'.")

        if deterministic:
            self._logvars = Zeros(device=device)
        else:
            if nonlinear_encoder_type == "mlp":
                self._logvars = mlp(input_dim, hidden_dim, output_dim, n_layers, activation, T=T)
            elif nonlinear_encoder_type == "conv":
                self._logvars = conv_encoder(input_dim, hidden_dim, output_dim, n_layers, activation, T=T,
                            kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            else:
                raise ValueError(f"Unknown nonlinear_encoder_type: {nonlinear_encoder_type}. Must be 'mlp' or 'conv'.")

    def forward(self, x):
        # handle input shape for conv vs mlp
        if self.nonlinear_encoder_type == 'conv' and not self.linear_encoder:
            # for conv, input should be (batch, 1, features, time))
            x_processed = self.reshape_for_conv(x)
        else:
            # for mlp or linear encoding, no shape transformation needed
            x_processed = x
            
        encoded_mean = self._mean(x_processed)
        if self.deterministic:
            encoded_vars = torch.exp(self._logvars(encoded_mean.shape))
        else:
            encoded_vars = torch.exp(self._logvars(x_processed))
            # encoded_vars = nn.functional.softplus(self._logvars(x_processed))
        return encoded_mean, encoded_vars

    def get_logvars(self, x):
        # handle input shape for conv vs mlp
        if self.nonlinear_encoder_type == 'conv' and not self.linear_encoder:
            x_processed = self.reshape_for_conv(x)
        else:
            # for mlp or linear, no shape transformation needed
            x_processed = x
        return self._logvars(x_processed)

    def get_mean(self, x):
        # handle input shape for conv vs mlp
        if self.nonlinear_encoder_type == 'conv' and not self.linear_encoder:
            x_processed = self.reshape_for_conv(x)
        else:
            x_processed = x
        return self._mean(x_processed)

    def reshape_for_conv(self, x):
        # takes in x as (batch, time, features) or (batch, features)
        # if x is (batch, features), add time dim of 1
        # return x_processed as (batch, 1, features, time) for conv
        if x.ndim == 2:
            x_processed = x.unsqueeze(1)
        else:
            x_processed = x
        
        x_processed = x_processed.permute(0, 2, 1)
        x_processed = x_processed.unsqueeze(1)
        
        return x_processed

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
        mi = infonce_upper_bound(scores, device=device)
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
