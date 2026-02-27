import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import os


"""
Encoders
"""

def mlp(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None):
    '''
    Multi-layer perceptron encoder.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    n_layers : int, optional
        Number of layers (there is 1 linear layer by default)
    activation : str, optional
        Activation function ('relu' by default)
    T : int, optional
        Time window    
    '''
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

   
def conv_spatial_encoder(input_dim, hidden_dim, output_dim, n_hidden_layers=0, activation='relu', T=None, kernel_size=3, stride=1, padding=1):
    '''
    2D convolutional encoder that convolves over features (spatial dimension)

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    n_hidden_layers : int, optional
        Number of hidden layers (there are 2 conv layers + 1 linear layer by default)
    activation : str, optional
        Activation function ('relu' by default)
    T : int, optional
        Time window    
    kernel_size : int, optional
        Convolutional kernel size (applied over feature dimension with kernel_height)
    stride : int, optional
        Convolutional stride
    padding : int, optional
        Convolutional padding
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
    
    conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))) # REMOVE?
    final_num_features = conv_output_dim(final_num_features, kernel_size, stride, padding)

    # dimension of the flattened output of the conv layers (before the linear layer)
    flattened_dim = hidden_dim * final_num_features
    if flattened_dim < 0:
        raise ValueError(f"flattened_dim is negative: {flattened_dim}")

    class ConvSpatialEncoder(nn.Module):
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
        //TODO: perhaps also add a flag to ConvSpatialEncoder class to toggle between flattening/mean pooling

        self.final_num_features = final_num_features

        mean pool over the channel dim
        output = torch.mean(output, dim=1) # now (batch, final_num_features, T)
        output = output.permute(0, 2, 1) # now (batch, T, final_num_features)
        '''
    
    return ConvSpatialEncoder(conv_layers, flattened_dim, output_dim)


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
    
    plt.title(f'{type} Layer {layer_idx} - {kernels.shape[0]} filters')
    
    plt.savefig(os.path.join(save_dir, f'{type}_kernel_layer_{layer_idx}.png'))
    plt.close()


def visualize_conv_kernels(model, save_dir=None):
    encoder = model.encoder
    
    encoder_type = getattr(encoder, "encoder_type", None)
    is_linear = getattr(encoder, "linear_encoder", False)
    if encoder_type not in ("conv", "conv_spatial") or is_linear:
        raise ValueError(
            f"Encoder is not a conv encoder or is using linear encoding. "
            f"encoder_type: {encoder_type}, linear_encoder: {is_linear}"
        )

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


"""
Encoder registry
"""

def _linear_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    # ignore hidden_dim, T and extra kwargs for a purely linear encoder
    return nn.Linear(input_dim, output_dim)


def _mlp_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 1)
    activation = kwargs.get("activation", "relu")
    return mlp(input_dim, hidden_dim, output_dim, n_layers=n_layers, activation=activation, T=T)


def _mlp_x2_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    activation = kwargs.get("activation", "relu")
    return mlp(input_dim, hidden_dim, output_dim, n_layers=2, activation=activation, T=T)


def _conv_spatial_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 0)
    activation = kwargs.get("activation", "relu")
    kernel_size = kwargs.get("conv_kernel_size", 3)
    stride = kwargs.get("conv_stride", 1)
    padding = kwargs.get("conv_padding", 1)
    return conv_spatial_encoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=n_layers,
        activation=activation,
        T=T,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def conv_spatiotemporal_encoder(
    input_dim,
    hidden_dim,
    output_dim,
    n_hidden_layers=0,
    activation="relu",
    T=None,
    kernel_size_feat=3,
    kernel_size_time=3,
    stride_feat=1,
    stride_time=1,
    padding_feat=1,
    padding_time=1,
):
    """
    2D convolutional encoder that convolves over both feature and time dimensions.
    Input (batch, 1, input_dim, T) -> output (batch, T, output_dim). Preserves T via padding.
    """
    if activation == "relu":
        activation_f = nn.ReLU()

    def conv2d_output_size(in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1
        return out_h, out_w

    conv_layers = []
    conv_layers.append(
        nn.Conv2d(
            1,
            hidden_dim,
            kernel_size=(kernel_size_feat, kernel_size_time),
            stride=(stride_feat, stride_time),
            padding=(padding_feat, padding_time),
        )
    )
    conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
    conv_layers.append(activation_f)
    in_T = 1 if T is None else max(1, T)
    h_out, w_out = conv2d_output_size(
        input_dim, in_T,
        kernel_size_feat, kernel_size_time,
        stride_feat, stride_time,
        padding_feat, padding_time,
    )
    for _ in range(n_hidden_layers):
        conv_layers.append(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(kernel_size_feat, kernel_size_time),
                stride=(stride_feat, stride_time),
                padding=(padding_feat, padding_time),
            )
        )
        h_out, w_out = conv2d_output_size(
            h_out, w_out,
            kernel_size_feat, kernel_size_time,
            stride_feat, stride_time,
            padding_feat, padding_time,
        )
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)
    conv_layers.append(
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(kernel_size_feat, kernel_size_time),
            stride=(stride_feat, stride_time),
            padding=(padding_feat, padding_time),
        )
    )
    h_out, w_out = conv2d_output_size(
        h_out, w_out,
        kernel_size_feat, kernel_size_time,
        stride_feat, stride_time,
        padding_feat, padding_time,
    )

    flattened_dim = hidden_dim * h_out * w_out
    if flattened_dim <= 0:
        raise ValueError(f"flattened_dim is non-positive: {flattened_dim}")

    class ConvSpatiotemporalEncoder(nn.Module):
        def __init__(self, conv_layers, flattened_dim, output_dim, out_T):
            super().__init__()
            self.conv_seq = nn.Sequential(*conv_layers)
            self.flattened_dim = flattened_dim
            self.output_dim = output_dim
            self.out_T = out_T
            self.linear = nn.Linear(flattened_dim, output_dim * out_T)

        def forward(self, x):
            out = self.conv_seq(x)
            B, C, H, W = out.shape
            out = out.permute(0, 2, 3, 1).reshape(B, -1)
            out = self.linear(out)
            out = out.view(B, self.out_T, self.output_dim)
            if out.shape[1] == 1:
                out = out.squeeze(1)
            return out

    out_T = in_T if T is not None else max(1, w_out)
    return ConvSpatiotemporalEncoder(conv_layers, flattened_dim, output_dim, out_T)


def _conv_spatiotemporal_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 0)
    activation = kwargs.get("activation", "relu")
    k = kwargs.get("conv_kernel_size", 3)
    s = kwargs.get("conv_stride", 1)
    p_feat = kwargs.get("conv_padding", 1)
    p_time = (k - 1) // 2
    return conv_spatiotemporal_encoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=n_layers,
        activation=activation,
        T=T,
        kernel_size_feat=k,
        kernel_size_time=k,
        stride_feat=s,
        stride_time=1,
        padding_feat=p_feat,
        padding_time=p_time,
    )


class TemporalConv1dEncoder(nn.Module):
    """
    1D temporal convolution along the time axis. Input (batch, T, D) -> output (batch, T, M).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, n_layers=1, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        padding = kernel_size // 2
        act = nn.ReLU() if activation == "relu" else nn.ReLU()
        layers = []
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding))
        layers.append(act)
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding))
            layers.append(act)
        self.conv_seq = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, T, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv_seq(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


def _conv1d_temporal_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    kernel_size = kwargs.get("kernel_size_1d", kwargs.get("conv_kernel_size", 3))
    n_layers = kwargs.get("n_layers", 1)
    activation = kwargs.get("activation", "relu")
    return TemporalConv1dEncoder(
        input_dim, hidden_dim, output_dim,
        kernel_size=kernel_size,
        n_layers=n_layers,
        activation=activation,
    )


class AttentionEncoder(nn.Module):
    """
    Self-attention over the time dimension. Input (batch, T, D) -> output (batch, T, M).
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        dim_feedforward=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        dim_feedforward = dim_feedforward or (4 * hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.output_proj(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


def _attention_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    num_heads = kwargs.get("num_heads", 4)
    num_layers = kwargs.get("num_layers", 2)
    dropout = kwargs.get("dropout", 0.1)
    dim_feedforward = kwargs.get("dim_feedforward", None)
    return AttentionEncoder(
        input_dim,
        hidden_dim,
        output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
    )


# Registry of encoder factories. Use encoder_type in encoder_params when building CPIC, e.g.:
#   encoder_params = {"encoder_type": "mlp", "n_layers": 1, "deterministic": False}
#   encoder_params = {"encoder_type": "mlp2", "deterministic": False}
#   encoder_params = {"encoder_type": "conv_spatial", "conv_kernel_size": 3}
#   encoder_params = {"encoder_type": "conv_spatiotemporal", "conv_kernel_size": 3}
#   encoder_params = {"encoder_type": "conv1d_temporal", "kernel_size_1d": 3}
#   encoder_params = {"encoder_type": "attention", "num_heads": 4, "num_layers": 2}
ENCODERS = {
    "linear": _linear_encoder_factory,
    "mlp": _mlp_encoder_factory,
    "mlp2": _mlp_x2_encoder_factory,
    "mlp_x2": _mlp_x2_encoder_factory,
    "conv": _conv_spatial_encoder_factory,
    "conv_spatial": _conv_spatial_encoder_factory,
    "conv_spatiotemporal": _conv_spatiotemporal_encoder_factory,
    "conv1d_temporal": _conv1d_temporal_encoder_factory,
    "attention": _attention_encoder_factory,
}

ENCODER_INPUT_SHAPE = {
    "linear": "flat",
    "mlp": "flat",
    "mlp2": "flat",
    "mlp_x2": "flat",
    "conv": "conv2d",
    "conv_spatial": "conv2d",
    "conv_spatiotemporal": "conv2d",
    "conv1d_temporal": "flat",
    "attention": "flat",
}


class StructuredEncoder(nn.Module):
    def __init__(
            self, 
            input_dim, hidden_dim, output_dim,
            T=4,
            device="cuda:0", 
            deterministic=False,
            encoder_type=None,
            linear_encoder=True,
            nonlinear_encoder_type="mlp",
            n_layers=1,
            activation='relu',
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            **extra_encoder_kwargs,
            ):
        super(StructuredEncoder, self).__init__()
        self.deterministic = deterministic

        if encoder_type is None:
            # linear vs nonlinear MLP/conv controlled by flags
            if linear_encoder:
                encoder_type = "linear"
            else:
                encoder_type = nonlinear_encoder_type

        # pack encoder-specific kwargs; factories will pick what they need (ADD as we add more encoder types)
        encoder_kwargs = {
            "n_layers": n_layers,
            "activation": activation,
            "conv_kernel_size": conv_kernel_size,
            "conv_stride": conv_stride,
            "conv_padding": conv_padding,
        }
        encoder_kwargs.update(extra_encoder_kwargs)

        if encoder_type not in ENCODERS:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Available types: {list(ENCODERS.keys())}")

        self.encoder_type = encoder_type
        # keep a linear_encoder flag for downstream code and visualisation
        self.linear_encoder = encoder_type == "linear"
        self._input_shape = ENCODER_INPUT_SHAPE.get(encoder_type, "flat")

        factory = ENCODERS[encoder_type]
        # mean encoder does not depend on T
        self._mean = factory(input_dim, hidden_dim, output_dim, T=None, **encoder_kwargs)
        if deterministic:
            self._logvars = Zeros(device=device)
        else:
            self._logvars = factory(input_dim, hidden_dim, output_dim, T=T, **encoder_kwargs)

    def forward(self, x):
        # handle input shape for conv vs mlp
        if self._input_shape == "conv2d" and not self.linear_encoder:
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
        if self._input_shape == "conv2d" and not self.linear_encoder:
            x_processed = self.reshape_for_conv(x)
        else:
            # for mlp or linear, no shape transformation needed
            x_processed = x
        return self._logvars(x_processed)

    def get_mean(self, x):
        # handle input shape for conv vs mlp
        if self._input_shape == "conv2d" and not self.linear_encoder:
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


"""
Critics
"""

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
        scores = torch.matmul(x_h, torch.transpose(y_g, 0, 1)) # Each element i,j is a scalar in R. f(x, y)
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


CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic
}


"""
Baselines
"""

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


BASELINES = {
    'constant': lambda: None,
    'unnormalized': UnnormalizedBaseline
}
