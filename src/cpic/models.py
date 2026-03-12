import torch
from torch import nn


"""
Encoders
"""

def MLP(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None):
    """
    Multi-layer perceptron. 
    Input (batch, 1, input_dim, T) -> output (batch, T, output_dim).

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
    """
    if activation == 'relu':
        activation_f = nn.ReLU()
    else:
        activation_f = nn.ReLU() # add other activation functions?
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


class ConvSpatialEncoder(nn.Module):
    """
    2D convolutional encoder that convolves over features (spatial dimension)
    Input (batch, 1, input_dim, T) -> output (batch, T, output_dim).

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    n_layers : int, optional
        Number of hidden layers
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
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=0, activation='relu', T=None, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # helper function to compute output dimension of a conv layer
        # will let us figure out the final size of input to the linear layer
        def conv_spatial_output_dim(input_dim, kernel_size, stride, padding):
            return (input_dim + (2 * padding) - kernel_size) // stride + 1

        if activation == 'relu':
            activation_f = nn.ReLU()
        else:
            activation_f = nn.ReLU() # add other activation functions?
        
        conv_layers = []

        # first conv layer: in_channels=1, out_channels=hidden_dim
        # kernel_size=(kernel_size, 1) to convolve over feature dimension with kernel_height
        conv_layers.append(nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
        final_num_features = conv_spatial_output_dim(input_dim, kernel_size, stride, padding)

        # batchnorm regardless of T to ensure stability
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)
        
        for _ in range(n_layers):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)))
            final_num_features = conv_spatial_output_dim(final_num_features, kernel_size, stride, padding)

            # batchnorm regardless of T to ensure stability
            conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
            conv_layers.append(activation_f)
        
        # dimension of the flattened output of the conv layers (before the linear layer)
        flattened_dim = hidden_dim * final_num_features
        if flattened_dim < 0:
            raise ValueError(f"flattened_dim is negative: {flattened_dim}")

        self.conv_seq = nn.Sequential(*conv_layers)
        self.flattened_dim = flattened_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.flattened_dim, self.output_dim)

    def forward(self, x):
        # input shape: (batch, 1, input_dim, T) - reshape_for_conv will handle this
        out = self.conv_seq(x)

        if out.shape[2] * out.shape[1] != self.flattened_dim:
            raise ValueError(f"output.shape[2] * output.shape[1] != flattened_dim: {out.shape[2] * out.shape[1]} != {self.flattened_dim}")

        out = out.permute(0, 3, 1, 2) # (batch, channels, hidden_dim, T) -> (batch, T, channels, hidden_dim)
        out = torch.flatten(out, start_dim=2) # (batch, T, channels, hidden_dim) -> (batch, T, channels * hidden_dim)
        out = self.linear(out) # output shape: (batch, T, output_dim)

        # if T is 1, change output to (batch, output_dim)
        if out.shape[1] == 1:
            out = out.squeeze(1)
            
        return out


class ConvSpatiotemporalEncoder(nn.Module):
    """
    2D convolutional encoder that convolves over both feature and time dimensions.
    Input (batch, 1, input_dim, T) -> output (batch, T, output_dim). Preserves T via padding.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    n_layers : int, optional
        Number of hidden layers
    activation : str, optional
        Activation function ('relu' by default)
    T : int, optional
        Time window    
    kernel_size_feat : int, optional
        Convolutional kernel size for feature dimension
    kernel_size_time : int, optional
        Convolutional kernel size for time dimension
    stride_feat : int, optional
        Convolutional stride for feature dimension
    stride_time : int, optional
        Convolutional stride for time dimension
    padding_feat : int, optional
        Convolutional padding for feature dimension
    padding_time : int, optional
        Convolutional padding for time dimension
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=0, activation="relu", T=None,
                 kernel_size_feat=3, kernel_size_time=3, stride_feat=1, stride_time=1, padding_feat=1, padding_time=1):
        super().__init__()

        def conv_spatiotemporal_output_dim(in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
            out_h = (in_h + 2 * p_h - k_h) // s_h + 1
            out_w = (in_w + 2 * p_w - k_w) // s_w + 1
            return out_h, out_w

        if activation == "relu":
            activation_f = nn.ReLU()

        conv_layers = []

        conv_layers.append(nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size_feat, kernel_size_time), stride=(stride_feat, stride_time), padding=(padding_feat, padding_time)))
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)

        in_T = 1 if T is None else max(1, T)
        h_out, w_out = conv_spatiotemporal_output_dim(input_dim, in_T, kernel_size_feat, kernel_size_time, stride_feat, stride_time, padding_feat, padding_time)

        for _ in range(n_layers):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size_feat, kernel_size_time), stride=(stride_feat, stride_time), padding=(padding_feat, padding_time)))
            h_out, w_out = conv_spatiotemporal_output_dim(h_out, w_out, kernel_size_feat, kernel_size_time, stride_feat, stride_time, padding_feat, padding_time)
            
            conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
            conv_layers.append(activation_f)

        flattened_dim = hidden_dim * h_out * w_out
        if flattened_dim < 0:
            raise ValueError(f"flattened_dim is negative: {flattened_dim}")

        out_T = in_T if T is not None else max(1, w_out)
        
        self.conv_seq = nn.Sequential(*conv_layers)
        self.flattened_dim = flattened_dim
        self.output_dim = output_dim
        self.out_T = out_T
        self.linear = nn.Linear(flattened_dim, output_dim * out_T)

    def forward(self, x):
        # input shape: (batch, 1, input_dim, T)
        out = self.conv_seq(x)
        B, C, H, W = out.shape
        out = torch.flatten(out, start_dim=1) # (batch, channels, hidden_dim, T) -> (batch, channels * hidden_dim * T) = (B, flattened_dim)
        out = self.linear(out) # (batch, output_dim * out_T)
        out = out.view(B, self.out_T, self.output_dim) # output shape: (batch, T, output_dim)

        # if T is 1, change output to (batch, output_dim)
        if out.shape[1] == 1:
            out = out.squeeze(1)

        return out
    

class ConvTemporalEncoder(nn.Module):
    """
    1D temporal convolution along the time axis. 
    Input (batch, 1, input_dim, T) -> output (batch, T, output_dim).

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    kernel_size : int, optional
        Convolutional kernel size
    n_layers : int, optional
        Number of hidden layers
    activation : str, optional
        Activation function ('relu' by default)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, n_layers=1, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        padding = kernel_size // 2

        activation_f = nn.ReLU() if activation == "relu" else nn.ReLU()

        layers = []

        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding))
        layers.append(activation_f)

        for _ in range(n_layers):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding))
            layers.append(activation_f)

        self.conv_seq = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # input shape: (batch, 1, features, time)
        # Conv1d wants (batch, features (channels), time), so
        x = x.squeeze(1)  # (batch, 1, D, T) -> (batch, D, T)

        out = self.conv_seq(x) # (batch, hidden_dim, T)
        out = out.permute(0, 2, 1) # (batch, T, hidden_dim) for linear
        out = self.linear(out) # output shape: (batch, T, output_dim)

        # if T is 1, change output to (batch, output_dim)
        if out.shape[1] == 1:
            out = out.squeeze(1)

        return out


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
    return MLP(input_dim, hidden_dim, output_dim, n_layers=n_layers, activation=activation, T=T)


def _mlp_x2_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    activation = kwargs.get("activation", "relu")
    return MLP(input_dim, hidden_dim, output_dim, n_layers=2, activation=activation, T=T)


def _conv_spatial_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 0)
    activation = kwargs.get("activation", "relu")
    kernel_size = kwargs.get("conv_kernel_size", 3)
    stride = kwargs.get("conv_stride", 1)
    padding = kwargs.get("conv_padding", 1)
    return ConvSpatialEncoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=n_layers,
        activation=activation,
        T=T,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def _conv_spatiotemporal_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 0)
    activation = kwargs.get("activation", "relu")
    k = kwargs.get("conv_kernel_size", 3)
    s = kwargs.get("conv_stride", 1)
    p_feat = kwargs.get("conv_padding", 1)
    p_time = (k - 1) // 2
    return ConvSpatiotemporalEncoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=n_layers,
        activation=activation,
        T=T,
        kernel_size_feat=k,
        kernel_size_time=k,
        stride_feat=s,
        stride_time=1,
        padding_feat=p_feat,
        padding_time=p_time,
    )


def _conv_temporal_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    kernel_size = kwargs.get("kernel_size_1d", kwargs.get("conv_kernel_size", 3))
    n_layers = kwargs.get("n_layers", 1)
    activation = kwargs.get("activation", "relu")
    return ConvTemporalEncoder(
        input_dim, hidden_dim, output_dim,
        kernel_size=kernel_size,
        n_layers=n_layers,
        activation=activation,
    )

# Registry of encoder factories. Use encoder_type in encoder_params when building CPIC, e.g.:
#   encoder_params = {"encoder_type": "mlp"}
#   encoder_params = {"encoder_type": "mlp2"}
#   encoder_params = {"encoder_type": "conv_spatial", "conv_kernel_size": 3}
#   encoder_params = {"encoder_type": "conv_spatiotemporal", "kernel_size_feat": 3, "kernel_size_time": 3}
#   encoder_params = {"encoder_type": "conv_temporal", "kernel_size": 3}
ENCODERS = {
    "linear": _linear_encoder_factory,
    "mlp": _mlp_encoder_factory,
    "mlp2": _mlp_x2_encoder_factory,
    "conv_spatial": _conv_spatial_encoder_factory,
    "conv_spatiotemporal": _conv_spatiotemporal_encoder_factory,
    "conv_temporal": _conv_temporal_encoder_factory,
}

ENCODER_INPUT_SHAPE = {
    "linear": "flat",
    "mlp": "flat",
    "mlp2": "flat",
    "conv_spatial": "conv",
    "conv_spatiotemporal": "conv",
    "conv_temporal": "conv",
}


class StructuredEncoder(nn.Module):
    """
    Structured encoder that combines multiple encoder types. 
    Encodes input (T x D) to output (T x M) using a specified encoder type.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (xdim)
    hidden_dim : int
        Number of channels in hidden layers & output layer
    output_dim : int
        Output feature dimension (ydim)
    T : int, optional
        Time window
    device : str, optional
        Device to use ('cuda:0' by default)
    deterministic : bool, optional
        Whether to use deterministic encoder. Default is False.
    encoder_type : str, optional
        Type of encoder. Default is 'mlp'.
    linear_encoder : bool, optional
        Whether to use linear encoder. Default is True.
    n_layers : int, optional
        Number of layers for nonlinear encoder. Default is 1.
    activation : str, optional
        Activation function for nonlinear encoder. Default is 'relu'.
    conv_kernel_size : int, optional
        Kernel size for convolutional encoder. Default is 3.
    conv_stride : int, optional
        Stride for convolutional encoder. Default is 1.
    conv_padding : int, optional
        Padding for convolutional encoder. Default is 1.
    **extra_encoder_kwargs : dict, optional
        Extra encoder-specific kwargs.

    Attributes
    ----------
    encoder_type : str
        Type of encoder.
    linear_encoder : bool
        Whether to use linear encoder.
    n_layers : int
        Number of layers for nonlinear encoder.
    activation : str
        Activation function for nonlinear encoder.
    conv_kernel_size : int
        Kernel size for convolutional encoder.
    conv_stride : int
        Stride for convolutional encoder.
    conv_padding : int
        Padding for convolutional encoder.
    _input_shape : str
        Input shape for encoder.
    _mean : nn.Module
        Mean encoder network.
    _logvars : nn.Module
        Variance encoder network.
    """
    def __init__(
            self, 
            input_dim, hidden_dim, output_dim,
            T=4,
            device="cuda:0", 
            deterministic=False,
            encoder_type="mlp",
            linear_encoder=True,
            n_layers=1,
            activation='relu',
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            **extra_encoder_kwargs,
            ):
        super(StructuredEncoder, self).__init__()
        self.deterministic = deterministic

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
        if encoder_type == "conv_spatiotemporal":
            self._mean = factory(input_dim, hidden_dim, output_dim, T=T, **encoder_kwargs)
        else:
            self._mean = factory(input_dim, hidden_dim, output_dim, T=None, **encoder_kwargs)
            
        if deterministic:
            self._logvars = Zeros(device=device)
        else:
            self._logvars = factory(input_dim, hidden_dim, output_dim, T=T, **encoder_kwargs)

    def forward(self, x):
        # handle input shape for conv vs mlp
        if self._input_shape == "conv" and not self.linear_encoder:
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
        if self._input_shape == "conv" and not self.linear_encoder:
            x_processed = self.reshape_for_conv(x)
        else:
            # for mlp or linear, no shape transformation needed
            x_processed = x
        return self._logvars(x_processed)

    def get_mean(self, x):
        # handle input shape for conv vs mlp
        if self._input_shape == "conv" and not self.linear_encoder:
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
        self._g = MLP(x_dim, hidden_dim, embed_dim, n_layers, activation)
        self._h = MLP(y_dim, hidden_dim, embed_dim, n_layers, activation)

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
        self._f = MLP(x_dim+y_dim, hidden_dim, 1, n_layers, activation)

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
        self._f = MLP(input_dim, hidden_dim, 1, n_layers, activation)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        scores = self._f(x)
        return scores


BASELINES = {
    'constant': lambda: None,
    'unnormalized': UnnormalizedBaseline
}
