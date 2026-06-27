import torch
from torch import nn


"""
Encoders
"""

def MLP(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None):
    """
    Multi-layer perceptron. 
    Input (batch, T, input_dim) -> output (batch, T, output_dim).

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
    2D convolutional encoder that convolves over features (spatial dimension), the spatial data is 1D (e.g. EEG recordings)
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

    def get_filters(self, layer_idx=0):
        """
        Return Conv2d kernel weights for analysis / visualization.

        Parameters
        ----------
        layer_idx : int
            Index among Conv2d layers in self.conv_seq (0 = first convolution).

        Returns
        -------
        weight : np.ndarray
            Shape (out_channels, in_channels, kernel_h, kernel_w) for this encoder.
            kernel_w is 1 (time), kernel_h is the spatial feature kernel size.
        meta : dict
            padding, stride, kernel_size tuples as in the nn.Conv2d module.
        """
        conv2ds = [m for m in self.conv_seq if isinstance(m, nn.Conv2d)]
        if layer_idx < 0 or layer_idx >= len(conv2ds):
            raise ValueError(f"layer_idx must be in [0, {len(conv2ds) - 1}], got {layer_idx}")
        layer = conv2ds[layer_idx]
        w = layer.weight.detach().cpu().numpy()
        meta = {"padding": layer.padding, "stride": layer.stride, "kernel_size": layer.kernel_size}
        return w, meta


class ConvSpatialParticle2DEncoder(nn.Module):
    """
    Convolutional encoder over a per-timestep particle layout.
    Input is expected as flattened interleaved coordinates (x0, y0, ..., xN, yN),
    and is internally reshaped per timestep to (num_particles, 2).

    Input (batch, T, input_dim=2*num_particles) -> output (batch, T, output_dim).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=0, activation="relu", T=None, kernel_size_particles=5, kernel_size_coords=2, stride_particles=1, padding_particles=2):
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError(f"ConvSpatialParticle2DEncoder expects even input_dim (x/y pairs), got {input_dim}.")
        if kernel_size_coords not in (1, 2):
            raise ValueError("kernel_size_coords must be 1 or 2.")

        def conv_out_dim(length, kernel_size, stride, padding):
            return (length + (2 * padding) - kernel_size) // stride + 1

        self.num_particles = input_dim // 2
        self.output_dim = output_dim

        if activation == 'relu':
            activation_f = nn.ReLU()
        else:
            activation_f = nn.ReLU()

        conv_layers = []

        conv_layers.append(nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size_particles, kernel_size_coords), stride=(stride_particles, 1), padding=(padding_particles, 0)))
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)

        p_out = conv_out_dim(self.num_particles, kernel_size_particles, stride_particles, padding_particles)
        coord_out = 3 - kernel_size_coords # width 2 with no padding, stride 1

        for _ in range(n_layers):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size_particles, 1), stride=(stride_particles, 1), padding=(padding_particles, 0)))
            p_out = conv_out_dim(p_out, kernel_size_particles, stride_particles, padding_particles)

            conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
            conv_layers.append(activation_f)

        flattened_dim = hidden_dim * p_out * coord_out
        if flattened_dim <= 0:
            raise ValueError(f"flattened_dim must be positive, got {flattened_dim}")

        self.conv_seq = nn.Sequential(*conv_layers)
        self.flattened_dim = flattened_dim
        self.linear = nn.Linear(flattened_dim, output_dim)

    def forward(self, x):
        # x: (batch, T, 2*num_particles) or (batch, 2*num_particles)
        if x.ndim == 2:
            batch_size, feat_dim = x.shape
            time_len = 1
            x = x.view(batch_size, 1, feat_dim)
        elif x.ndim == 3:
            batch_size, time_len, feat_dim = x.shape
        else:
            raise ValueError(f"Expected x.ndim in {{2,3}}, got {x.ndim}")

        if feat_dim != self.num_particles * 2:
            raise ValueError(f"Expected feature dim {self.num_particles * 2}, got {feat_dim}.")

        x_2d = x.view(batch_size * time_len, self.num_particles, 2).unsqueeze(1) # (batch * T, 1, num_particles, 2)
        out = self.conv_seq(x_2d) # (batch * T, channels, hidden_dim, num_particles)
        out = out.flatten(start_dim=1) # (batch * T, channels * hidden_dim * num_particles)
        out = self.linear(out) # (batch * T, output_dim)
        out = out.view(batch_size, time_len, self.output_dim) # (batch, T, output_dim)

        # if T is 1, change output to (batch, output_dim)
        if out.shape[1] == 1:
            out = out.squeeze(1) 
        return out

    def get_filters(self, layer_idx=0):
        conv2ds = [m for m in self.conv_seq if isinstance(m, nn.Conv2d)]
        if layer_idx < 0 or layer_idx >= len(conv2ds):
            raise ValueError(f"layer_idx must be in [0, {len(conv2ds) - 1}], got {layer_idx}")
        layer = conv2ds[layer_idx]
        w = layer.weight.detach().cpu().numpy()
        meta = {"padding": layer.padding, "stride": layer.stride, "kernel_size": layer.kernel_size}
        return w, meta


class ConvPhysicalEncoder(nn.Module):
    """
    Convolutional encoder over a 2D physical occupancy grid.

    At each timestep, interleaved particle coordinates [x0,y0,...,xN,yN] are binned
    onto a (grid_size x grid_size) count grid over [-spatial_bounds, spatial_bounds]^2.
    Standard symmetric 2D convolutions are applied over this spatial grid, giving
    filters that are directly interpretable as spatial detectors in physical space.

    Input (batch, T, input_dim=2*num_particles) -> output (batch, T, output_dim).

    Parameters
    ----------
    input_dim : int
        Must be even (2 * num_particles).
    hidden_dim : int
        Conv channel count and output projection width.
    output_dim : int
        Latent dimension.
    n_layers : int
        Additional hidden conv layers (total = n_layers + 1).
    activation : str
        Activation function ('relu').
    T : int, optional
        Accepted for API consistency; not used internally.
    grid_size : int
        H = W of the 2D spatial grid.
    spatial_bounds : float
        Binning range: [-spatial_bounds, spatial_bounds]^2. Should match the
        coordinate units of the encoder input (standardized by default).
    kernel_size : int
        Square conv kernel size applied symmetrically to both spatial axes.
    stride : int
        Conv stride (symmetric).
    padding : int
        Conv padding (symmetric).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=0, activation='relu',
                 T=None, grid_size=20, spatial_bounds=3.0, kernel_size=3, stride=1, padding=1):
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError(f"ConvPhysicalEncoder expects even input_dim (x/y pairs), got {input_dim}.")

        def conv_physical_output_dim(size, k, s, p):
            return (size + 2 * p - k) // s + 1

        self.num_particles = input_dim // 2
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spatial_bounds = spatial_bounds

        activation_f = nn.ReLU()

        conv_layers = []
        conv_layers.append(nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding)))
        conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
        conv_layers.append(activation_f)

        h_out = conv_physical_output_dim(grid_size, kernel_size, stride, padding)
        for _ in range(n_layers):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding)))
            conv_layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-5))
            conv_layers.append(activation_f)
            h_out = conv_physical_output_dim(h_out, kernel_size, stride, padding)

        flattened_dim = hidden_dim * h_out * h_out
        if flattened_dim <= 0:
            raise ValueError(f"flattened_dim={flattened_dim}. Reduce kernel_size or increase grid_size.")

        self.conv_seq = nn.Sequential(*conv_layers)
        self.flattened_dim = flattened_dim
        self.linear = nn.Linear(flattened_dim, output_dim)

    def forward(self, x):
        # x: (batch, T, 2*num_particles) or (batch, 2*num_particles)
        if x.ndim == 2:
            batch_size, feat_dim = x.shape
            time_len = 1
            x = x.view(batch_size, 1, feat_dim)
        elif x.ndim == 3:
            batch_size, time_len, feat_dim = x.shape
        else:
            raise ValueError(f"Expected x.ndim in {{2,3}}, got {x.ndim}")

        # Recover per-particle (x, y): (BT, N, 2)
        x_particles = x.view(batch_size * time_len, self.num_particles, 2)

        # Bin to [0, grid_size-1] integer indices
        coords = x_particles.clamp(-self.spatial_bounds, self.spatial_bounds)
        scale = self.grid_size / (2.0 * self.spatial_bounds)
        idx = ((coords + self.spatial_bounds) * scale).long().clamp(0, self.grid_size - 1)
        i_col = idx[..., 0]  # x -> column
        i_row = idx[..., 1]  # y -> row
        flat_idx = i_row * self.grid_size + i_col  # (BT, N)

        # Scatter particle counts into (BT, 1, H, W)
        BT = batch_size * time_len
        grid_flat = torch.zeros(BT, self.grid_size * self.grid_size, device=x.device, dtype=x.dtype) # (BT, H * W)
        ones = torch.ones(BT, self.num_particles, device=x.device, dtype=x.dtype) # (BT, N)
        grid_flat.scatter_add_(1, flat_idx, ones) # (BT, H * W)
        grid = grid_flat.view(BT, 1, self.grid_size, self.grid_size) # (BT, 1, H, W)

        out = self.conv_seq(grid) # (BT, channels, H, W)
        out = out.flatten(start_dim=1) # (BT, channels * H * W)
        out = self.linear(out) # (BT, output_dim)
        out = out.view(batch_size, time_len, self.output_dim) # (batch, T, output_dim)

        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    def get_filters(self, layer_idx=0):
        conv2ds = [m for m in self.conv_seq if isinstance(m, nn.Conv2d)]
        if layer_idx < 0 or layer_idx >= len(conv2ds):
            raise ValueError(f"layer_idx must be in [0, {len(conv2ds) - 1}], got {layer_idx}")
        layer = conv2ds[layer_idx]
        w = layer.weight.detach().cpu().numpy()
        meta = {
            "padding": layer.padding, "stride": layer.stride, "kernel_size": layer.kernel_size,
            "grid_size": self.grid_size, "spatial_bounds": self.spatial_bounds,
        }
        return w, meta


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


class FeatureMaskMLPEncoder(nn.Module):
    """
    Feature-wise masking followed by an MLP projection.
    Input (batch, T, input_dim) or (batch, input_dim) -> output (..., output_dim).

    The encoder learns (or uses fixed) per-feature gates and applies them
    multiplicatively before passing the masked input through a standard MLP.
    This keeps the representation model flexible while making feature
    importance explicit and easy to inspect via ``get_feature_mask()``.

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
    mask_learnable : bool, optional
        Whether to learn the mask
    mask_init : str, optional
        Initialization for the mask. "random" initializes logits ~ N(0,1).
    mask_init_values : torch.Tensor, optional
        Initialization values for the mask

    Attributes
    ----------
    input_dim : int
        Input feature dimension (xdim)
    mask_learnable : bool
        Whether to learn the mask
    mask_init : str
        Initialization for the mask
    mask_init_values : torch.Tensor
        Initialization values for the mask
    mask_logits : nn.Parameter
        Learnable logits (sigmoid -> mask probabilities in [0, 1])
    projector : nn.Module
        MLP projection
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, activation="relu", T=None, mask_learnable=True, mask_init="uniform", mask_init_values=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if mask_init == "random" and mask_init_values is None:
            # Random mode initializes the *logits* directly in unconstrained space. 
            # The resulting mask is sigmoid(logits) in (0,1) (i.e. continuous gates in [0, 1]).
            logits = torch.randn(input_dim, dtype=torch.float32)
        else:
            # For explicit init scores (PI or user-provided), values are treated as probabilities and mapped into logit space.
            if mask_init_values is not None:
                init_values = torch.as_tensor(mask_init_values, dtype=torch.float32)
                if init_values.numel() != input_dim:
                    raise ValueError(f"mask_init_values must have length {input_dim}, got {init_values.numel()}")
                init_values = init_values.clamp(0.0, 1.0)
            else:
                if mask_init in ("uniform", "pi"):
                    # Every gate starts at prob 0.5 (logit 0), so learning moves each
                    # feature purely by gradient (no random head start). "pi" falls back
                    # to this when explicit PI scores are not provided.
                    init_values = torch.full((input_dim,), 0.5, dtype=torch.float32)
                elif mask_init == "ones":
                    init_values = torch.ones(input_dim, dtype=torch.float32)
                else:
                    raise ValueError(f"Invalid mask_init: {mask_init}")
            # Clamp away from {0,1} to avoid infinite logits.
            logits = torch.logit(init_values.clamp(1e-4, 1.0 - 1e-4))
        self.mask_logits = nn.Parameter(logits, requires_grad=mask_learnable)
        self.projector = MLP(input_dim, hidden_dim, output_dim, n_layers=n_layers, activation=activation, T=T)

    def get_feature_mask(self):
        """
        Return a feature mask sampled from Bernoulli(sigmoid(logits))
        using a straight-through estimator so gradients flow through the underlying probabilities.

        Originally a sigmoid-only mask (deterministic soft gate), change to a Bernoulli-sampled mask (stochastic binary gate).
        """
        probs = torch.sigmoid(self.mask_logits)

        # At inference use a deterministic mask: sampling Bernoulli at eval time
        # injects gate noise into the latents (heavy test-time dropout when gates
        # sit near 0.5), which corrupts downstream probes. Threshold at 0.5 so a
        # confident gate -> 1/0 and the encoder reduces to a plain masked MLP.
        if not self.training:
            return (probs > 0.5).to(probs.dtype)

        # Hard binary sample
        hard = torch.bernoulli(probs)
        # Straight-through estimator: in forward use hard, in backward use probs.
        return hard + (probs - probs.detach())

    def forward(self, x):
        # Match mask dtype/device to input and broadcast across batch/time.
        mask = self.get_feature_mask().to(dtype=x.dtype, device=x.device)
        if x.ndim == 3:
            # x: (batch, time, features)
            x_masked = x * mask.view(1, 1, -1)
        elif x.ndim == 2:
            # x: (batch, features)
            x_masked = x * mask.view(1, -1)
        else:
            raise ValueError(f"Expected x.ndim in {{2,3}}, got {x.ndim}")

        # Project masked features into the latent/output space.
        return self.projector(x_masked)


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


def _conv_particle_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 0)
    activation = kwargs.get("activation", "relu")
    kernel_size_particles = kwargs.get("conv_kernel_size", 5)
    stride_particles = kwargs.get("conv_stride", 1)
    padding_particles = kwargs.get("conv_padding", 2)
    kernel_size_coords = kwargs.get("conv_coord_kernel_size", 2)
    return ConvSpatialParticle2DEncoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=n_layers,
        activation=activation,
        T=T,
        kernel_size_particles=kernel_size_particles,
        stride_particles=stride_particles,
        padding_particles=padding_particles,
        kernel_size_coords=kernel_size_coords,
    )


def _conv_physical_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    return ConvPhysicalEncoder(
        input_dim, hidden_dim, output_dim,
        n_layers=kwargs.get("n_layers", 0),
        activation=kwargs.get("activation", "relu"),
        T=T,
        grid_size=kwargs.get("grid_size", 20),
        spatial_bounds=kwargs.get("spatial_bounds", 3.0),
        kernel_size=kwargs.get("conv_kernel_size", 3),
        stride=kwargs.get("conv_stride", 1),
        padding=kwargs.get("conv_padding", 1),
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


def _mask_mlp_encoder_factory(input_dim, hidden_dim, output_dim, T=None, **kwargs):
    n_layers = kwargs.get("n_layers", 1)
    activation = kwargs.get("activation", "relu")
    return FeatureMaskMLPEncoder(
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=n_layers,
        activation=activation,
        T=T,
        mask_learnable=kwargs.get("mask_learnable", True),
        mask_init=kwargs.get("mask_init", "ones"),
        mask_init_values=kwargs.get("mask_init_values", None),
    )


# Registry of encoder factories. Use encoder_type in encoder_params when building CPIC, e.g.:
#   encoder_params = {"encoder_type": "mlp"}
#   encoder_params = {"encoder_type": "mlp2"}
#   encoder_params = {"encoder_type": "conv_spatial", "conv_kernel_size": 3}
#   encoder_params = {"encoder_type": "conv_particle", "conv_kernel_size": 5}
#   encoder_params = {"encoder_type": "conv_physical", "grid_size": 20, "spatial_bounds": 3.0}
#   encoder_params = {"encoder_type": "conv_spatiotemporal", "kernel_size_feat": 3, "kernel_size_time": 3}
#   encoder_params = {"encoder_type": "conv_temporal", "kernel_size": 3}
#   encoder_params = {"encoder_type": "mask_mlp", "mask_learnable": True, "mask_init": "ones", "mask_init_values": None}
ENCODERS = {
    "linear": _linear_encoder_factory,
    "mlp": _mlp_encoder_factory,
    "mlp2": _mlp_x2_encoder_factory,
    "conv_spatial": _conv_spatial_encoder_factory,
    "conv_particle": _conv_particle_encoder_factory,
    "conv_physical": _conv_physical_encoder_factory,
    "conv_spatiotemporal": _conv_spatiotemporal_encoder_factory,
    "conv_temporal": _conv_temporal_encoder_factory,
    "mask_mlp": _mask_mlp_encoder_factory,
}

ENCODER_INPUT_SHAPE = {
    "linear": "flat",
    "mlp": "flat",
    "mlp2": "flat",
    "conv_spatial": "conv",
    "conv_particle": "flat",
    "conv_physical": "flat",
    "conv_spatiotemporal": "conv",
    "conv_temporal": "conv",
    "mask_mlp": "flat",
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

    def get_filters(self, layer_idx=0):
        """
        For Conv-based mean encoders; return first (or indexed) convolutional kernel weights.
        """
        if hasattr(self._mean, "get_filters"):
            return self._mean.get_filters(layer_idx=layer_idx)
        raise NotImplementedError(f"get_filters is not implemented for encoder type {self.encoder_type!r}")

    def get_feature_mask(self):
        """Return the feature mask for mask-capable mean encoders."""
        if hasattr(self._mean, "get_feature_mask"):
            return self._mean.get_feature_mask()
        raise NotImplementedError(f"get_feature_mask is not implemented for encoder type {self.encoder_type!r}")

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
