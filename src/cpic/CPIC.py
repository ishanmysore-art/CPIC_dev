from errno import ECANCELED
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from .models import StructuredEncoder, CRITICS, BASELINES
from .mi import estimate_mutual_information
from .utils.helpers import visualize_conv_kernels


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
    ydim : int
        Dimensionality of the output data.
    T : int
        Length of the time window to compute PI.
    xdim : int, optional
        Dimensionality of the input data. If None, use the encoder to infer input dimension. The default is None.
    mi_params : dict, optional
        Parameters for mutual information estimation. A dictionary with keys:
            estimator_compress : str
                Estimator for I_compress. One of ['infonce_lower', 'nwj_lower', 'tuba_lower']. Default is 'infonce_lower'.
            estimator_predictive : str
                Estimator for I_predictive. One of ['infonce_lower', 'nwj_lower', 'tuba_lower']. Default is 'infonce_lower'.
            critic : str
                Critic type for mutual information estimation. One of ['separable', 'concat']. Default is 'concat'.
            baseline : str
                Baseline type for mutual information estimation. One of ['constant', 'unnormalized']. Default is 'constant'.
    critic_params : dict, optional
        Parameters for critic network for I_predictive. A dictionary with keys:
            x_dim : int
                Input dimension for critic. Default is T * ydim.
            y_dim : int
                Output dimension for critic. Default is T * ydim or T * xdim depending on predictive_space.
            hidden_dim : int
                Hidden dimension for critic. Default is 256.
    baseline_params : dict, optional
        Parameters for baseline network for I_predictive. A dictionary with keys:
            hidden_dim : int
                Hidden dimension for baseline. Default is 256.
    encoder_params : dict, optional
        Parameters for encoder. A dictionary with keys:
            deterministic : bool
                Whether to use deterministic encoder. Default is False.
            encoder_type : str
                Type of nonlinear encoder. Default is 'mlp'.
            linear_encoder : bool
                Whether to use linear encoder. Default is True.
            n_layers : int
                Number of layers for nonlinear encoder. Default is 1.
            activation : str
                Activation function for nonlinear encoder. One of ['relu', 'tanh', 'elu']. Default is 'relu'.
            conv_kernel_size : int
                Kernel size for convolutional encoder. Default is 3.
            conv_stride : int
                Stride for convolutional encoder. Default is 1.
            conv_padding : int
                Padding for convolutional encoder. Default is 1.
    critic_params_YX : dict, optional
        Parameters for critic network for I_YX. A dictionary with keys:
            x_dim : int
                Input dimension for critic.
            y_dim : int
                Output dimension for critic.
            hidden_dim : int
                Hidden dimension for critic.
    hidden_dim : int, optional
        Hidden dimension for encoder and critic networks. The default is 256.
    beta : float, optional
        Weight for compression term I_compress. The default is 1e-3.
    beta1 : float, optional
        Weight for predictive term I_predictive. The default is 1.
    beta2 : float, optional
        Weight for I_YX term. The default is 0.
    device : str, optional
        Device to use. The default is 'cuda:0'.
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

    def __init__(
            self,
            ydim,
            T,
            xdim=None, 
            mi_params=None,
            critic_params=None,
            baseline_params=None,
            encoder_params={},
            critic_params_YX=None,
            hidden_dim=256,
            beta=1e-3, beta1=1.0, beta2=0,
            device='cuda:0', 
            predictive_space="latent",
            regularization_weight=0
            ):
        super(CPIC, self).__init__()

        self.predictive_space = predictive_space
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.xdim = xdim
        self.ydim = ydim
        self.hidden_dim = hidden_dim
        self.T = T

        self.device=device

        # initialize encoder network
        self.encoder_params = encoder_params or {}
        self.encoder_kwargs = dict(self.encoder_params) if self.encoder_params is not None else {}
        self.deterministic = self.encoder_kwargs.pop('deterministic', False)
        self.encoder_type = self.encoder_kwargs.pop('encoder_type', 'mlp')
        self.encoder = None
        if self.xdim is not None: # if xdim is not specified, initialize encoder network in fit method
            self.encoder = StructuredEncoder(
                input_dim=self.xdim,
                output_dim=self.ydim,
                hidden_dim=self.hidden_dim,
                T=self.T,
                device=self.device,
                deterministic=self.deterministic,
                encoder_type=self.encoder_type,
                **self.encoder_kwargs,
            )
            self.encoder.to(self.device)

        # initialize critic and baseline networks for I_predictive bound
        if mi_params is None:
            mi_params = {'estimator_compress': 'infonce_lower', 'estimator_predictive': 'infonce_lower',
                         'critic': 'concat', 'baseline': 'constant'}

        # critic network for I_predictive
        if critic_params is None:
            if self.predictive_space == "latent":
                critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
            elif self.predictive_space == "observation":
                assert self.xdim is not None, "xdim must be specified for predictive_space='observation'."
                critic_params = {"x_dim": T * ydim, "y_dim": T * self.xdim, "hidden_dim": hidden_dim}
        self.critic = CRITICS[mi_params.get('critic', 'concat')](**critic_params)
        self.critic.to(device)
   
        # baseline network for I_predictive (baseline(y); y is latent or raw future per predictive_space)
        if baseline_params is None:
            baseline_params = {"hidden_dim": hidden_dim}
        self.baseline_params = baseline_params
        baseline_type = mi_params.get('baseline', 'constant')
        if baseline_type == "constant":
            self.baseline = BASELINES[baseline_type]()
        else:
            if self.predictive_space == "latent":
                baseline_input_dim = self.T * self.ydim
            elif self.predictive_space == "observation":
                assert self.xdim is not None, "xdim must be specified for predictive_space='observation'."
                baseline_input_dim = self.T * self.xdim

            self.baseline = BASELINES[baseline_type](input_dim=baseline_input_dim, **self.baseline_params)
            self.baseline.to(device)
            

        # initialize critic network for I_YX
        if self.beta2 > 0:
            self.critic_YX = CRITICS[mi_params.get('critic', 'concat')](**critic_params_YX)
            self.critic_YX.to(device)

        self.mi_params = mi_params
        self.regularization_weight=regularization_weight


    def forward(self, X_past, X_future, debug=False):
        """
        Forward pass of the CPIC model.

        Parameters
        ----------
        X_past : torch.Tensor
            Past time-series data tensor.
        X_future : torch.Tensor
            Future time-series data tensor.
        debug : bool, optional
            Whether to print debug information. The default is False.

        Returns
        -------
        L : torch.Tensor
            CPIC loss.
        I_compress_bound : torch.Tensor
            Estimated compression mutual information bound.
        I_predictive_bound : torch.Tensor
            Estimated predictive mutual information bound.
        """    
        batch_size = X_past.shape[0]

        encoded_past_mean, encoded_past_vars = self.encoder(X_past)
        encoded_past = encoded_past_mean + torch.sqrt(encoded_past_vars) * \
                       torch.randn(*encoded_past_mean.size()).to(self.device)
        encoded_past_reshaped = encoded_past.reshape(batch_size, -1)

        encoded_future_mean, encoded_future_vars = self.encoder(X_future)
        encoded_future = encoded_future_mean + torch.sqrt(encoded_future_vars) * \
                       torch.randn(*encoded_future_mean.size()).to(self.device)
        encoded_future_reshaped = encoded_future.reshape(batch_size, -1)

        future_reshaped = X_future.reshape(batch_size, -1)

        if self.deterministic:
            I_compress_bound = torch.tensor([0]).to(self.device)
        else:
            I_compress_bound = estimate_mutual_information(self.mi_params['estimator_compress'], X_past,
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
        if debug:
            estimate_mutual_information(self.mi_params['estimator_compress'], X_past, encoded_past_reshaped,
                                        decoder=self.encoder, device=self.device, debug=debug)
            
        return L, I_compress_bound, I_predictive_bound


    def encode(self, X):
        """
        Encode the input data X into the latent space.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data tensor.
        """
        encoded_mean = self.encoder.get_mean(X)
        return encoded_mean


    def compute_encoded_mean_stats(self, X, batch_size=256, writer=None, step=0):
        """
        Compute and store the mean and variance of the encoded past mean and encoded future mean
        across the dataset. Stats are computed per latent dimension (sample mean and sample variance
        over samples). Optionally log them to a TensorBoard writer.

        Parameters
        ----------
        X : PastFutureDataset
            Dataset yielding (X_past, X_future) pairs.
        batch_size : int, optional
            Batch size for encoding. The default is 256.
        writer : SummaryWriter, optional
            TensorBoard SummaryWriter. If provided, histograms and scalar summaries of the stats
            are logged (encoded_past_mean_stats, encoded_future_mean_stats).
        step : int, optional
            Global step for writer. The default is 0.

        Attributes set
        ---------------
        encoded_past_mean_stats : dict
            'mean': np.ndarray (latent_dim,), 'variance': np.ndarray (latent_dim,).
        encoded_future_mean_stats : dict
            'mean': np.ndarray (latent_dim,), 'variance': np.ndarray (latent_dim,).
        """
        self.eval()
        past_means, future_means = [], []
        loader = DataLoader(X, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for X_past_batch, X_future_batch in loader:
                X_past_batch = X_past_batch.to(torch.float).to(self.device)
                X_future_batch = X_future_batch.to(torch.float).to(self.device)
                enc_past_mean, _ = self.encoder(X_past_batch)
                enc_future_mean, _ = self.encoder(X_future_batch)
                batch_size_actual = enc_past_mean.shape[0]
                # take the first timestamp of time series for the encoded past mean
                past_means.append(enc_past_mean[:, 0, :].cpu().numpy())
                # take the first timestamp of time series for the encoded future mean
                future_means.append(enc_future_mean[:, 0, :].cpu().numpy())
        self.train()
        past_means = np.concatenate(past_means, axis=0)
        future_means = np.concatenate(future_means, axis=0)
        self.encoded_past_mean_stats = {
            "mean": np.mean(past_means, axis=0),
            "variance": np.var(past_means, axis=0),
        }
        self.encoded_future_mean_stats = {
            "mean": np.mean(future_means, axis=0),
            "variance": np.var(future_means, axis=0),
        }
        if writer is not None:
            writer.add_histogram("encoded_mean_stats/encoded_past_mean/mean_in_latent_dimensions", self.encoded_past_mean_stats["mean"], step)
            writer.add_histogram("encoded_mean_stats/encoded_past_mean/variance_in_latent_dimensions", self.encoded_past_mean_stats["variance"], step)
            writer.add_histogram("encoded_mean_stats/encoded_future_mean/mean_in_latent_dimensions", self.encoded_future_mean_stats["mean"], step)
            writer.add_histogram("encoded_mean_stats/encoded_future_mean/variance_in_latent_dimensions", self.encoded_future_mean_stats["variance"], step)
            writer.add_scalar("encoded_mean_stats/encoded_past_mean/mean_over_dims", np.mean(self.encoded_past_mean_stats["mean"]), step)
            writer.add_scalar("encoded_mean_stats/encoded_past_mean/variance_over_dims", np.mean(self.encoded_past_mean_stats["variance"]), step)
            writer.add_scalar("encoded_mean_stats/encoded_future_mean/mean_over_dims", np.mean(self.encoded_future_mean_stats["mean"]), step)
            writer.add_scalar("encoded_mean_stats/encoded_future_mean/variance_over_dims", np.mean(self.encoded_future_mean_stats["variance"]), step)
        return self.encoded_past_mean_stats, self.encoded_future_mean_stats


    def fit(self, X, init_weights=None, epochs=100, batch_size=64, lr=1e-4, early_stop=10, writer=None, compute_encoded_mean_stats=True):
        """
        Fit the CPIC model to the data X.

        Parameters
        ----------
        X : PastFutureDataset
            Input data as a PastFutureDataset object. Contains past and future time-series data.
        init_weights : np.ndarray, optional
            Initial weights for the encoder mean layer. The default is None.
        epochs : int, optional
            Number of training epochs. The default is 100.
        batch_size : int, optional
            Batch size for training. The default is 64.
        lr : float, optional
            Learning rate for the optimizer. The default is 1e-4.
        early_stop : int, optional
            Early stopping patience. The default is 10.
        writer : SummaryWriter, optional
            tensorBoardX SummaryWriter for logging. The default is None.
        compute_encoded_mean_stats : bool, optional
            If True, compute and store mean/variance of encoded past and future means after training,
            and log them to writer if provided. The default is True.
        """
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)

        if self.xdim is None:
            self.xdim = X[0][0].shape[-1]
            self.encoder = StructuredEncoder(
                input_dim=self.xdim,
                output_dim=self.ydim,
                hidden_dim=self.hidden_dim,
                T=self.T,
                device=self.device,
                deterministic=self.deterministic,
                encoder_type=self.encoder_type,
                **self.encoder_kwargs,
            )
            self.encoder.to(self.device)

        # initialize encoder weights
        if init_weights is not None:
            self.encoder._mean.weight = torch.nn.parameter.Parameter(
                torch.from_numpy(init_weights.T).to(self.encoder._mean.weight.dtype).to(self.device))
        self.init_weights = init_weights

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        no_improve = 0 # counter for early stopping
        global_step = 0 # for tensorboard logging

        stats = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max} # for mutual information bounds

        if self.init_weights is not None:
            do_init = True
            optimizer_init = torch.optim.Adam(list(self.critic.parameters()), lr=lr)
        else:
            do_init = False

        for epoch in tqdm.tqdm(range(epochs)):
            loss_by_epoch, I_compress_bound_by_epoch, I_predictive_bound_by_epoch = [], [], []
            for X_past_batch, X_future_batch in train_loader:
                X_past_batch = X_past_batch.to(torch.float).to(self.device)
                X_future_batch = X_future_batch.to(torch.float).to(self.device)

                loss, I_compress_bound, I_predictive_bound = self(X_past_batch, X_future_batch)

                loss.backward()

                # Check if gradients are NaN
                grad_bool = True
                for name, param in self.named_parameters():
                    if not torch.isfinite(param.grad).all():
                        print(epoch, name, torch.isfinite(param.grad).all())
                        grad_bool = False
                        break
                if not grad_bool:
                    break
                if do_init and epoch < (epochs / 4):
                    optimizer_init.step()
                    optimizer_init.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if writer:
                    writer.add_scalar("batch/loss", loss.item(), global_step)
                    writer.add_scalar("batch/I_compress", I_compress_bound.item(), global_step)
                    writer.add_scalar("batch/I_predictive", I_predictive_bound.item(), global_step)
                    global_step += 1

                loss_by_epoch.append(loss.item())
                I_compress_bound_by_epoch.append(I_compress_bound.item())
                I_predictive_bound_by_epoch.append(I_predictive_bound.item())
            
            mean_loss = np.mean(loss_by_epoch)
            mean_I_compress = np.mean(I_compress_bound_by_epoch)
            mean_I_predictive = np.mean(I_predictive_bound_by_epoch)
            print(f"Epoch {epoch}: loss={mean_loss:.4f}, I_compress_bound={mean_I_compress:.4f}, I_predictive_bound={mean_I_predictive:.4f}")
            if writer:
                writer.add_scalar("epoch/loss/mean", mean_loss, global_step=epoch)
                
                if len(I_compress_bound_by_epoch) > 0:
                    for name, fn in stats.items():                    
                        writer.add_scalar(f"epoch/I_compress/{name}", fn(I_compress_bound_by_epoch), global_step=epoch) 
                    writer.add_histogram("epoch/I_compress_dist", np.array(I_compress_bound_by_epoch), epoch)

                if len(I_predictive_bound_by_epoch) > 0:
                    for name, fn in stats.items():                    
                        writer.add_scalar(f"epoch/I_predictive/{name}", fn(I_predictive_bound_by_epoch), global_step=epoch)
                    writer.add_histogram("epoch/I_predictive_dist", np.array(I_predictive_bound_by_epoch), epoch)

                if compute_encoded_mean_stats:
                    self.compute_encoded_mean_stats(X, batch_size=batch_size, writer=writer, step=epoch)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_I_compress = mean_I_compress
                best_I_predictive = mean_I_predictive
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    print("Early stopping...")
                    break
                
        return best_loss, best_I_compress, best_I_predictive


    def transform(self, X):
        """
        Encode the input data X into the latent space.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        encoded_X : torch.Tensor
            Encoded latent representations.
        """
        with torch.no_grad():
            encoded_X = self.encode(X.to(torch.float).to(self.device))
        return encoded_X
    

    def fit_transform(self, X, **kwargs):
        """
        Fit the CPIC model to the dataset and 
        encode the entire dataset into latent space.

        Parameters
        ----------
        X : PastFutureDataset
            Input data as a PastFutureDataset object. Contains past and future time-series data.
        **kwargs : dict
            Additional arguments for the fit method.
        
        Returns
        -------
        encoded_X : torch.Tensor
            Encoded latent representations.
        """
        self.fit(X, **kwargs)
        X_all = torch.cat([X[i][0].unsqueeze(0) for i in range(len(X))], dim=0)
        encoded_X = self.transform(X_all)
        return encoded_X


    def score(self, X_past, X_future):
        """
        Return the loss, I_compress_bound, and I_predictive_bound 
        estimates for given X_past and X_future.

        Parameters
        ----------
        X_past : torch.Tensor
            Past time-series data tensor.
        X_future : torch.Tensor
            Future time-series data tensor.

        Returns
        -------
        loss : float
            CPIC loss.
        I_compress_bound : float
            Estimated compression mutual information bound.
        I_predictive_bound : float
            Estimated predictive mutual information bound.
        """
        with torch.no_grad():
            loss, I_compress_bound, I_predictive_bound = self(X_past.float(), X_future.float())
        return loss.item(), I_compress_bound.item(), I_predictive_bound.item()


    def visualize_kernels(self, kernel_save_suffix=None, signature=None):
        """
        Visualize the convolutional kernels of ONLY an ConvSpatialEncoder, ConvSpatiotemporalEncoder, or ConvTemporalEncoder.

        Parameters
        ----------
        kernel_save_suffix : str, optional
            Optional suffix for kernel visualization directory. The default is None.
        signature : str or int, optional
            Signature/identifier for kernel visualization directory. The default is None.
        """
        if getattr(self.encoder, "encoder_type", None) in ('conv_spatial', 'conv_spatiotemporal', 'conv_temporal') and not self.encoder.linear_encoder:
            if signature is not None:
                if kernel_save_suffix is not None:
                    kernel_save_dir = f"kernel_visualizations/{signature}/{kernel_save_suffix}"
                else:
                    kernel_save_dir = f"kernel_visualizations/{signature}"
            else:
                if kernel_save_suffix is not None:
                    kernel_save_dir = f"kernel_visualizations/{kernel_save_suffix}"
                else:
                    kernel_save_dir = "kernel_visualizations"
            print(f'Visualizing kernels to {kernel_save_dir}...')
            visualize_conv_kernels(self, kernel_save_dir)
        else:
            raise ValueError("The encoder is not a ConvSpatialEncoder, ConvSpatiotemporalEncoder, or Conv1dTemporalEncoder.")
    

class SparseCPIC(CPIC):
    """
    Sparse CPIC model.
    """
    def __init__(self, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        # sparse regularization threshold
        self.gamma = gamma
        
        # initialize decoder network for sparse regularization, use a linear decoder (no bias) for sparse regularization
        self.decoder = None
        if self.xdim is not None: # if xdim is not specified, initialize decoder network in fit method
            self.decoder = nn.Linear(self.ydim, self.xdim, bias=False)
            self.decoder.to(self.device)


    def forward(self, X_past, X_future, debug=False):
        """
        Forward pass of the SparseCPIC model.

        Parameters
        ----------
        X_past : torch.Tensor
            Past time-series data tensor.
        X_future : torch.Tensor
            Future time-series data tensor.
        debug : bool, optional
            Whether to print debug information. The default is False.

        Returns
        -------
        L : torch.Tensor
            CPIC loss.
        I_compress_bound : torch.Tensor
            Estimated compression mutual information bound.
        I_predictive_bound : torch.Tensor
            Estimated predictive mutual information bound.
        decoder_loss : torch.Tensor
            Decoder loss.
        """    
        L, I_compress_bound, I_predictive_bound = super().forward(X_past, X_future, debug=debug)

        encoded_past_mean, encoded_past_vars = self.encoder(X_past)
        encoded_past = encoded_past_mean + torch.sqrt(encoded_past_vars) * \
                torch.randn(*encoded_past_mean.size()).to(self.device)
        # add proximal operator to encoded_past: shrink towards 0 if abs(value) > gamma, else unchanged
        encoded_past_proximal = torch.where(
            torch.abs(encoded_past) > self.gamma,
            torch.sign(encoded_past) * (torch.abs(encoded_past) - self.gamma),
            encoded_past
        )
        # Straight-through: forward = proximal, backward = identity
        encoded_past_proximal = encoded_past_proximal.detach() + (encoded_past - encoded_past.detach())

        decoded_past = self.decoder(encoded_past_proximal)


        encoded_future_mean, encoded_future_vars = self.encoder(X_future)
        encoded_future = encoded_future_mean + torch.sqrt(encoded_future_vars) * \
            torch.randn(*encoded_future_mean.size()).to(self.device)
        # add proximal operator to encoded_future: shrink towards 0 if abs(value) > gamma, else unchanged
        encoded_future = torch.where(
            torch.abs(encoded_future) > self.gamma,
            torch.sign(encoded_future) * (torch.abs(encoded_future) - self.gamma),
            encoded_future
        )
        # stop the gradient of encoded_future
        encoded_future = encoded_future.detach()
        decoded_future = self.decoder(encoded_future) 

        decoder_loss = (torch.mean((decoded_past - X_past) ** 2) + torch.mean((decoded_future - X_future) ** 2)) / 2

        L = L + decoder_loss
            
        return L, I_compress_bound, I_predictive_bound, decoder_loss


    def fit(self, X, init_weights=None, epochs=100, batch_size=64, lr=1e-4, early_stop=10, writer=None, compute_encoded_mean_stats=True):
        """
        Fit the CPIC model to the data X.

        Parameters
        ----------
        X : PastFutureDataset
            Input data as a PastFutureDataset object. Contains past and future time-series data.
        init_weights : np.ndarray, optional
            Initial weights for the encoder mean layer. The default is None.
        epochs : int, optional
            Number of training epochs. The default is 100.
        batch_size : int, optional
            Batch size for training. The default is 64.
        lr : float, optional
            Learning rate for the optimizer. The default is 1e-4.
        early_stop : int, optional
            Early stopping patience. The default is 10.
        writer : SummaryWriter, optional
            tensorBoardX SummaryWriter for logging. The default is None.
        compute_encoded_mean_stats : bool, optional
            If True, compute and store mean/variance of encoded past and future means after training,
            and log them to writer if provided. The default is True.
        """
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)

        if self.xdim is None:
            self.xdim = X[0][0].shape[-1]
            self.encoder = StructuredEncoder(
                input_dim=self.xdim,
                output_dim=self.ydim,
                hidden_dim=self.hidden_dim,
                T=self.T,
                device=self.device,
                deterministic=self.deterministic,
                encoder_type=self.encoder_type,
                **self.encoder_kwargs,
            )
            self.encoder.to(self.device)

            self.decoder = nn.Linear(self.ydim, self.xdim, bias=False)
            self.decoder.to(self.device)

        # initialize encoder weights
        if init_weights is not None:
            self.encoder._mean.weight = torch.nn.parameter.Parameter(
                torch.from_numpy(init_weights.T).to(self.encoder._mean.weight.dtype).to(self.device))
        self.init_weights = init_weights
                
        # initialize decoder weights using random normal distribution (weight shape is (out_features, in_features) = (xdim, ydim))
        self.decoder.weight = torch.nn.parameter.Parameter(
            torch.randn(self.xdim, self.ydim).to(self.device))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        no_improve = 0 # counter for early stopping
        global_step = 0 # for tensorboard logging

        stats = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max} # for mutual information bounds

        if self.init_weights is not None:
            do_init = True
            optimizer_init = torch.optim.Adam(list(self.critic.parameters()), lr=lr)
        else:
            do_init = False

        for epoch in tqdm.tqdm(range(epochs)):
            loss_by_epoch, I_compress_bound_by_epoch, I_predictive_bound_by_epoch, decoder_loss_by_epoch = [], [], [], []
            for X_past_batch, X_future_batch in train_loader:
                X_past_batch = X_past_batch.to(torch.float).to(self.device)
                X_future_batch = X_future_batch.to(torch.float).to(self.device)

                loss, I_compress_bound, I_predictive_bound, decoder_loss = self(X_past_batch, X_future_batch)

                loss.backward()

                # Check if gradients are NaN
                grad_bool = True
                for name, param in self.named_parameters():
                    if not torch.isfinite(param.grad).all():
                        print(epoch, name, torch.isfinite(param.grad).all())
                        grad_bool = False
                        break
                if not grad_bool:
                    break
                if do_init and epoch < (epochs / 4):
                    optimizer_init.step()
                    optimizer_init.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if writer:
                    writer.add_scalar("batch/loss", loss.item(), global_step)
                    writer.add_scalar("batch/I_compress", I_compress_bound.item(), global_step)
                    writer.add_scalar("batch/I_predictive", I_predictive_bound.item(), global_step)
                    writer.add_scalar("batch/decoder_loss", decoder_loss.item(), global_step)
                    global_step += 1

                loss_by_epoch.append(loss.item())
                I_compress_bound_by_epoch.append(I_compress_bound.item())
                I_predictive_bound_by_epoch.append(I_predictive_bound.item())
                decoder_loss_by_epoch.append(decoder_loss.item())
            
            mean_loss = np.mean(loss_by_epoch)
            mean_I_compress = np.mean(I_compress_bound_by_epoch)
            mean_I_predictive = np.mean(I_predictive_bound_by_epoch)
            mean_decoder_loss = np.mean(decoder_loss_by_epoch)
            print(f"Epoch {epoch}: loss={mean_loss:.4f}, I_compress_bound={mean_I_compress:.4f}, I_predictive_bound={mean_I_predictive:.4f}, decoder_loss={mean_decoder_loss:.4f}")
            if writer:
                writer.add_scalar("epoch/loss/mean", mean_loss, global_step=epoch)
                
                if len(I_compress_bound_by_epoch) > 0:
                    for name, fn in stats.items():                    
                        writer.add_scalar(f"epoch/I_compress/{name}", fn(I_compress_bound_by_epoch), global_step=epoch) 
                    writer.add_histogram("epoch/I_compress_dist", np.array(I_compress_bound_by_epoch), epoch)

                if len(I_predictive_bound_by_epoch) > 0:
                    for name, fn in stats.items():                    
                        writer.add_scalar(f"epoch/I_predictive/{name}", fn(I_predictive_bound_by_epoch), global_step=epoch)
                    writer.add_histogram("epoch/I_predictive_dist", np.array(I_predictive_bound_by_epoch), epoch)

                if len(decoder_loss_by_epoch) > 0:
                    for name, fn in stats.items():                    
                        writer.add_scalar(f"epoch/decoder_loss/{name}", fn(decoder_loss_by_epoch), global_step=epoch)
                    writer.add_histogram("epoch/decoder_loss_dist", np.array(decoder_loss_by_epoch), epoch)

                if compute_encoded_mean_stats:
                    self.compute_encoded_mean_stats(X, batch_size=batch_size, writer=writer, step=epoch)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_I_compress = mean_I_compress
                best_I_predictive = mean_I_predictive
                best_decoder_loss = mean_decoder_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    print("Early stopping...")
                    break

        return best_loss, best_I_compress, best_I_predictive, best_decoder_loss