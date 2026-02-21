import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from .models import StructuredEncoder, CRITICS, BASELINES, visualize_conv_kernels
from .mi import estimate_mutual_information


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
            linear_encoder : bool
                Whether to use linear encoder. Default is True.
            nonlinear_encoder_type : str
                Type of nonlinear encoder. One of ['mlp', 'conv']. Default is 'mlp'.
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
        self.deterministic = encoder_params.get('deterministic', False)
        self.linear_encoder = encoder_params.get('linear_encoder', True)
        self.nonlinear_encoder_type = encoder_params.get('nonlinear_encoder_type', 'mlp')
        self.n_layers = encoder_params.get('n_layers', 1)
        self.activation = encoder_params.get('activation', 'relu')
        self.conv_kernel_size = encoder_params.get('conv_kernel_size', 3)
        self.conv_stride = encoder_params.get('conv_stride', 1)
        self.conv_padding = encoder_params.get('conv_padding', 1)

        if mi_params is None:
            mi_params = {'estimator_compress': 'infonce_lower', 'estimator_predictive': 'infonce_lower',
                         'critic': 'concat', 'baseline': 'constant'}
        if critic_params is None:
            critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
        if baseline_params is None:
            baseline_params = {"hidden_dim": hidden_dim}

        # initialize critic and baseline for I_compress, I_predictive
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


    def fit(self, X, init_weights=None, epochs=100, batch_size=64, lr=1e-4, early_stop=10, writer=None, kernel_save_suffix=None, signature=None):
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
        kernel_save_suffix : str, optional
            Optional suffix for kernel visualization directory. The default is None.
        signature : str or int, optional
            Signature/identifier for kernel visualization directory. The default is None.
        """
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)

        if self.xdim is None:
            self.xdim = X[0][0].shape[-1]
        self.encoder = StructuredEncoder(input_dim=self.xdim, output_dim=self.ydim, hidden_dim=self.hidden_dim, 
                                         T=self.T, 
                                         device=self.device,
                                         deterministic=self.deterministic,
                                         linear_encoder=self.linear_encoder,
                                         nonlinear_encoder_type=self.nonlinear_encoder_type,
                                         n_layers=self.n_layers,
                                         activation=self.activation,
                                         conv_kernel_size=self.conv_kernel_size,
                                         conv_stride=self.conv_stride,
                                         conv_padding=self.conv_padding)
        self.encoder.to(self.device)
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
                for name, fn in stats.items():                    
                    writer.add_scalar(f"epoch/I_compress/{name}", fn(I_compress_bound_by_epoch), global_step=epoch) 
                    writer.add_scalar(f"epoch/I_predictive/{name}", fn(I_predictive_bound_by_epoch), global_step=epoch)
                writer.add_histogram("epoch/I_compress_dist", np.array(I_compress_bound_by_epoch), epoch)
                writer.add_histogram("epoch/I_predictive_dist", np.array(I_predictive_bound_by_epoch), epoch)

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

        # Visualize convolutional kernels if using conv encoder
        if self.encoder.nonlinear_encoder_type == 'conv' and not self.encoder.linear_encoder:
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

        return best_loss, best_I_compress, best_I_predictive


    def transform(self, X):
        """
        Enocode the input data X into the latent space.
        
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
    