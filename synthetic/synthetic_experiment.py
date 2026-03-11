"""
Run CPIC experiments for synthetic data.

Usage:
cd /path/to/CPIC_dev/synthetic

python synthetic_experiment.py --config lorenz_stochastic_infonce_demo
"""

from cpic import CPIC
from cpic.utils.data import PastFutureDataset
from cpic.utils.helpers import DCA_init
from utils.data_util import linear_alignment, compute_R2
import torch
from tensorboardX import SummaryWriter
import h5py
from utils.plot_util import plot_lorenz_3d_colored
import matplotlib.pyplot as plt
from configparser import ConfigParser
import argparse
import os
import pickle
import numpy as np


config_file_dict = {
        'lorenz_stochastic_infonce_demo': './config/config_lorenz_stochastic_infonce_demo.ini',
        'lorenz_deterministic_infonce': './config/config_lorenz_deterministic_infonce.ini',
        'lorenz_deterministic_nwj': './config/config_lorenz_deterministic_nwj.ini',
        'lorenz_deterministic_mine': './config/config_lorenz_deterministic_mine.ini',
        'lorenz_deterministic_tuba': './config/config_lorenz_deterministic_tuba.ini',
        'lorenz_stochastic_infonce': './config/config_lorenz_stochastic_infonce.ini',
        'lorenz_stochastic_nwj': './config/config_lorenz_stochastic_nwj.ini',
        'lorenz_stochastic_mine': './config/config_lorenz_stochastic_mine.ini',
        'lorenz_stochastic_tuba': './config/config_lorenz_stochastic_tuba.ini',
        'lorenz_deterministic_infonce_exploration': './config/config_lorenz_deterministic_infonce_exploration.ini',
        'lorenz_deterministic_infonce_obs_exploration': './config/config_lorenz_deterministic_infonce_obs_exploration.ini',
        'lorenz_stochastic_infonce_exploration': './config/config_lorenz_stochastic_infonce_exploration.ini',
        'lorenz_stochastic_infonce_obs_exploration': './config/config_lorenz_stochastic_infonce_obs_exploration.ini',
        'lorenz_stochastic_infonce_obs_exploration_conv_s': './config/config_lorenz_stochastic_infonce_obs_exploration_conv_s.ini',
        'lorenz_stochastic_infonce_obs_exploration_conv_st': './config/config_lorenz_stochastic_infonce_obs_exploration_conv_st.ini',
        'lorenz_stochastic_infonce_obs_exploration_conv_t': './config/config_lorenz_stochastic_infonce_obs_exploration_conv_t.ini',
        'lorenz_stochastic_infonce_exploration_conv_s': './config/config_lorenz_stochastic_infonce_exploration_conv_s.ini',
    }


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def plot_latent_trials(X_dynamics, X_pca_trans=None, X_dca_trans=None, X_CPIC_trans=None, num_vis=500, snr_val=None, save_dir=None,
                       plot_lorenz_func=plot_lorenz_3d_colored, show_title=False, max_2norm=3.2):
    snr_val = np.round(snr_val, 3)
    # max_2norm_pca = np.max(np.linalg.norm(X_pca_trans[:num_vis] - X_dynamics[:num_vis], axis=1))
    # max_2norm_dca = np.max(np.linalg.norm(X_dca_trans[:num_vis] - X_dynamics[:num_vis], axis=1))
    # max_2norm_CPIC = np.max(np.linalg.norm(X_CPIC_trans[:num_vis] - X_dynamics[:num_vis], axis=1))
    # print("snr_val:{}, max R2:{}".format(snr_val, np.max((max_2norm_dca, max_2norm_CPIC))))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_lorenz_func(ax, X_dynamics[:num_vis], X_dynamics[:num_vis], linewidth_3d, max_2norm=max_2norm)
    plt.title("True dynamics")
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "/true_dynamics.png")
    plt.close(fig)
    if X_pca_trans is not None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plot_lorenz_func(ax, X_pca_trans[:num_vis], X_dynamics[:num_vis], linewidth_3d, max_2norm=max_2norm)
        if show_title:
            plt.title("Embedded dynamics by PCA, SNR={}".format(snr_val))
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(save_dir + "/pca_dynamics_snr_{}.png".format(snr_val))
        plt.close(fig)
    if X_dca_trans is not None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plot_lorenz_func(ax, X_dca_trans[:num_vis], X_dynamics[:num_vis], linewidth_3d, max_2norm=max_2norm)
        if show_title:
            plt.title("Embedded dynamics by DCA, SNR={}".format(snr_val))
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(save_dir + "/dca_dynamics_snr_{}.png".format(snr_val))
        plt.close(fig)
    if X_CPIC_trans is not None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = plot_lorenz_func(ax, X_CPIC_trans[:num_vis], X_dynamics[:num_vis], linewidth_3d, max_2norm=max_2norm)
        if show_title:
            plt.title("Embedded dynamics by CPIC, SNR={}".format(snr_val))
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(save_dir + "/CPIC_dynamics_snr_{}.png".format(snr_val))
        plt.close(fig)

    # import matplotlib as mpl
    fig, ax = plt.subplots()
    cbar = plt.colorbar(p, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    ax.remove()
    plt.savefig(save_dir + "/snr_{}_cbar.png".format(snr_val))
    plt.close(fig)

def load_encoder_params(cfg):
    deterministic = cfg.getboolean('Hyperparameters', 'deterministic')
    
    # read encoder parameters from config
    linear_encoder = cfg.getboolean('Hyperparameters', 'linear_encoding') if cfg.has_option('Hyperparameters', 'linear_encoding') else True
    encoder_type = cfg.get('Hyperparameters', 'encoder_type') if cfg.has_option('Hyperparameters', 'encoder_type') else 'mlp'
    n_layers = cfg.getint('Hyperparameters', 'n_layers') if cfg.has_option('Hyperparameters', 'n_layers') else 1
    activation = cfg.get('Hyperparameters', 'activation') if cfg.has_option('Hyperparameters', 'activation') else 'relu'
    
    # conv-specific parameters
    conv_kernel_size = cfg.getint('Hyperparameters', 'conv_kernel_size') if cfg.has_option('Hyperparameters', 'conv_kernel_size') else 3
    conv_stride = cfg.getint('Hyperparameters', 'conv_stride') if cfg.has_option('Hyperparameters', 'conv_stride') else 1
    conv_padding = cfg.getint('Hyperparameters', 'conv_padding') if cfg.has_option('Hyperparameters', 'conv_padding') else 1
    
    encoder_params = {
        "deterministic": deterministic,
        "linear_encoder": linear_encoder,
        "nonlinear_encoder_type": encoder_type,
        "n_layers": n_layers,
        "activation": activation,
        "conv_kernel_size": conv_kernel_size,
        "conv_stride": conv_stride,
        "conv_padding": conv_padding
    }
    return encoder_params

def load_cpic_params(cfg):
    beta = cfg.getfloat('Hyperparameters', 'beta')
    xdim = cfg.getint('Hyperparameters', 'xdim')
    ydim = cfg.getint('Hyperparameters', 'ydim')
    T = cfg.getint('Hyperparameters', 'T')
    hidden_dim = cfg.getint('Hyperparameters', 'hidden_dim')

    estimator_compress = cfg.get('Hyperparameters', 'estimator_compress')
    estimator_predictive = cfg.get('Hyperparameters', 'estimator_predictive')
    critic = cfg.get('Hyperparameters', 'critic')
    baseline = cfg.get('Hyperparameters', 'baseline')
    predictive_space = cfg.get('Hyperparameters', 'predictive_space')
    mi_params = {'estimator_compress': estimator_compress, "estimator_predictive": estimator_predictive, "critic": critic,
                 "baseline": baseline}
    if predictive_space == "latent":
        critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
    elif predictive_space == "observation":
        critic_params = {"x_dim": T * ydim, "y_dim": T * xdim, "hidden_dim": hidden_dim}
    baseline_params = {"hidden_dim": hidden_dim}

    return {
        "beta": beta,
        "xdim": xdim,
        "ydim": ydim,
        "T": T,
        "hidden_dim": hidden_dim,
        "mi_params": mi_params,
        "critic_params": critic_params,
        "baseline_params": baseline_params,
        "predictive_space": predictive_space,
    }

linewidth_3d = 0.5


# uses config_lorenz_stochastic_infonce_exploration.ini arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC experiments.')
    parser.add_argument('--config', type=str, default="lorenz_stochastic_infonce_obs_exploration")
    parser.add_argument('--seed', type=int, default=22, help='seed for the DCA initialization')
    parser.add_argument('--signature', type=int, default=22, help='signature for the CPIC object')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.config in config_file_dict:
        config_file = config_file_dict[args.config]
    else:
        raise ValueError("{} has not been implemented!".format(args.config))

    cfg = myconf()
    cfg.read(config_file)
    RESULTS_FILENAME = cfg.get('User', 'RESULTS_FILENAME')
    saved_root = cfg.get('User', 'saved_root')
    if not os.path.exists(saved_root):
        os.makedirs(saved_root, exist_ok=True)

    # set CPIC hyper-parameters
    ydim, xdim, T, hidden_dim, beta, mi_params, critic_params, baseline_params, predictive_space = \
        (lambda p: (p["ydim"], p["xdim"], p["T"], p["hidden_dim"], p["beta"],
                    p["mi_params"], p["critic_params"], p["baseline_params"], p["predictive_space"]))(load_cpic_params(cfg))

    # set encoder parameters
    encoder_params = load_encoder_params(cfg)

    # set training parameters
    do_vis_latent_trials, batch_size, num_epochs, num_early_stop, num_vis, do_dca_init, lr = (
        cfg.getboolean('Training', 'do_vis_latent_trials'),
        cfg.getint('Training', 'batch_size'),
        cfg.getint('Training', 'num_epochs'),
        cfg.getint('Training', 'num_early_stop'),
        cfg.getint('Training', 'num_vis'),
        cfg.getboolean('Training', 'do_dca_init'),
        cfg.getfloat('Training', 'lr')
    )
    # check if cuda is available. if not, set device to cpu, else use the device specified in the config file or command line argument
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device or cfg.get('Training', 'device')
    signature = args.signature

    # load data
    with h5py.File(RESULTS_FILENAME, "r") as f:
        snr_vals = f.attrs["snr_vals"][:]
        X = f["X"][:]
        X_dynamics = f["X_dynamics"][:]
        X_noisy_dset = f["X_noisy"][:]
        X_pca_trans_dset = f["X_pca_trans"][:]
        X_dca_trans_dset = f["X_dca_trans"][:]
    # assert data dimensions match the CPIC hyper-parameters
    assert X.shape[-1] == xdim, f"X.shape[-1] ({X.shape[-1]}) != xdim ({xdim})" # (N, xdim)

    # run CPIC training
    R2_metrics = []
    losses = []
    inferred_CPIC_trials = []
    for snr_val, X_pca_trans, X_dca_trans, X_noisy in zip(snr_vals, X_pca_trans_dset, X_dca_trans_dset, X_noisy_dset):
        train_data = PastFutureDataset([X_noisy], window_size=T)

        # initialize the model parameters if do_dca_init is True
        if do_dca_init:
            init_weights = DCA_init(X_noisy, T=T, d=ydim, rng_or_seed=args.seed)
        else:
            init_weights = None

        # include SNR in kernel save path so each SNR's kernels are saved in a separate folder
        kernel_suffix = f"{args.config}/snr_{snr_val}" if args.config else f"snr_{snr_val}"

        cpic = CPIC(ydim=ydim, 
                    mi_params=mi_params, 
                    critic_params=critic_params, 
                    baseline_params=baseline_params,
                    encoder_params=encoder_params,
                    T=T,
                    hidden_dim=hidden_dim,
                    beta=beta, 
                    device=device,
                    predictive_space=predictive_space).to(device)

        loss, _, _ = cpic.fit(X=train_data, 
                              init_weights=init_weights,
                              epochs=num_epochs, 
                              batch_size=batch_size, 
                              lr=lr, 
                              early_stop=num_early_stop, 
                              writer=SummaryWriter(log_dir="tensor_logs/{}".format(signature)),
                              kernel_save_suffix=kernel_suffix,
                              signature=args.signature)

        encoder_type = getattr(cpic.encoder, "encoder_type", None)
        X_true_r2 = X_dynamics
        if encoder_type == "conv_spatiotemporal":
            past_windows = np.stack([X_noisy[t - T:t] for t in range(T, len(X_noisy))], axis=0) # (N-T, T, xdim)
            past_tensor = torch.from_numpy(past_windows).to(device)
            center_indices = [t - 1 for t in range(T, len(X_noisy))] # last index of each past window
            X_true_r2 = X_dynamics[center_indices]

            encoded_windows = cpic.encode(past_tensor) # (N-T, T, ydim)
            encoded_repr = encoded_windows[:, -1, :] # (N-T, ydim)
            X_CPIC_trans = aligned_encoded_mean = linear_alignment(encoded_repr.cpu().detach().numpy(), X_true_r2)
        else:
            encoded_mean = cpic.encode(torch.from_numpy(X_noisy).to(device))
            X_CPIC_trans = aligned_encoded_mean = linear_alignment(encoded_mean.cpu().detach().numpy(), X_dynamics)

        R2_PCA = compute_R2(X_pca_trans, X_dynamics)
        R2_DCA = compute_R2(X_dca_trans, X_dynamics)
        R2_CPIC = compute_R2(aligned_encoded_mean, X_true_r2)
        print("R2(PCA): {}, R2(DCA): {}, R2(CPIC): {}".format(R2_PCA, R2_DCA, R2_CPIC))
        R2_metrics.append([R2_PCA, R2_DCA, R2_CPIC])
        losses.append(loss)
        inferred_CPIC_trials.append(X_CPIC_trans)
        if do_vis_latent_trials:
            plot_latent_trials(X_dynamics, X_pca_trans, X_dca_trans, X_CPIC_trans, num_vis, snr_val=snr_val, save_dir=saved_root)
    
    # save the results
    R2_metrics = np.stack(R2_metrics)
    seed_str = f"_seed{args.seed}" if args.seed is not None else ""
    with open(f"{saved_root}/latent_R2{seed_str}.pkl", "wb") as f:
        pickle.dump({"R2_metrics": R2_metrics, "snr_vals": snr_vals, "losses": losses}, f)
    with open(f"{saved_root}/inferred_trials{seed_str}.pkl", "wb") as f:
        pickle.dump({"inferred_CPIC_trials": inferred_CPIC_trials, "snr_vals": snr_vals}, f)
