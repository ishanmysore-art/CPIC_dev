from cpic import CPIC
from utils import PastFutureDataset, DCA_init
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
    # import pdb; pdb.set_trace()


linewidth_3d = 0.5


# uses config_lorenz_stochastic_infonce_exploration.ini arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC experiments.')
    parser.add_argument('--config', type=str, default="lorenz_stochastic_infonce_obs_exploration")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    if args.config == 'lorenz_deterministic_infonce':
        config_file = './config/config_lorenz_deterministic_infonce.ini'
    elif args.config == 'lorenz_deterministic_nwj':
        config_file = './config/config_lorenz_deterministic_nwj.ini'
    elif args.config == 'lorenz_deterministic_mine':
        config_file = './config/config_lorenz_deterministic_mine.ini'
    elif args.config == 'lorenz_deterministic_tuba':
        config_file = './config/config_lorenz_deterministic_tuba.ini'
    elif args.config == 'lorenz_stochastic_infonce':
        config_file = './config/config_lorenz_stochastic_infonce.ini'
    elif args.config == 'lorenz_stochastic_nwj':
        config_file = './config/config_lorenz_stochastic_nwj.ini'
    elif args.config == 'lorenz_stochastic_mine':
        config_file = './config/config_lorenz_stochastic_mine.ini'
    elif args.config == 'lorenz_stochastic_tuba':
        config_file = './config/config_lorenz_stochastic_tuba.ini'
    elif args.config == 'lorenz_deterministic_infonce_exploration':
        config_file = './config/config_lorenz_deterministic_infonce_exploration.ini'
    elif args.config == 'lorenz_deterministic_infonce_obs_exploration':
        config_file = './config/config_lorenz_deterministic_infonce_obs_exploration.ini'
    elif args.config == 'lorenz_stochastic_infonce_exploration':
        config_file = './config/config_lorenz_stochastic_infonce_exploration.ini'
    elif args.config == 'lorenz_stochastic_infonce_obs_exploration':
        config_file = './config/config_lorenz_stochastic_infonce_obs_exploration.ini'
    elif args.config == 'lorenz_stochastic_infonce_obs_exploration_conv':
        config_file = './config/config_lorenz_stochastic_infonce_obs_exploration_conv.ini'
    elif args.config == 'lorenz_stochastic_infonce_exploration_conv':
        config_file = './config/config_lorenz_stochastic_infonce_exploration_conv.ini'
    else:
        raise ValueError("{} has not been implemented!".format(args.config))

    cfg = myconf()
    cfg.read(config_file)
    RESULTS_FILENAME = cfg.get('User', 'RESULTS_FILENAME')
    saved_root = cfg.get('User', 'saved_root')
    if not os.path.exists(saved_root):
        os.makedirs(saved_root, exist_ok=True)

    # set hyper-parameters
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

    # set training parameters
    do_vis_latent_trials = cfg.getboolean('Training', 'do_vis_latent_trials')
    batch_size = cfg.getint('Training', 'batch_size')
    num_epochs = cfg.getint('Training', 'num_epochs')
    num_early_stop = cfg.getint('Training', 'num_early_stop')
    num_vis = cfg.getint('Training', 'num_vis')
    do_dca_init = cfg.getboolean('Training', 'do_dca_init')
    lr = cfg.getfloat('Training', 'lr')
    if args.device is None:
        device = cfg.get('Training', 'device')
    else:
        device = args.device

    signature=22

    # load data
    with h5py.File(RESULTS_FILENAME, "r") as f:
        snr_vals = f.attrs["snr_vals"][:]
        X = f["X"][:]
        X_dynamics = f["X_dynamics"][:]
        X_noisy_dset = f["X_noisy"][:]
        X_pca_trans_dset = f["X_pca_trans"][:]
        X_dca_trans_dset = f["X_dca_trans"][:]

    R2_metrics = []
    losses = []
    inferred_CPIC_trials = []
    for snr_val, X_pca_trans, X_dca_trans, X_noisy in zip(snr_vals, X_pca_trans_dset, X_dca_trans_dset, X_noisy_dset):
        train_data = PastFutureDataset([X_noisy], window_size=T)

        # train data
        if do_dca_init:
            init_weights = DCA_init(X_noisy, T=T, d=ydim, rng_or_seed=args.seed)
        else:
            init_weights = None

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
                              writer=SummaryWriter(log_dir="tensor_logs/{}".format(signature)))
                    
        encoded_mean = cpic.encode(torch.from_numpy(X_noisy).to(device))
        X_CPIC_trans = aligned_encoded_mean = linear_alignment(encoded_mean.cpu().detach().numpy(), X_dynamics)
        R2_PCA = compute_R2(X_pca_trans, X_dynamics)
        R2_DCA = compute_R2(X_dca_trans, X_dynamics)
        R2_CPIC = compute_R2(aligned_encoded_mean, X_dynamics)
        print("R2(PCA): {}, R2(DCA): {}, R2(CPIC): {}".format(R2_PCA, R2_DCA, R2_CPIC))
        R2_metrics.append([R2_PCA, R2_DCA, R2_CPIC])
        losses.append(loss)
        inferred_CPIC_trials.append(X_CPIC_trans)
        if do_vis_latent_trials:
            plot_latent_trials(X_dynamics, X_pca_trans, X_dca_trans, X_CPIC_trans, num_vis, snr_val=snr_val, save_dir=saved_root)
    R2_metrics = np.stack(R2_metrics)
    if args.seed is not None:
        with open(saved_root + "/latent_R2_seed{}.pkl".format(args.seed), "wb") as f:
            pickle.dump({"R2_metrics": R2_metrics, "snr_vals": snr_vals, "losses": losses}, f)
        with open(saved_root + "/inferred_trials_seed{}.pkl".format(args.seed), "wb") as f:
            pickle.dump({"inferred_CPIC_trials": inferred_CPIC_trials, "snr_vals": snr_vals}, f)
    else:
        with open(saved_root + "/latent_R2.pkl", "wb") as f:
            pickle.dump({"R2_metrics": R2_metrics, "snr_vals": snr_vals, "losses": losses}, f)
        with open(saved_root + "/inferred_trials.pkl", "wb") as f:
            pickle.dump({"inferred_CPIC_trials": inferred_CPIC_trials, "snr_vals": snr_vals}, f)
