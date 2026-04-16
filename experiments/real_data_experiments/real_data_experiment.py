import argparse
from configparser import ConfigParser
import os

import torch
from torch.utils.data import DataLoader
from dca import data_util

import numpy as np
from sklearn.linear_model import LinearRegression as LR
from synthetic.utils.data_util import CrossValidate
from synthetic.utils.cov_util import form_lag_matrix
import pickle


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    """Train a linear model on the training set and test on the test set.
    This will work with batched training data and/or batched test data.
    X_train : ndarray (time, channels) or (batches, time, channels)
        Feature training data for regression.
    Y_train : ndarray (time, channels) or (batches, time, channels)
        Target training data for regression.
    X_test : ndarray (time, channels) or (batches, time, channels)
        Feature test data for regression.
    Y_test : ndarray (time, channels) or (batches, time, channels)
        Target test data for regression.
    decoding_window : int
        Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    offset : int
        Temporal offset for prediction (0 is same-time prediction).
    """

    if isinstance(X_train, np.ndarray) and X_train.ndim == 2:
        X_train = [X_train]
    if isinstance(Y_train, np.ndarray) and Y_train.ndim == 2:
        Y_train = [Y_train]

    if isinstance(X_test, np.ndarray) and X_test.ndim == 2:
        X_test = [X_test]
    if isinstance(Y_test, np.ndarray) and Y_test.ndim == 2:
        Y_test = [Y_test]

    X_train_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_train]
    X_test_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_test]

    Y_train = [Yi[decoding_window // 2:] for Yi in Y_train]
    Y_train = [Yi[:len(Xi)] for Yi, Xi in zip(Y_train, X_train_lags)]
    if offset >= 0:
        Y_train = [Yi[offset:] for Yi in Y_train]
    else:
        Y_train = [Yi[:Yi.shape[0] + offset] for Yi in Y_train]

    Y_test = [Yi[decoding_window // 2:] for Yi in Y_test]
    Y_test = [Yi[:len(Xi)] for Yi, Xi in zip(Y_test, X_test_lags)]
    if offset >= 0:
        Y_test = [Yi[offset:] for Yi in Y_test]
    else:
        Y_test = [Yi[:Yi.shape[0] + offset] for Yi in Y_test]

    if offset >= 0:
        X_train_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_train_lags]
        X_test_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_test_lags]
    else:
        X_train_lags = [Xi[-offset:] for Xi in X_train_lags]
        X_test_lags = [Xi[-offset:] for Xi in X_test_lags]

    if len(X_train_lags) == 1:
        X_train_lags = X_train_lags[0]
    else:
        X_train_lags = np.concatenate(X_train_lags)

    if len(Y_train) == 1:
        Y_train = Y_train[0]
    else:
        Y_train = np.concatenate(Y_train)

    if len(X_test_lags) == 1:
        X_test_lags = X_test_lags[0]
    else:
        X_test_lags = np.concatenate(X_test_lags)

    if len(Y_test) == 1:
        Y_test = Y_test[0]
    else:
        Y_test = np.concatenate(Y_test)

    model = LR().fit(X_train_lags, Y_train)
    r2 = model.score(X_test_lags, Y_test)
    return r2


def run_analysis_cpic(X, Y, T_pi_vals, dim_vals, offset_vals, num_cv_folds, decoding_window,
                 n_init=1, verbose=False, Kernel=None, xdim=None, beta=1e-3, beta1=1, beta2=0, critic_params=None,
                 critic_params_YX=None, good_ts=None):
    """

    :param X:
    :param Y:
    :param T_pi_vals: window size for DCA and CPIC
    :param dim_vals: compressed dimension
    :param offset_vals: Temporal offsets for prediction (0 is same-time prediction)
    :param num_cv_folds:
    :param decoding_window: Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    :param n_init: the number of initialization for DCA
    :param verbose:
    :return:
    """
    results_size = (num_cv_folds, len(dim_vals), len(offset_vals), len(T_pi_vals))
    results = np.zeros(results_size)
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    if Kernel is not None:
        xdim = int(xdim + xdim * (xdim + 1) / 2)  # polynomial

    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)

    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        if Kernel is not None:
            X_train = [Kernel(Xi) for Xi in X_train]
            X_test = Kernel(X_test)

        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            # loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):
                T_pi = T_pi_vals[T_pi_idx]
                critic_params = {"x_dim": T_pi * ydim, "y_dim": T_pi * ydim, "hidden_dim": hidden_dim}
                critic_params_YX = {"x_dim": T_pi * ydim, "y_dim": T_pi * xdim, "hidden_dim": hidden_dim}
                # train data
                if do_dca_init and linear_encoding:
                    init_weights = DCA_init(np.concatenate(X_train_ctd, axis=0), T=T_pi, d=dim, n_init=n_init)
                else:
                    init_weights = None

                train_data = PastFutureDataset(X_train_ctd, window_size=T_pi)
                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

                CPIC, _ = train_CPIC(beta, xdim, dim, mi_params, critic_params, baseline_params, num_epochs,
                                  train_dataloader, T=T_pi,
                                  signature=args.config, deterministic=deterministic, linear_encoding=linear_encoding,
                                  init_weights=init_weights, lr=lr, num_early_stop=num_early_stop, device=device,
                                  beta1=beta1, beta2=beta2, critic_params_YX=critic_params_YX)
                CPIC = CPIC.to(device)
                CPIC.eval()

                # encode train data and test data via CPIC
                X_train_cpic = [CPIC.encode(torch.from_numpy(Xi).to(torch.float).to(device)) for Xi in X_train_ctd]
                X_train_cpic = [Xi.cpu().detach().numpy() for Xi in X_train_cpic]
                # X_train_dca = [np.dot(Xi, init_weights) for Xi in X_train_ctd]
                X_test_cpic = CPIC.encode(torch.from_numpy(X_test_ctd).to(torch.float).to(device))
                X_test_cpic = X_test_cpic.cpu().detach().numpy()
                # X_test_dca = np.dot(X_test_ctd, init_weights)
                ### save encoded test data

                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_cpic = linear_decode_r2(X_train_cpic, Y_train_ctd, X_test_cpic, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    # r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    results[fold_idx, dim_idx, offset_idx, T_pi_idx] = r2_cpic
            print("fold_idx: {}, dim_idx: {}, R2: {}".format(fold_idx, dim_vals[dim_idx], results[fold_idx, dim_idx]))
            # import pdb; pdb.set_trace()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC for real data.')
    parser.add_argument('--config', type=str, default="m1_stochastic_infonce")
    parser.add_argument('--model', type=str, default="CPIC")
    parser.add_argument('--nonlinear_encoding', action='store_true')
    args = parser.parse_args()
    if args.model == "CPIC":
        from cpic.CPIC import PastFutureDataset, train_CPIC, DCA_init, Polynomial_expand
    if args.model == "PFPC_RC":
        from PFPC_RC import PastFutureDataset, train_CPIC, DCA_init, Polynomial_expand
    if args.config == 'm1_stochastic_infonce':
        config_file = 'config/config_m1_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'm1_stochastic_infonce_alt':
        config_file = 'config/config_m1_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'm1_deterministic_infonce_alt':
        config_file = 'config/config_m1_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_stochastic_infonce':
        config_file = 'config/config_hc_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_stochastic_infonce_alt':
        config_file = 'config/config_hc_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_deterministic_infonce_alt':
        config_file = 'config/config_hc_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'temp_stochastic_infonce':
        config_file = 'config/config_temp_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'temp_stochastic_infonce_alt':
        config_file = 'config/config_temp_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'ms_stochastic_infonce':
        config_file = 'config/config_ms_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'ms_stochastic_infonce_alt':
        config_file = 'config/config_ms_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    else:
        raise ValueError("{} has not been implemented!".format(args.config))

    cfg = myconf()
    cfg.read(config_file)
    RESULTS_FILENAME = cfg.get('User', 'RESULTS_FILENAME')
    saved_root = cfg.get('User', 'saved_root')
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)

    # set hyper-parameters
    beta = cfg.getfloat('Hyperparameters', 'beta')
    beta1 = cfg.getfloat('Hyperparameters', 'beta1')
    beta2 = cfg.getfloat('Hyperparameters', 'beta2')
    xdim = cfg.getint('Hyperparameters', 'xdim')
    Ts = cfg.get('Hyperparameters', 'T')
    Ts = Ts.split(' ')
    Ts = [int(T) for T in Ts]
    hidden_dim = cfg.getint('Hyperparameters', 'hidden_dim')

    estimator_compress = cfg.get('Hyperparameters', 'estimator_compress')
    estimator_predictive = cfg.get('Hyperparameters', 'estimator_predictive')
    critic = cfg.get('Hyperparameters', 'critic')
    baseline = cfg.get('Hyperparameters', 'baseline')
    kernel = cfg.get('Hyperparameters', 'kernel')
    mi_params = {'estimator_compress': estimator_compress, "estimator_predictive": estimator_predictive,
                 "critic": critic,
                 "baseline": baseline}
    # critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
    baseline_params = {"hidden_dim": hidden_dim}
    deterministic = cfg.getboolean('Hyperparameters', 'deterministic')
    # set training parameters
    do_vis_latent_trials = cfg.getboolean('Training', 'do_vis_latent_trials')
    batch_size = cfg.getint('Training', 'batch_size')
    num_epochs = cfg.getint('Training', 'num_epochs')
    num_early_stop = cfg.getint('Training', 'num_early_stop')
    num_vis = cfg.getint('Training', 'num_vis')
    do_dca_init = cfg.getboolean('Training', 'do_dca_init')
    device = cfg.get('Training', 'device')
    lr = cfg.getfloat('Training', 'lr')

    if args.config == "m1_stochastic_infonce" or args.config == "m1_stochastic_infonce_alt" \
            or args.config == "m1_deterministic_infonce_alt":
        # M1 = data_util.load_sabes_data('/home/fan/Data/M1/indy_20160627_01.mat')
        M1 = data_util.load_sabes_data('/home/rui/Data/M1/indy_20160627_01.mat')
        X, Y = M1['M1'], M1['cursor']
        good_ts = None
    if args.config == "hc_stochastic_infonce" or args.config == "hc_stochastic_infonce_alt"\
            or args.config == "hc_deterministic_infonce_alt":
        # HC = data_util.load_kording_paper_data('/home/fan/Data/HC/example_data_hc.pickle')
        HC = data_util.load_kording_paper_data('/home/rui/Data/HC/example_data_hc.pickle')
        X, Y = HC['neural'], HC['loc']
        good_ts = 22000
    if args.config == "temp_stochastic_infonce" or args.config == "temp_stochastic_infonce_alt":
        # weather = data_util.load_weather_data('/home/fan/Data/TEMP/temperature.csv')
        weather = data_util.load_weather_data('/home/rui/Data/TEMP/temperature.csv')
        X, Y = weather, weather
        good_ts = None
    if args.config == "ms_stochatic_infonce" or args.config == "ms_stochastic_infonce_alt":
        ms = data_util.load_accel_data('/home/rui/Data/motion_sense/A_DeviceMotion_data/std_6/sub_19.csv')
        X, Y = ms, ms
        good_ts = None

    T_pi_vals = np.array(Ts)
    offsets = np.array([5, 10, 15])

    win = 3
    n_cv = 5
    n_init = 5

    # rewrite ydim
    # m1_ydims = np.array([5,10,20,30])
    # hc_ydims = np.array([5,10,15,25])
    # temp_ydims = np.array([3, 4, 5, 6])
    ydims = np.array([5])
    if kernel == "Linear":
        Kernel = None
    elif kernel == "Polynomial":
        Kernel = Polynomial_expand
    else:
        raise ValueError("This kernel is not available.")

    if args.nonlinear_encoding:
        linear_encoding=False
    else:
        linear_encoding=True

    for ydim in ydims:
        # rewrite critic_params
        # critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
        # critic_params_YX = {"x_dim": T * ydim, "y_dim": T * xdim, "hidden_dim": hidden_dim}
        critic_params = None
        critic_params_YX = None
        result = run_analysis_cpic(X, Y, T_pi_vals, dim_vals=[ydim], offset_vals=offsets, num_cv_folds=n_cv, decoding_window=win,
                          n_init=n_init, verbose=True, Kernel=Kernel, xdim=xdim, beta=beta, beta1=beta1, beta2=beta2,
                          critic_params=critic_params, critic_params_YX=critic_params_YX, good_ts=good_ts)

        with open(saved_root + "/result_dim{}.pkl".format(ydim), "wb") as f:
            pickle.dump(result, f)
        # import pdb; pdb.set_trace()
