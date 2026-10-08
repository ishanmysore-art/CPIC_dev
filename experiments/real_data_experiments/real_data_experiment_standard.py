import argparse
from configparser import ConfigParser
import os
from pathlib import Path
import random
import time

import torch
from cpic.exp_utils import data_util

import numpy as np
from sklearn.linear_model import LinearRegression as LR
from cpic.exp_utils.cov_util import form_lag_matrix
import pickle

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_path(*parts):
    """Resolve ``data/real_data/...`` paths from the repository root."""
    return str(REPO_ROOT.joinpath(*parts))


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def prepare_linear_decode_data(X, Y, decoding_window=1, offset=0):
    """
    Apply the exact lag/offset alignment used by linear_decode_r2,
    but do not fit a regression model.
    """
    if isinstance(X, np.ndarray) and X.ndim == 2:
        X = [X]
    else:
        X = list(X)

    if isinstance(Y, np.ndarray) and Y.ndim == 2:
        Y = [Y]
    else:
        Y = list(Y)

    X_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X]

    Y_aligned = [Yi[decoding_window // 2:] for Yi in Y]
    Y_aligned = [
        Yi[:len(Xi)]
        for Yi, Xi in zip(Y_aligned, X_lags)
    ]

    if offset >= 0:
        Y_aligned = [Yi[offset:] for Yi in Y_aligned]
        X_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_lags]
    else:
        Y_aligned = [
            Yi[:Yi.shape[0] + offset]
            for Yi in Y_aligned
        ]
        X_lags = [Xi[-offset:] for Xi in X_lags]

    if len(X_lags) == 1:
        X_lags = X_lags[0]
    else:
        X_lags = np.concatenate(X_lags)

    if len(Y_aligned) == 1:
        Y_aligned = Y_aligned[0]
    else:
        Y_aligned = np.concatenate(Y_aligned)

    return X_lags, Y_aligned


def fit_linear_decoder(X_train, Y_train, decoding_window=1, offset=0):
    """
    Fit the same linear behavioral decoder used by linear_decode_r2.
    """
    X_train_lags, Y_train_aligned = prepare_linear_decode_data(
        X_train,
        Y_train,
        decoding_window=decoding_window,
        offset=offset,
    )

    return LR().fit(X_train_lags, Y_train_aligned)


def score_linear_decoder(
    model,
    X_test,
    Y_test,
    decoding_window=1,
    offset=0,
):
    """
    Score an already-fitted behavioral decoder without refitting it.
    """
    X_test_lags, Y_test_aligned = prepare_linear_decode_data(
        X_test,
        Y_test,
        decoding_window=decoding_window,
        offset=offset,
    )

    return model.score(X_test_lags, Y_test_aligned)


def linear_decode_r2(
    X_train,
    Y_train,
    X_test,
    Y_test,
    decoding_window=1,
    offset=0,
):
    """
    Original behavior: fit on X_train/Y_train and score X_test/Y_test.
    """
    model = fit_linear_decoder(
        X_train,
        Y_train,
        decoding_window=decoding_window,
        offset=offset,
    )

    return score_linear_decoder(
        model,
        X_test,
        Y_test,
        decoding_window=decoding_window,
        offset=offset,
    )


def run_analysis_cpic(X, Y, T_pi_vals, dim_vals, offset_vals, decoding_window,
                      n_init=1, verbose=False, Kernel=None, xdim=None, beta=1e-3, beta1=1, beta2=0, good_ts=None,
                      standardize_Y=False, train_test_ratio=0.8, regularization_weight=0,
                      predictive_loss="mi", reconstruction_targets=("past",), predictive_space="latent",
                      hidden_dim=256, n_layers=1, neuron_dropout_p=0.0, neuron_dropout_seed=0, 
                      eval_missing_ps=(0.0,), eval_mask_reps=1, eval_mask_seed=4242, availability_mask=False):
    """
    :param X: N x XDim
    :param Y: N x YDim
    :param T_pi_vals: window size for DCA and CPIC
    :param dim_vals: compressed dimension
    :param offset_vals: Temporal offsets for prediction (0 is same-time prediction)
    :param decoding_window: Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    :param n_init: the number of initialization for DCA
    :param verbose:
    :return:
    """
    results_r2_size = (len(dim_vals), len(offset_vals), len(T_pi_vals))
    results_MI_size = (len(dim_vals), len(T_pi_vals), 2)
    results_r2 = np.zeros(results_r2_size)
    results_MI = np.zeros(results_MI_size)

    results_runtime = np.zeros((len(dim_vals), len(T_pi_vals)))
    results_peak_memory = np.zeros((len(dim_vals), len(T_pi_vals)))

    results_masked_r2 = np.full(
        (
            len(dim_vals),
            len(eval_missing_ps),
            eval_mask_reps,
            len(offset_vals),
            len(T_pi_vals),
        ),
        np.nan,
        dtype=float,
    )

    # Same shape as results_masked_r2, but the behavioral decoder
    # is fitted ONCE on full-neuron training latents and then frozen.
    results_frozen_masked_r2 = np.full(
        (
            len(dim_vals),
            len(eval_missing_ps),
            eval_mask_reps,
            len(offset_vals),
            len(T_pi_vals),
        ),
        np.nan,
        dtype=float,
    )


    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]

    actual_xdim = X.shape[-1]

    if xdim is not None and xdim != actual_xdim:
        print(f"Warning: configured xdim={xdim}, "
            f"but processed data has {actual_xdim} features.")

    # Number of actual neural channels after preprocessing.
    neural_xdim = actual_xdim
    xdim = neural_xdim

    # For this controlled experiment, availability-mask augmentation
    # is only supported on the original neural feature space.
    if availability_mask and Kernel is not None:
        raise ValueError("availability_mask is currently supported only with Kernel=None")

    # Mask-aware CPIC receives:
    # [neural activity, binary availability mask]
    encoder_xdim = (2 * neural_xdim if availability_mask else neural_xdim)

    def build_encoder_input(Xi, neuron_mask):
        """
        Zero-fill unavailable neurons and optionally append
        the binary neuron-availability mask.
        """
        X_masked = Xi * neuron_mask

        if not availability_mask:
            return X_masked

        availability = np.broadcast_to(neuron_mask, Xi.shape,).astype(Xi.dtype, copy=False)

        return np.concatenate([X_masked, availability], axis=-1,)

    if availability_mask:
        print(f"Mask-aware encoder enabled: "
            f"{neural_xdim} neural dims -> "
            f"{encoder_xdim} encoder input dims")

    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    if Kernel is not None:
        xdim = int(xdim + xdim * (xdim + 1) / 2)  # polynomial
        encoder_xdim = xdim



    n = X.shape[0]
    n_train = int(n * train_test_ratio)
    X_train = X[:n_train]
    X_test = X[n_train:]
    Y_train = Y[:n_train]
    Y_test = Y[n_train:]

    if Kernel is not None:
        X_train = [Kernel(Xi) for Xi in X_train]
        X_test = Kernel(X_test)

    # mean-center X and Y
    X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
    X_train_ctd = [Xi - X_mean for Xi in X_train]
    X_train_ctd = [np.stack(X_train_ctd)]
    X_test_ctd = X_test - X_mean
    if standardize_Y:
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean
        Y_train = Y_train_ctd
        Y_test = Y_test_ctd
    Y_train = [np.stack(Y_train)]

    # loop over dimensionalities
    for dim_idx in range(len(dim_vals)):
        dim = dim_vals[dim_idx]
        if verbose:
            print("dim", dim_idx + 1, "of", len(dim_vals))

        # loop over T_pi vals
        for T_pi_idx in range(len(T_pi_vals)):
            T_pi = T_pi_vals[T_pi_idx]
            critic_params = {"x_dim": T_pi * dim, "y_dim": T_pi * dim, "hidden_dim": hidden_dim,}
            critic_params_YX = {"x_dim": T_pi * dim, "y_dim": T_pi * encoder_xdim, "hidden_dim": hidden_dim,}

            if predictive_loss == "mi" and predictive_space == "observation":
                critic_params = {"x_dim": T_pi * dim, "y_dim": T_pi * encoder_xdim, "hidden_dim": hidden_dim,}
            # train data
            if do_dca_init:
                init_weights = DCA_init(np.concatenate(X_train_ctd, axis=0), T=T_pi, d=dim, n_init=n_init,)

                if availability_mask:
                    # DCA is fit only on the original neural channels.
                    # The new availability-mask inputs start with zero influence.
                    init_weights = np.concatenate([init_weights, np.zeros_like(init_weights),], axis=0,)

                    print(f"DCA init augmented for availability mask: "
                          f"{init_weights.shape}")
            else:
                init_weights = None

            train_data = PastFutureDataset(X_train_ctd, window_size=T_pi)

            encoder_params = {
                "deterministic": deterministic,
                "n_layers": n_layers,
            }

            if kernel == "Linear":
                encoder_params["encoder_type"] = "linear"
            else:
                encoder_params["encoder_type"] = "mlp"

            cpic_kwargs = {
                "ydim": dim,
                "xdim": encoder_xdim,
                "T": T_pi,
                "mi_params": mi_params,
                "baseline_params": baseline_params,
                "encoder_params": encoder_params,
                "hidden_dim": hidden_dim,
                "beta": beta,
                "beta1": beta1,
                "beta2": beta2,
                "beta_warmup_epochs": beta_warmup_epochs,
                "beta_ramp_epochs": beta_ramp_epochs,
                "device": device,
                "predictive_space": predictive_space,
                "predictive_loss": predictive_loss,
                "reconstruction_targets": reconstruction_targets,
                "regularization_weight": regularization_weight,
            }
            if predictive_loss == "mi":
                cpic_kwargs["critic_params"] = critic_params
                if beta2 > 0:
                    cpic_kwargs["critic_params_YX"] = critic_params_YX

            CPIC_model = CPIC(**cpic_kwargs)
            CPIC_model = CPIC_model.to(device)

            # Measure CPIC training cost only
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            train_start = time.perf_counter()

            _, I_compress, I_predictive = CPIC_model.fit(
                train_data,
                init_weights=init_weights,
                epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                early_stop=num_early_stop,
                verbose=verbose,
                neuron_dropout_p=neuron_dropout_p,
                neuron_dropout_seed=neuron_dropout_seed,
                availability_mask=availability_mask,
            )

            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

            train_runtime = time.perf_counter() - train_start

            if str(device).startswith("cuda"):
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_memory_mb = 0.0

            results_runtime[dim_idx, T_pi_idx] = train_runtime
            results_peak_memory[dim_idx, T_pi_idx] = peak_memory_mb

            print(
                f"Training runtime: {train_runtime:.2f}s | "
                f"Peak GPU memory: {peak_memory_mb:.1f} MB"
            )
            
            results_MI[dim_idx, T_pi_idx, 0] = I_compress
            results_MI[dim_idx, T_pi_idx, 1] = I_predictive

            # Encode complete-neuron train/test data via CPIC.
            # For mask-aware CPIC, all neurons are explicitly marked available.
            full_neuron_mask = np.ones(neural_xdim, dtype=np.float32)

            X_train_encoder = [build_encoder_input(Xi, full_neuron_mask) for Xi in X_train_ctd]

            X_test_encoder = build_encoder_input(X_test_ctd, full_neuron_mask,)

            with torch.no_grad():
                X_train_cpic = [CPIC_model.encode(torch.from_numpy(Xi).to(torch.float).to(device)).cpu().numpy()
                                for Xi in X_train_encoder]

                X_test_cpic = CPIC_model.encode(torch.from_numpy(X_test_encoder).to(torch.float).to(device)).cpu().numpy()


            # Standard evaluation: fit and test using the complete-neuron data.
            for offset_idx in range(len(offset_vals)):
                offset = offset_vals[offset_idx]
                r2_cpic = linear_decode_r2(
                    X_train_cpic,
                    Y_train,
                    X_test_cpic,
                    Y_test,
                    decoding_window=decoding_window,
                    offset=offset,
                )
                results_r2[dim_idx, offset_idx, T_pi_idx] = r2_cpic

            # Frozen-decoder experiment:
            # fit ONCE using complete-neuron training representations.
            frozen_decoders = {}

            for offset_idx in range(len(offset_vals)):
                offset = offset_vals[offset_idx]

                frozen_decoders[offset_idx] = fit_linear_decoder(
                    X_train_cpic,
                    Y_train,
                    decoding_window=decoding_window,
                    offset=offset,
                )

            # --------------------------------------------------------
            # Fixed pseudo-session neuron-missingness evaluation.
            #
            # For each condition we select an EXACT number of missing
            # neurons and keep that same subset absent throughout both
            # the train and test recordings.
            #
            # The behavioral decoder is then fit on masked-train
            # latents and evaluated on masked-test latents.
            # --------------------------------------------------------
            for missing_idx, missing_p in enumerate(eval_missing_ps):
                if not 0.0 <= missing_p < 1.0:
                    raise ValueError(
                        f"eval missingness must satisfy 0 <= p < 1; got {missing_p}"
                    )

                n_missing = int(round(float(missing_p) * neural_xdim))

                for mask_rep in range(eval_mask_reps):
                    # Seed depends ONLY on evaluation settings, not
                    # model/dropout RNG. This guarantees identical
                    # evaluation masks for ordinary and mask-trained CPIC.
                    seed_sequence = np.random.SeedSequence(
                        [
                            int(eval_mask_seed),
                            int(round(float(missing_p) * 10000)),
                            int(mask_rep),
                        ]
                    )
                    eval_rng = np.random.default_rng(seed_sequence)

                    neuron_mask = np.ones(neural_xdim, dtype=np.float32)

                    if n_missing > 0:
                        dropped_neurons = np.sort(
                            eval_rng.choice(
                                neural_xdim,
                                size=n_missing,
                                replace=False,
                            )
                        )
                        neuron_mask[dropped_neurons] = 0.0
                    else:
                        dropped_neurons = np.array([], dtype=int)

                    X_train_eval = [build_encoder_input(Xi, neuron_mask) for Xi in X_train_ctd]
                    X_test_eval = build_encoder_input(X_test_ctd, neuron_mask,)

                    with torch.no_grad():
                        X_train_eval_cpic = [
                            CPIC_model.encode(
                                torch.from_numpy(Xi)
                                .to(torch.float)
                                .to(device)
                            ).cpu().numpy()
                            for Xi in X_train_eval
                        ]

                        X_test_eval_cpic = CPIC_model.encode(
                            torch.from_numpy(X_test_eval)
                            .to(torch.float)
                            .to(device)
                        ).cpu().numpy()

                    rep_r2 = []
                    frozen_rep_r2 = []

                    for offset_idx in range(len(offset_vals)):
                        offset = offset_vals[offset_idx]

                        # ------------------------------------------------
                        # Existing within-pseudo-session evaluation:
                        # refit the behavioral probe after neurons vanish.
                        # ------------------------------------------------
                        r2_masked = linear_decode_r2(
                            X_train_eval_cpic,
                            Y_train,
                            X_test_eval_cpic,
                            Y_test,
                            decoding_window=decoding_window,
                            offset=offset,
                        )

                        results_masked_r2[
                            dim_idx,
                            missing_idx,
                            mask_rep,
                            offset_idx,
                            T_pi_idx,
                        ] = r2_masked

                        rep_r2.append(float(r2_masked))

                        # ------------------------------------------------
                        # NEW: frozen clean -> masked transfer.
                        #
                        # This decoder was trained only on COMPLETE-neuron
                        # training latents. It never sees masked training
                        # representations and is not refit here.
                        # ------------------------------------------------
                        frozen_r2 = score_linear_decoder(
                            frozen_decoders[offset_idx],
                            X_test_eval_cpic,
                            Y_test,
                            decoding_window=decoding_window,
                            offset=offset,
                        )

                        results_frozen_masked_r2[
                            dim_idx,
                            missing_idx,
                            mask_rep,
                            offset_idx,
                            T_pi_idx,
                        ] = frozen_r2

                        frozen_rep_r2.append(float(frozen_r2))

                    print(
                        f"Mask eval | missing={missing_p:.2f} "
                        f"| rep={mask_rep} "
                        f"| dropped={n_missing}/{neural_xdim} "
                        f"| refit_mean_R2={np.mean(rep_r2):.6f} "
                        f"| frozen_mean_R2={np.mean(frozen_rep_r2):.6f}"
                    )

                    print(
                        "  refit R2:",
                        np.round(rep_r2, 6).tolist()
                    )

                    print(
                        "  frozen R2:",
                        np.round(frozen_rep_r2, 6).tolist()
                    )

                    if n_missing > 0:
                        print(
                            "  dropped neuron indices:",
                            dropped_neurons.tolist()
                        )

        print("dim_idx: {}, R2: {}".format(dim_vals[dim_idx], results_r2[dim_idx]))
    return (
        results_r2,
        results_MI,
        results_runtime,
        results_peak_memory,
        results_masked_r2,
        results_frozen_masked_r2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC for real data.')
    parser.add_argument('--config', type=str, default="hc_stochastic_infonce_alt")
    parser.add_argument('--model', type=str, default="CPIC")
    parser.add_argument('--ydim', type=int, default=None, help='Override latent dimension')
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible experiments")
    parser.add_argument(
        "--kernel",
        choices=["Linear", "MLP"],
        default=None,
        help="Override the configured encoder architecture",
    )
    parser.add_argument(
        "--neuron-dropout-p",
        type=float,
        default=0.0,
        help="Probability of dropping each neuron during CPIC training",
    )
    parser.add_argument(
        "--train-mask-seed",
        type=int,
        default=10000,
        help="Independent RNG seed for training neuron dropout",
    )
    parser.add_argument(
        "--eval-missing-ps",
        type=str,
        default="0,0.1,0.25,0.5",
        help="Comma-separated fixed test missing-neuron fractions",
    )
    parser.add_argument(
        "--eval-mask-reps",
        type=int,
        default=3,
        help="Number of fixed neuron-subset replicates per missingness level",
    )
    parser.add_argument(
        "--eval-mask-seed",
        type=int,
        default=4242,
        help="Seed defining fixed pseudo-session evaluation masks",
    )

    parser.add_argument(
        "--availability-mask",
        action="store_true",
        help=(
            "Append a binary neuron-availability mask to CPIC input. "
            "Missing neural values remain zero-filled."
        ),
    )

    args = parser.parse_args()

    eval_missing_ps = tuple(
        float(x.strip())
        for x in args.eval_missing_ps.split(",")
        if x.strip()
    )

    if len(eval_missing_ps) == 0:
        raise ValueError("--eval-missing-ps must contain at least one value")

    if any((p < 0.0 or p >= 1.0) for p in eval_missing_ps):
        raise ValueError(
            f"All evaluation missingness values must satisfy 0 <= p < 1; "
            f"got {eval_missing_ps}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using random seed: {args.seed}")

    if args.model == "CPIC":
        from cpic.CPIC import CPIC
        from cpic.utils.data import PastFutureDataset
        from cpic.utils.helpers import DCA_init, Polynomial_expand, resolve_device

    if args.model == "PFPC_RC":
        from PFPC_RC import PastFutureDataset, train_CPIC, DCA_init, Polynomial_expand

    # Keep original config-selection logic
    if args.config == 'm1_stochastic_infonce':
        config_file = 'config/config_m1_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'm1_scaling_benchmark':
        config_file = 'config/config_m1_scaling_benchmark.ini'
        ydims = np.array([32]).astype(int)

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

    elif args.config == 'hc_reconstruction_alt':
        config_file = 'config/config_hc_reconstruction_alt.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'temp_stochastic_infonce':
        config_file = 'config/config_temp_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'temp_stochastic_infonce_alt':
        config_file = 'config/config_temp_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'temp_deterministic_infonce_alt':
        config_file = 'config/config_temp_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'ms_stochastic_infonce':
        config_file = 'config/config_ms_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'ms_stochastic_infonce_alt':
        config_file = 'config/config_ms_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)

    elif args.config == 'ms_deterministic_infonce_alt':
        config_file = 'config/config_ms_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)

    else:
        raise ValueError("{} has not been implemented!".format(args.config))

    cfg = myconf()
    cfg.read(config_file)

    if args.ydim is not None:
        ydims = np.array([args.ydim]).astype(int)
    else:
        ydims = np.array([
            cfg.getint('Hyperparameters', 'ydim')
        ]).astype(int)

    print("Using latent dimensions:", ydims)

    RESULTS_FILENAME = cfg.get('User', 'RESULTS_FILENAME')
    saved_root = cfg.get('User', 'saved_root')

    if not os.path.exists(saved_root):
        os.makedirs(saved_root, exist_ok=True)

    # set hyper-parameters
    beta = cfg.getfloat('Hyperparameters', 'beta')
    beta1 = cfg.getfloat('Hyperparameters', 'beta1')
    beta2 = cfg.getfloat('Hyperparameters', 'beta2')
    beta_warmup_epochs = cfg.getint('Hyperparameters', 'beta_warmup_epochs', fallback=0)
    beta_ramp_epochs = cfg.getint('Hyperparameters', 'beta_ramp_epochs', fallback=0)
    xdim = cfg.getint('Hyperparameters', 'xdim')
    Ts = cfg.get('Hyperparameters', 'T')
    Ts = Ts.split(' ')
    Ts = [int(T) for T in Ts]
    # import pdb; pdb.set_trace()
    if cfg.has_option('Hyperparameters', 'n_layers'):
        n_layers = cfg.getint('Hyperparameters', 'n_layers')
    else:
        n_layers = 1

    hidden_dim = (
        args.hidden_dim
        if args.hidden_dim is not None
        else cfg.getint('Hyperparameters', 'hidden_dim')
    )

    n_layers = (
        args.n_layers
        if args.n_layers is not None
        else cfg.getint('Hyperparameters', 'n_layers', fallback=1)
    )

    estimator_compress = cfg.get('Hyperparameters', 'estimator_compress')
    if cfg.has_option('Hyperparameters', 'estimator_predictive'):
        estimator_predictive = cfg.get('Hyperparameters', 'estimator_predictive')
        critic = cfg.get('Hyperparameters', 'critic')
        baseline = cfg.get('Hyperparameters', 'baseline')
    else:
        estimator_predictive = 'infonce_lower'
        critic = 'concat'
        baseline = 'constant'
    kernel = (
        args.kernel
        if args.kernel is not None
        else cfg.get('Hyperparameters', 'kernel')
    )
    mi_params = {'estimator_compress': estimator_compress}
    if cfg.has_option('Hyperparameters', 'predictive_loss'):
        predictive_loss = cfg.get('Hyperparameters', 'predictive_loss')
    else:
        predictive_loss = 'mi'
    if predictive_loss == 'mi':
        mi_params.update({
            'estimator_predictive': estimator_predictive,
            'critic': critic,
            'baseline': baseline,
        })
    if cfg.has_option('Hyperparameters', 'reconstruction_targets'):
        reconstruction_targets = tuple(cfg.get('Hyperparameters', 'reconstruction_targets').split())
    else:
        reconstruction_targets = ('past',)
    if cfg.has_option('Hyperparameters', 'predictive_space'):
        predictive_space = cfg.get('Hyperparameters', 'predictive_space')
    else:
        predictive_space = 'latent'
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

    if kernel == "MLP":
        do_dca_init = False
    device = resolve_device(cfg.get('Training', 'device'))
    if device != cfg.get('Training', 'device'):
        print(f"Using device {device!r} (config requested {cfg.get('Training', 'device')!r})")
    lr = cfg.getfloat('Training', 'lr')

    if args.config in (
        "m1_stochastic_infonce",
        "m1_stochastic_infonce_alt",
        "m1_deterministic_infonce_alt",
        "m1_scaling_benchmark",
    ):
        M1 = data_util.load_sabes_data(data_path('data', 'real_data', 'M1', 'indy_20160627_01.mat'))
        X, Y = M1['M1'], M1['cursor']
        good_ts = None
        standardize_Y = False
    if args.config == "hc_stochastic_infonce" or args.config == "hc_stochastic_infonce_alt"\
            or args.config == "hc_deterministic_infonce_alt" or args.config == "hc_reconstruction_alt":
        HC = data_util.load_kording_paper_data(data_path('data', 'real_data', 'HC', 'example_data_hc.pickle'))
        X, Y = HC['neural'], HC['loc']
        good_ts = 22000
        # good_ts = None
        standardize_Y = False
    if args.config == "temp_stochastic_infonce" or args.config == "temp_stochastic_infonce_alt"\
            or args.config == "temp_deterministic_infonce_alt":
        weather = data_util.load_weather_data(data_path('data', 'real_data', 'TEMP', 'temperature.csv'))
        X, Y = weather, weather
        good_ts = None
        standardize_Y = True
    if args.config == "ms_stochatic_infonce" or args.config == "ms_stochastic_infonce_alt"\
            or args.config == "ms_deterministic_infonce_alt":
        ms = data_util.load_accel_data(data_path('data', 'real_data', 'motion_sense', 'A_DeviceMotion_data', 'std_6', 'sub_19.csv'))
        X, Y = ms, ms
        good_ts = None
        standardize_Y = True

    T_pi_vals = np.array(Ts)
    offsets = np.array([5, 10, 15])

    win = 3
    n_init = 5

    # rewrite ydim
    # m1_ydims = np.array([5,10,20,30])
    # hc_ydims = np.array([5,10,15,25])
    # temp_ydims = np.array([3, 4, 5, 6])
    # ydims = np.array([5])
    if kernel == "Linear":
        Kernel = None
    elif kernel == "Polynomial":
        Kernel = Polynomial_expand
    elif kernel == "MLP":
        Kernel = None
    else:
        raise ValueError("This kernel is not available.")

    for ydim in ydims:
        regularzation_weight = 0
        (
            result_r2,
            result_MI,
            result_runtime,
            result_peak_memory,
            result_masked_r2,
            result_frozen_masked_r2,
        ) = run_analysis_cpic(X, Y, T_pi_vals, dim_vals=[ydim], offset_vals=offsets, decoding_window=win,
                          n_init=n_init, verbose=True, Kernel=Kernel, xdim=xdim, beta=beta, beta1=beta1, beta2=beta2,
                          good_ts=good_ts, standardize_Y=standardize_Y, regularization_weight=regularzation_weight,
                          predictive_loss=predictive_loss, reconstruction_targets=reconstruction_targets,
                          predictive_space=predictive_space, hidden_dim=hidden_dim, n_layers=n_layers,
                          neuron_dropout_p=args.neuron_dropout_p,
                          neuron_dropout_seed=args.train_mask_seed,
                          eval_missing_ps=eval_missing_ps,
                          eval_mask_reps=args.eval_mask_reps,
                          eval_mask_seed=args.eval_mask_seed,
                          availability_mask=args.availability_mask)

        result = {
            "ydim": int(ydim),
            "seed": int(args.seed),

            "r2": result_r2,

            # Behavioral decoder refit separately for each
            # pseudo-session neuron population.
            "mask_eval_r2": result_masked_r2,

            # Behavioral decoder trained once using full-neuron
            # representations and frozen across test masks.
            "frozen_mask_eval_r2": result_frozen_masked_r2,

            "mi": result_MI,

            "masking": {
                "train_neuron_dropout_p": float(args.neuron_dropout_p),
                "train_mask_seed": int(args.train_mask_seed),
                "eval_missing_ps": [float(p) for p in eval_missing_ps],
                "eval_mask_reps": int(args.eval_mask_reps),
                "eval_mask_seed": int(args.eval_mask_seed),
                "availability_mask": bool(args.availability_mask),
            },

            "train_runtime_seconds": result_runtime,
            "peak_gpu_memory_mb": result_peak_memory,

            "config": {
                "dataset": "indy_20160627_01",
                "xdim": int(xdim),
                "T": list(T_pi_vals),
                "hidden_dim": int(hidden_dim),
                "n_layers": int(n_layers),
                "beta": float(beta),
                "batch_size": int(batch_size),
                "num_epochs": int(num_epochs),
                "lr": float(lr),
                "encoder_xdim": int(xdim * (2 if args.availability_mask else 1)),
            },
        }

        run_tag = f"{kernel.lower()}_avail{int(args.availability_mask)}"

        output_file = os.path.join(
            saved_root,
            f"result_{run_tag}_ydim{ydim}_seed{args.seed}"
            f"_traindrop{int(round(args.neuron_dropout_p * 100)):03d}.pkl"
        )

        result["config"].update({
            "encoder_kernel": kernel,
            "availability_mask": bool(args.availability_mask),
            "do_dca_init": bool(do_dca_init),
        })

        with open(output_file, "wb") as f:
            pickle.dump(result, f)

        print(f"Saved benchmark result to: {output_file}")
        # import pdb; pdb.set_trace()
