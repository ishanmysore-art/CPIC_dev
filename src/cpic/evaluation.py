"""
Held-out evaluation for CPIC: predictive-information bounds on frozen models and
linear behavioral probes.

Behavior enters only through the probe functions below, which are applied after the
representation is frozen. ``heldout_bounds`` never updates parameters or buffers
(e.g. BatchNorm running statistics) and leaves the global torch RNG untouched.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score

from .exp_utils.cov_util import form_lag_matrix
from .mi import infonce_lower_bound


def _window_batches(n_windows, batch_size, seed, max_batches=None):
    """Seeded, full-size batches of window start indices (the partial remainder is dropped)."""
    n_batches = n_windows // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, int(max_batches))
    if n_batches == 0:
        raise ValueError(f"Need at least {batch_size} windows; got {n_windows}")
    order = np.random.default_rng(seed).permutation(n_windows)
    return order[:n_batches * batch_size].reshape(n_batches, batch_size)


def heldout_bounds(model, X, batch_size=512, seed=0, max_batches=None, sample_latents=True):
    """
    InfoNCE predictive bound (and compression bound) of a frozen CPIC model on one segment.

    Windows follow ``PastFutureDataset``: past = X[i:i+T], future = X[i+T:i+2T]. Negatives
    are the other windows of the same batch. Every batch has exactly ``batch_size`` windows,
    so every batch shares the log(batch_size) ceiling.

    Parameters
    ----------
    model : CPIC
        Trained model with ``predictive_loss="mi"`` and ``predictive_space="latent"``.
    X : array (n_bins, n_features)
        One contiguous, already-preprocessed segment (e.g. the test split).
    sample_latents : bool
        True scores sampled latents exactly as in training (``model.forward``); False scores
        the deterministic encoder means with the same critic (no compression bound).
    """
    if model.predictive_loss != "mi" or model.predictive_space != "latent":
        raise ValueError("heldout_bounds requires predictive_loss='mi' and predictive_space='latent'")
    T = model.T
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=model.device)
    n_windows = len(X) - 2 * T
    batches = _window_batches(n_windows, batch_size, seed, max_batches)
    offsets = torch.arange(2 * T, device=model.device)
    devices = [model.device] if str(model.device).startswith("cuda") else []

    I_predictive, I_compress = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for idx in batches:
                windows = X[torch.as_tensor(idx, device=model.device)[:, None] + offsets]
                past, future = windows[:, :T], windows[:, T:]
                if sample_latents:
                    _, i_compress, i_predictive = model(past, future)
                    I_compress.append(float(i_compress))
                else:
                    z_past = model.encoder.get_mean(past).reshape(batch_size, -1)
                    z_future = model.encoder.get_mean(future).reshape(batch_size, -1)
                    i_predictive = infonce_lower_bound(model.critic(z_past, z_future))
                I_predictive.append(float(i_predictive))
    finally:
        model.train(was_training)

    I_predictive = np.asarray(I_predictive)
    result = {
        "I_predictive_mean": float(I_predictive.mean()),
        "I_predictive_sd": float(I_predictive.std(ddof=1)) if len(I_predictive) > 1 else float("nan"),
        "I_predictive_per_batch": I_predictive,
        "n_batches": int(len(batches)),
        "batch_size": int(batch_size),
        "log_batch_size_ceiling": float(np.log(batch_size)),
        "n_windows": int(n_windows),
        "sample_latents": bool(sample_latents),
        "seed": int(seed),
    }
    if I_compress:
        result["I_compress_mean"] = float(np.mean(I_compress))
    return result


def encode(model, X):
    """Deterministic latent means for a preprocessed (n_bins, n_features) segment."""
    with torch.no_grad():
        Z = model.encode(torch.as_tensor(np.asarray(X), dtype=torch.float32, device=model.device))
    return Z.cpu().numpy()


def cursor_velocity(position, bin_width_s):
    """Finite-difference velocity of a (n_bins, n_dims) kinematic trace (evaluation target only)."""
    return np.gradient(np.asarray(position, dtype=np.float64), axis=0) / bin_width_s


def prepare_linear_decode_data(X, Y, decoding_window=1, offset=0):
    """Lag/offset alignment identical to the legacy ``linear_decode_r2`` (single segment)."""
    X_lags = form_lag_matrix(np.asarray(X), decoding_window)
    Y_aligned = np.asarray(Y)[decoding_window // 2:][:len(X_lags)]
    if offset >= 0:
        return X_lags[:X_lags.shape[0] - offset], Y_aligned[offset:]
    return X_lags[-offset:], Y_aligned[:Y_aligned.shape[0] + offset]


def fit_linear_decoder(Z_train, Y_train, decoding_window=1, offset=0):
    design, targets = prepare_linear_decode_data(Z_train, Y_train, decoding_window, offset)
    return LR().fit(design, targets)


def score_linear_decoder(decoder, Z_test, Y_test, decoding_window=1, offset=0):
    """R² (uniform average over outputs) of an already-fitted decoder; nothing is refit."""
    design, targets = prepare_linear_decode_data(Z_test, Y_test, decoding_window, offset)
    return float(decoder.score(design, targets))


def block_bootstrap_r2(decoder, Z_test, Y_test, decoding_window=1, offset=0,
                       block_bins=1200, n_boot=1000, seed=0, ci=0.95):
    """
    Point R² plus a contiguous-block bootstrap CI over the test segment.

    The CI reflects within-session temporal sampling only; it says nothing about
    variability across sessions.
    """
    design, targets = prepare_linear_decode_data(Z_test, Y_test, decoding_window, offset)
    predictions = decoder.predict(design)
    blocks = np.array_split(np.arange(len(targets)), max(1, len(targets) // block_bins))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        boot[b] = r2_score(targets[idx], predictions[idx])
    alpha = (1 - ci) / 2
    return {"r2": float(r2_score(targets, predictions)),
            "ci_low": float(np.quantile(boot, alpha)),
            "ci_high": float(np.quantile(boot, 1 - alpha)),
            "n_blocks": len(blocks), "block_bins": int(block_bins)}


def probe_suite(Z_fit, Y_fit, Z_eval, Y_eval, offsets=(5, 10, 15), decoding_window=3):
    """
    Fit one linear decoder per offset on (Z_fit, Y_fit) and score it on (Z_eval, Y_eval).

    Which claim the score supports depends on where Z_fit and Z_eval come from: the same
    representation and session (refit probe), clean source latents vs. corrupted or
    target-session latents (frozen decoder), etc. Callers must label it accordingly.
    """
    decoders = {int(o): fit_linear_decoder(Z_fit, Y_fit, decoding_window, int(o)) for o in offsets}
    scores = {o: score_linear_decoder(d, Z_eval, Y_eval, decoding_window, o)
              for o, d in decoders.items()}
    return {"r2_by_offset": scores, "r2_mean": float(np.mean(list(scores.values()))),
            "decoders": decoders}
