"""
Sparse CPIC on flattened video frames (.npz / .npy), same training pattern as
``synthetic/synthetic_sparse_experiment.py`` without HDF5 / SNR sweeps.

HDF5 inputs were removed; use ``synthetic/synthetic_sparse_experiment.py`` for that layout.

Usage::

    python experiments/video_experiment/run_sparse_cpic.py \\
        --config experiments/video_experiment/config/config_video_sparse_cpic.ini
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_ROOT = _REPO_ROOT / "synthetic"
if str(_SYNTHETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYNTHETIC_ROOT))

from cpic import SparseCPIC
from cpic.utils.data import PastFutureDataset
from cpic.utils.helpers import DCA_init
from utils.data_util import compute_R2, linear_alignment

_EXPERIMENT_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _EXPERIMENT_ROOT / "config"


class myconf(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    if not p.parts:
        raise ValueError("Empty path in config")
    if p.is_absolute():
        return p
    cand = _EXPERIMENT_ROOT / p
    return cand.resolve() if cand.exists() else (_REPO_ROOT / p).resolve()


def _opt_int(cfg: myconf, section: str, key: str) -> int | None:
    if not cfg.has_section(section) or not cfg.has_option(section, key):
        return None
    v = cfg.get(section, key).strip()
    return int(v) if v else None


def _enc_params(cfg: myconf) -> dict:
    h = "Hyperparameters"
    g = cfg.getboolean
    gi = cfg.getint
    go = lambda k, d=None: cfg.get(h, k) if cfg.has_option(h, k) else d
    return {
        "deterministic": g(h, "deterministic"),
        "linear_encoder": g(h, "linear_encoding") if cfg.has_option(h, "linear_encoding") else True,
        "encoder_type": go("encoder_type", "mlp"),
        "n_layers": gi(h, "n_layers") if cfg.has_option(h, "n_layers") else 1,
        "activation": go("activation", "relu"),
        "conv_kernel_size": gi(h, "conv_kernel_size") if cfg.has_option(h, "conv_kernel_size") else 3,
        "conv_stride": gi(h, "conv_stride") if cfg.has_option(h, "conv_stride") else 1,
        "conv_padding": gi(h, "conv_padding") if cfg.has_option(h, "conv_padding") else 1,
    }


def _sparse_params(cfg: myconf) -> dict:
    h = "Hyperparameters"
    xdim, ydim, T = cfg.getint(h, "xdim"), cfg.getint(h, "ydim"), cfg.getint(h, "T")
    hidden = cfg.getint(h, "hidden_dim")
    mi = {
        "estimator_compress": cfg.get(h, "estimator_compress"),
        "estimator_predictive": cfg.get(h, "estimator_predictive"),
        "critic": cfg.get(h, "critic"),
        "baseline": cfg.get(h, "baseline"),
    }
    ps = cfg.get(h, "predictive_space")
    if ps == "latent":
        critic_p = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden}
    elif ps == "observation":
        critic_p = {"x_dim": T * ydim, "y_dim": T * xdim, "hidden_dim": hidden}
    else:
        raise ValueError(ps)
    return {
        "beta": cfg.getfloat(h, "beta"),
        "xdim": xdim,
        "ydim": ydim,
        "T": T,
        "hidden_dim": hidden,
        "mi_params": mi,
        "critic_params": critic_p,
        "baseline_params": {"hidden_dim": hidden},
        "predictive_space": ps,
        "gamma": cfg.getfloat(h, "gamma"),
    }


def _load_video(cfg: myconf) -> np.ndarray:
    path = _resolve_path(cfg.get("User", "video_path"))
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        frames = z["frames"] if "frames" in z.files else z[z.files[0]]
    elif path.suffix.lower() == ".npy":
        frames = np.asarray(np.load(path, mmap_mode="r"))
    else:
        raise ValueError("Use .npz or .npy")

    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C), got {frames.shape}")

    mf = _opt_int(cfg, "Data", "max_frames")
    if mf is not None:
        frames = frames[:mf]

    stride = _opt_int(cfg, "Data", "frame_stride")
    if stride is not None:
        if stride < 1:
            raise ValueError("Data.frame_stride must be >= 1")
        if stride > 1:
            frames = frames[::stride]
    if frames.shape[0] == 0:
        raise ValueError("No frames left after max_frames / frame_stride; relax subsampling.")

    rh, rw = _opt_int(cfg, "Data", "resize_height"), _opt_int(cfg, "Data", "resize_width")
    if rh is not None and rw is not None:
        t = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
        frames = (t.permute(0, 2, 3, 1).clamp(0, 1) * 255.0).round().detach().cpu().numpy().astype(np.uint8)

    T, H, W, C = frames.shape
    x = frames.reshape(T, H * W * C).astype(np.float32)
    if not (cfg.has_section("Data") and cfg.has_option("Data", "normalize")) or cfg.getboolean("Data", "normalize"):
        x /= 255.0
    return x


def _log_memory_context(X: np.ndarray, T_win: int, batch_size: int, device: str) -> None:
    """Explain RAM drivers; SIGKILL (exit 137) on macOS often follows huge preallocated arrays."""
    T, xdim = X.shape[0], X.shape[1]
    n_win = T - 2 * T_win
    raw_gib = X.nbytes / (1024**3)
    # Old PastFutureDataset pre-stacked past_ts and future_ts (same dtype as X).
    materialized_extra_gib = (
        2 * n_win * T_win * xdim * np.dtype(np.float32).itemsize / (1024**3)
    )
    batch_payload_gib = (
        2 * batch_size * T_win * xdim * np.dtype(np.float32).itemsize / (1024**3)
    )
    print(
        f"RAM context: X is {raw_gib:.2f} GiB ({T}×{xdim} float32); "
        f"{n_win} sliding windows (lazy dataset, no {materialized_extra_gib:.1f} GiB stack)."
    )
    print(
        f"  Each train batch moves ~{batch_payload_gib:.3f} GiB of past+future values to "
        f"{device} before conv/MI (lower batch_size if GPU/driver OOMs)."
    )


def _resolve_config_arg(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    for alt in (
        _EXPERIMENT_ROOT / arg,
        _CONFIG_DIR / arg,
        _CONFIG_DIR / Path(arg).name,
        _REPO_ROOT / arg,
    ):
        if alt.is_file():
            return alt
    raise FileNotFoundError(arg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sparse CPIC on video frames.")
    ap.add_argument("--config", type=str, default=str(_CONFIG_DIR / "config_video_sparse_cpic.ini"))
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--signature", type=int, default=22)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = myconf()
    cfg.read(_resolve_config_arg(args.config))

    if cfg.has_option("User", "data_source") and cfg.get("User", "data_source").strip().lower() == "hdf5":
        raise ValueError(
            "data_source=hdf5 is not supported in this script; use synthetic/synthetic_sparse_experiment.py."
        )

    saved_root = str(_resolve_path(cfg.get("User", "saved_root")))
    os.makedirs(saved_root, exist_ok=True)

    # load the video data
    X_noisy = _load_video(cfg)

    # Check if xdim is set in the config, if not, set it from X_noisy.shape[-1]
    xdim_cfg_val = cfg.get("Hyperparameters", "xdim")
    if (xdim_cfg_val is None) or (str(xdim_cfg_val).strip() == ""):
        xdim = X_noisy.shape[-1]
        cfg.set("Hyperparameters", "xdim", str(xdim))

    p = _sparse_params(cfg)
    ydim, xdim, T = p["ydim"], p["xdim"], p["T"]
    enc = _enc_params(cfg)

    tr = "Training"
    batch_size = cfg.getint(tr, "batch_size")
    num_epochs = cfg.getint(tr, "num_epochs")
    num_early_stop = cfg.getint(tr, "num_early_stop")
    do_dca_init = cfg.getboolean(tr, "do_dca_init")
    lr = cfg.getfloat(tr, "lr")
    device = "cpu" if not torch.cuda.is_available() else (args.device or cfg.get(tr, "device"))
    print(f"Device: {device}")

    if X_noisy.shape[-1] != xdim:
        raise AssertionError(f"data xdim {X_noisy.shape[-1]} != config xdim {xdim}")
    if X_noisy.shape[0] <= 2 * T:
        raise ValueError(f"Need length > 2*T ({2 * T}), got {X_noisy.shape[0]}")

    _log_memory_context(X_noisy, T_win=T, batch_size=batch_size, device=device)

    train_data = PastFutureDataset([X_noisy], window_size=T)
    init_weights = DCA_init(X_noisy, T=T, d=ydim, rng_or_seed=args.seed) if do_dca_init else None

    # Print train data information
    print(f"Train data shape: {X_noisy.shape}")
    print(f"PastFutureDataset: {len(train_data)} samples, window_size={T}")

    # Print initial weights information
    if init_weights is not None:
        if isinstance(init_weights, dict):
            print("DCA_init returned a dict with keys:", list(init_weights.keys()))
            for key, val in init_weights.items():
                print(f"  - {key}: {type(val)} shape {getattr(val, 'shape', None)}")
        else:
            print("DCA_init returned:", type(init_weights))
            if hasattr(init_weights, 'shape'):
                print("  - shape:", init_weights.shape)
    else:
        print("DCA_init not used (init_weights is None)")

    model = SparseCPIC(
        xdim=xdim,
        ydim=ydim,
        mi_params=p["mi_params"],
        critic_params=p["critic_params"],
        baseline_params=p["baseline_params"],
        encoder_params=enc,
        T=T,
        hidden_dim=p["hidden_dim"],
        beta=p["beta"],
        gamma=p["gamma"],
        device=device,
        predictive_space=p["predictive_space"],
    ).to(device)

    log_dir = os.path.join(saved_root, "tensor_logs", str(args.signature))
    os.makedirs(log_dir, exist_ok=True)
    loss, _, _, _ = model.fit(
        X=train_data,
        init_weights=init_weights,
        epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        early_stop=num_early_stop,
        writer=SummaryWriter(log_dir=log_dir),
    )

    et = getattr(model.encoder, "encoder_type", None)
    if et in ("conv_spatial", "conv_spatiotemporal", "conv_temporal"):
        model.visualize_kernels(kernel_save_suffix="video_sparse", signature=args.signature)

    if et == "conv_spatiotemporal":
        past = np.stack([X_noisy[t - T : t] for t in range(T, len(X_noisy))], axis=0)
        past_t = torch.from_numpy(past).float().to(device)
        centers = [t - 1 for t in range(T, len(X_noisy))]
        y_obs = X_noisy[centers]
        enc_out = model.encode(past_t)
        encoded_repr = enc_out[:, -1, :]
    else:
        encoded_repr = model.encode(torch.from_numpy(X_noisy).float().to(device))
        y_obs = X_noisy

    aligned = linear_alignment(encoded_repr.detach().cpu().numpy(), y_obs)
    r2 = compute_R2(aligned, y_obs)
    print(f"R2(sparse CPIC vs observations, linearly aligned): {r2}")

    snr_vals = np.array([0.0], dtype=np.float32)
    seed_str = f"_seed{args.seed}"
    payload = {"snr_vals": snr_vals, "losses": [loss], "R2_metrics": np.array([[np.nan, np.nan, r2]])}
    with open(os.path.join(saved_root, f"encoded_representations{seed_str}.pkl"), "wb") as f:
        pickle.dump({"encoded_representations": [encoded_repr.detach().cpu().numpy()], "snr_vals": snr_vals}, f)
    with open(os.path.join(saved_root, f"latent_R2{seed_str}.pkl"), "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(saved_root, f"inferred_trials{seed_str}.pkl"), "wb") as f:
        pickle.dump({"inferred_sparse_CPIC_trials": [aligned], "snr_vals": snr_vals}, f)
