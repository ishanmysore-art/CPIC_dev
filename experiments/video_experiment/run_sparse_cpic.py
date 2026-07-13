"""
Sparse CPIC on flattened video frames (.npz / .npy), same training pattern as
``synthetic/synthetic_sparse_experiment.py`` without HDF5 / SNR sweeps.

HDF5 inputs were removed; use ``synthetic/synthetic_sparse_experiment.py`` for that layout.

Usage::

    python experiments/video_experiment/run_sparse_cpic.py --config experiments/video_experiment/config/config_video_sparse_cpic.ini
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cpic import SparseCPIC
from cpic.utils.data import PastFutureDataset
from cpic.utils.helpers import DCA_init
from experiments.synthetic_lorenz_experiment.utils.data_util import compute_R2, linear_alignment
from experiments.video_experiment.patch_utils import extract_patch_series, patch_mode_on

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


def _parse_path_list(raw: str) -> list[str]:
    """Split a config path field on commas / newlines; ignore blanks."""
    parts: list[str] = []
    for chunk in raw.replace(",", "\n").splitlines():
        s = chunk.strip()
        if s:
            parts.append(s)
    return parts


def _resolve_video_paths(cfg: myconf) -> list[Path]:
    """Resolve one or more video files from ``[User]``.

    Compatible with a single ``video_path``. For multiple videos, either:
      - set ``video_paths`` to a comma/newline-separated list, or
      - put multiple comma-separated entries in ``video_path``.
    If ``video_paths`` is present and non-empty, it takes precedence.
    """
    raw = ""
    if cfg.has_option("User", "video_paths"):
        raw = cfg.get("User", "video_paths").strip()
    if not raw:
        if not cfg.has_option("User", "video_path"):
            raise ValueError("Set [User] video_path (single or comma-separated) or video_paths.")
        raw = cfg.get("User", "video_path").strip()
    if not raw:
        raise ValueError("Empty [User] video_path / video_paths.")

    paths = [_resolve_path(p) for p in _parse_path_list(raw)]
    if not paths:
        raise ValueError("No video paths resolved from config.")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing video file(s): {missing}")
    return paths


def _load_frames(path: Path, cfg: myconf) -> np.ndarray:
    """Load one ``.npz`` / ``.npy`` video and apply max_frames / stride / resize.

    Returns:
        frames of shape ``(T, H, W, C)`` (uint8 after optional resize).
    """
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        frames = z["frames"] if "frames" in z.files else z[z.files[0]]
    elif path.suffix.lower() == ".npy":
        frames = np.asarray(np.load(path, mmap_mode="r"))
    else:
        raise ValueError(f"Use .npz or .npy; got {path}")

    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C), got {frames.shape} for {path}")

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
        raise ValueError(f"No frames left after max_frames / frame_stride for {path}.")

    rh, rw = _opt_int(cfg, "Data", "resize_height"), _opt_int(cfg, "Data", "resize_width")
    if rh is not None and rw is not None:
        t = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
        frames = (t.permute(0, 2, 3, 1).clamp(0, 1) * 255.0).round().detach().cpu().numpy().astype(np.uint8)
    return frames


def _frames_to_series(frames: np.ndarray, cfg: myconf) -> list[np.ndarray]:
    """Convert one video ``(T,H,W,C)`` into one or more observation time series.

    Patch mode: ``num_patches`` series of shape ``(T, patch_dim)``.
    Whole-frame: a single series of shape ``(T, H*W*C)``.
    """
    if patch_mode_on(cfg):
        return extract_patch_series(frames, cfg)

    T, H, W, C = frames.shape
    x = frames.reshape(T, H * W * C).astype(np.float32)
    if not (cfg.has_section("Data") and cfg.has_option("Data", "normalize")) or cfg.getboolean("Data", "normalize"):
        x /= 255.0
    return [x]


def _load_video(cfg: myconf) -> list[np.ndarray]:
    """Load one or more videos into a flat list of time series for ``PastFutureDataset``.

    Pipeline per video: frames ``(T,H,W,C)`` -> patch series (or one whole-frame series).
    All series from all videos are concatenated into one list (windows never cross
    series / video boundaries).

    Returns:
        ``list`` of arrays, each ``(T_i, obs_dim)``. Length is ``n_videos`` (whole-frame)
        or ``n_videos * num_patches`` (patch mode). A single ``video_path`` still works.
    """
    paths = _resolve_video_paths(cfg)
    series_list: list[np.ndarray] = []
    for path in paths:
        frames = _load_frames(path, cfg)
        series_list.extend(_frames_to_series(frames, cfg))
    if not series_list:
        raise ValueError("No time series produced from configured video path(s).")

    obs_dims = {s.shape[-1] for s in series_list}
    if len(obs_dims) != 1:
        raise ValueError(
            f"All series must share the same obs dim; got {sorted(obs_dims)}. "
            "Use the same resize / patch settings for every video."
        )
    print(f"Loaded {len(paths)} video(s) -> {len(series_list)} series "
          f"(obs_dim={next(iter(obs_dims))}).")
    return series_list


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
    ap.add_argument(
        "--signature",
        type=int,
        default=int(datetime.now().strftime("%Y%m%d%H%M%S")),
        help="Run id for tensor_logs and checkpoints (default: local wall time as YYYYMMDDHHMMSS).",
    )
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

    # load the video data (one or many videos -> flat list of series)
    series_list = _load_video(cfg)
    patch_mode = patch_mode_on(cfg)
    multi_series = len(series_list) > 1
    obs_dim = series_list[0].shape[-1]  # P*P(*C) for patches, H*W*C otherwise

    # Check if xdim is set in the config, if not, set it from the observation dim.
    xdim_cfg_val = cfg.get("Hyperparameters", "xdim")
    if (xdim_cfg_val is None) or (str(xdim_cfg_val).strip() == ""):
        xdim = obs_dim
        cfg.set("Hyperparameters", "xdim", str(xdim))

    p = _sparse_params(cfg)
    ydim, xdim, T = p["ydim"], p["xdim"], p["T"]
    enc = _enc_params(cfg)

    tr = "Training"
    batch_size = cfg.getint(tr, "batch_size")
    num_epochs = cfg.getint(tr, "num_epochs")
    num_early_stop = cfg.getint(tr, "num_early_stop")
    do_dca_init = cfg.getboolean(tr, "do_dca_init")
    decoder_loss_warmup = cfg.getboolean(tr, "decoder_loss_warmup") if cfg.has_option(tr, "decoder_loss_warmup") else False
    decoder_loss_warmup_epochs = cfg.getint(tr, "decoder_loss_warmup_epochs") if cfg.has_option(tr, "decoder_loss_warmup_epochs") else None
    lr = cfg.getfloat(tr, "lr")
    device = "cpu" if not torch.cuda.is_available() else (args.device or cfg.get(tr, "device"))
    print(f"Device: {device}")

    if obs_dim != xdim:
        raise AssertionError(f"data xdim {obs_dim} != config xdim {xdim}")
    if any(s.shape[0] <= 2 * T for s in series_list):
        shortest = min(s.shape[0] for s in series_list)
        raise ValueError(f"Need each series length > 2*T ({2 * T}), got shortest {shortest}")

    # DCA_init expects a single 2-D series.
    if do_dca_init and (patch_mode or multi_series):
        raise NotImplementedError(
            "do_dca_init requires a single whole-frame series; "
            "set do_dca_init = False for patch_mode or multi-video runs."
        )

    if patch_mode:
        print(
            f"Patch mode: {len(series_list)} series "
            f"(shapes vary in T; obs_dim={obs_dim})."
        )
    elif multi_series:
        lengths = [s.shape[0] for s in series_list]
        print(f"Multi-video: {len(series_list)} whole-frame series, lengths={lengths}, obs_dim={obs_dim}.")
        _log_memory_context(series_list[0], T_win=T, batch_size=batch_size, device=device)
    else:
        _log_memory_context(series_list[0], T_win=T, batch_size=batch_size, device=device)

    train_data = PastFutureDataset(series_list, window_size=T)
    init_weights = (
        DCA_init(series_list[0], T=T, d=ydim, rng_or_seed=args.seed) if do_dca_init else None
    )

    # Print train data information
    if multi_series or patch_mode:
        print(f"Train data: {len(series_list)} series, obs_dim={obs_dim}")
    else:
        print(f"Train data shape: {series_list[0].shape}")
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
        decoder_loss_warmup=decoder_loss_warmup,
        decoder_loss_warmup_epochs=decoder_loss_warmup_epochs,
    )
    ckpt_path = os.path.join(saved_root, f"sparse_cpic_checkpoint_sig{args.signature}_seed{args.seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_path": str(_resolve_config_arg(args.config)),
            "signature": args.signature,
            "seed": args.seed,
            "xdim": xdim,
            "ydim": ydim,
            "T": T,
            "encoder_params": enc,
            "sparse_params": p,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    et = getattr(model.encoder, "encoder_type", None)
    if et in ("conv_spatial", "conv_spatiotemporal", "conv_temporal"):
        model.visualize_kernels(kernel_save_suffix="video_sparse", signature=args.signature)

    encoded_list = []
    aligned_list = []
    r2_list = []
    for series in series_list:
        encoded_repr = model.encode(torch.from_numpy(series).float().to(device))
        # Export the sparse latent code (post-proximal soft-thresholding), consistent with SparseCPIC training.
        encoded_repr = torch.sign(encoded_repr) * torch.clamp(torch.abs(encoded_repr) - model.gamma, min=0.0)
        encoded_np = encoded_repr.detach().cpu().numpy()
        aligned = linear_alignment(encoded_np, series)
        r2 = compute_R2(aligned, series)
        encoded_list.append(encoded_np)
        aligned_list.append(aligned)
        r2_list.append(r2)

    r2 = float(np.mean(r2_list))
    if patch_mode:
        print(f"R2(sparse CPIC vs observations, linearly aligned): mean={r2:.6f} over {len(r2_list)} patch series")
    else:
        print(f"R2(sparse CPIC vs observations, linearly aligned): {r2}")

    snr_vals = np.array([0.0], dtype=np.float32)
    seed_str = f"_seed{args.seed}"
    payload = {"snr_vals": snr_vals, "losses": [loss], "R2_metrics": np.array([[np.nan, np.nan, r2]])}
    with open(os.path.join(saved_root, f"encoded_nxt{seed_str}.pkl"), "wb") as f:
        pickle.dump({"encoded_nxt": encoded_list, "snr_vals": snr_vals}, f)
    with open(os.path.join(saved_root, f"latent_R2{seed_str}.pkl"), "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(saved_root, f"inferred_trials{seed_str}.pkl"), "wb") as f:
        pickle.dump({"inferred_sparse_CPIC_trials": aligned_list, "snr_vals": snr_vals}, f)
