#!/usr/bin/env python3
"""
Batch particle dynamics experiment across encoder types and noise levels,
configured via INI files (similar workflow to synthetic/synthetic_experiment.py).

Example
-------
python run_particle_experiment.py --config particle_circle
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
DATA_GEN_PATH = ROOT / "experiments" / "particle_experiment"
for folder in (SRC_PATH, DATA_GEN_PATH):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cpic import CPIC
from cpic.utils.data import PastFutureDataset
from filter_visualization import (
    save_encoder_filter_artifacts,
    compute_trajectory_in_grid_coords,
    compute_avg_density_grid,
)
from probe_visualization import save_probe_viz_artifacts
from generate_particle_dynamics import generate_particle_process_timeseries  # type: ignore[reportMissingImports]


_CONFIG_DIR = Path(__file__).resolve().parent / "config"
# Named-config shortcuts for the current configs. A full path to any .ini also works
# (and is the normal way to run these).
config_file_dict = {
    "particle_beta_sweep": str(_CONFIG_DIR / "config_particle_beta_sweep.ini"),
    "particle_noise_sweep": str(_CONFIG_DIR / "config_particle_noise_sweep.ini"),
    "particle_noise_sweep_noblob": str(_CONFIG_DIR / "config_particle_noise_sweep_noblob.ini"),
    "particle_blob_sweep": str(_CONFIG_DIR / "config_particle_blob_sweep.ini"),
    "particle_nonclosed_noise_sweep": str(_CONFIG_DIR / "config_particle_nonclosed_noise_sweep.ini"),
    "particle_noblob_control": str(_CONFIG_DIR / "config_particle_noblob_control.ini"),
    "particle_probe_figs": str(_CONFIG_DIR / "config_particle_probe_figs.ini"),
}


class MyConf(ConfigParser):
    def optionxform(self, optionstr):
        """Preserve option key casing when reading INI files."""
        return optionstr


@dataclass(frozen=True)
class EncoderSpec:
    label: str
    encoder_params: dict
    predictive_space: str


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated integer list from config text."""
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty integer list")
    return [int(v) for v in vals]


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated float list from config text (empty -> [])."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def bool_or_default(cfg: ConfigParser, section: str, key: str, default: bool) -> bool:
    """Read a boolean config value or return a provided default."""
    return cfg.getboolean(section, key) if cfg.has_option(section, key) else default


def int_or_default(cfg: ConfigParser, section: str, key: str, default: int) -> int:
    """Read an integer config value or return a provided default."""
    return cfg.getint(section, key) if cfg.has_option(section, key) else default


def float_or_default(cfg: ConfigParser, section: str, key: str, default: float) -> float:
    """Read a float config value or return a provided default."""
    return cfg.getfloat(section, key) if cfg.has_option(section, key) else default


def get_summary_writer_cls():
    """Resolve an available SummaryWriter implementation, if installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter
        except Exception:
            return None


def load_encoder_specs(cfg: ConfigParser) -> list[EncoderSpec]:
    """Build enabled encoder/predictive-space combinations from config."""
    base = {
        "deterministic": False,
        "linear_encoder": False,
        "n_layers": cfg.getint("Model", "n_layers"),
        "activation": cfg.get("Model", "activation"),
    }

    specs: list[EncoderSpec] = []
    if bool_or_default(cfg, "Sweep", "include_mlp", True):
        mlp = {**base, "encoder_type": "mlp"}
        specs.append(EncoderSpec("MLP (L)", mlp, "latent"))
        specs.append(EncoderSpec("MLP (O)", mlp, "observation"))

    if bool_or_default(cfg, "Sweep", "include_conv_spatial", True):
        conv_sp = {
            **base,
            "encoder_type": "conv_spatial",
            "conv_kernel_size": cfg.getint("Model", "conv_spatial_kernel_size"),
            "conv_stride": cfg.getint("Model", "conv_spatial_stride"),
            "conv_padding": cfg.getint("Model", "conv_spatial_padding"),
        }
        specs.append(EncoderSpec("ConvSpatial (L)", conv_sp, "latent"))
        specs.append(EncoderSpec("ConvSpatial (O)", conv_sp, "observation"))

    if bool_or_default(cfg, "Sweep", "include_conv_particle", True):
        conv_pt = {
            **base,
            "encoder_type": "conv_particle",
            "conv_kernel_size": cfg.getint("Model", "conv_particle_kernel_size"),
            "conv_stride": cfg.getint("Model", "conv_particle_stride"),
            "conv_padding": cfg.getint("Model", "conv_particle_padding"),
            "conv_coord_kernel_size": cfg.getint("Model", "conv_particle_coord_kernel_size"),
        }
        specs.append(EncoderSpec("ConvParticle (L)", conv_pt, "latent"))
        specs.append(EncoderSpec("ConvParticle (O)", conv_pt, "observation"))

    if bool_or_default(cfg, "Sweep", "include_conv_physical", False):
        conv_phys = {
            **base,
            "encoder_type": "conv_physical",
            "grid_size": int_or_default(cfg, "Model", "conv_physical_grid_size", 20),
            "spatial_bounds": float_or_default(cfg, "Model", "conv_physical_spatial_bounds", 3.0),
            "conv_kernel_size": int_or_default(cfg, "Model", "conv_physical_kernel_size", 3),
            "conv_stride": int_or_default(cfg, "Model", "conv_physical_stride", 1),
            "conv_padding": int_or_default(cfg, "Model", "conv_physical_padding", 1),
        }
        specs.append(EncoderSpec("ConvPhysical (L)", conv_phys, "latent"))
        specs.append(EncoderSpec("ConvPhysical (O)", conv_phys, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_uniform_static", False):
        mask_uniform = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": False,
            "mask_init": "uniform",
            "mask_strategy": "uniform_static",
        }
        specs.append(EncoderSpec("MaskUniformStatic (L)", mask_uniform, "latent"))
        specs.append(EncoderSpec("MaskUniformStatic (O)", mask_uniform, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_uniform_learned", False):
        mask_uniform_learned = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": True,
            "mask_init": "uniform",
            "mask_strategy": "uniform_init_learned",
        }
        specs.append(EncoderSpec("MaskUniformLearned (L)", mask_uniform_learned, "latent"))
        specs.append(EncoderSpec("MaskUniformLearned (O)", mask_uniform_learned, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_random", False):
        mask_random = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": False,
            "mask_init": "random",
            "mask_strategy": "random",
        }
        specs.append(EncoderSpec("MaskRandom (L)", mask_random, "latent"))
        specs.append(EncoderSpec("MaskRandom (O)", mask_random, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_random_learned", False):
        mask_random_learned = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": True,
            "mask_init": "random",
            "mask_strategy": "random_init_learned",
        }
        specs.append(EncoderSpec("MaskRandomLearned (L)", mask_random_learned, "latent"))
        specs.append(EncoderSpec("MaskRandomLearned (O)", mask_random_learned, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_pi_static", False):
        mask_pi_static = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": False,
            "mask_init": "pi",
            "mask_strategy": "pi_static",
        }
        specs.append(EncoderSpec("MaskPIStatic (L)", mask_pi_static, "latent"))
        specs.append(EncoderSpec("MaskPIStatic (O)", mask_pi_static, "observation"))

    if bool_or_default(cfg, "Sweep", "include_mask_pi_init_learned", False):
        mask_pi_learned = {
            **base,
            "encoder_type": "mask_mlp",
            "mask_learnable": True,
            "mask_init": "pi",
            "mask_strategy": "pi_init_learned",
        }
        specs.append(EncoderSpec("MaskPIInitLearned (L)", mask_pi_learned, "latent"))
        specs.append(EncoderSpec("MaskPIInitLearned (O)", mask_pi_learned, "observation"))

    if not specs:
        raise ValueError("No encoders enabled in config [Sweep].")
    return specs


def pick_device(device_pref: str, use_gpu_flag: bool) -> str:
    """Resolve compute device from preference and hardware availability."""
    if device_pref.lower() != "auto":
        return device_pref
    if use_gpu_flag and torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_oom_error(exc: BaseException) -> bool:
    """Heuristically classify exceptions as out-of-memory failures."""
    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda out of memory",
        "mps backend out of memory",
        "not enough memory",
        "can't allocate memory",
        "cannot allocate memory",
        "allocator",
    )
    return any(marker in msg for marker in oom_markers)


def cleanup_memory(device: str) -> None:
    """Run garbage collection and clear CUDA cache when applicable."""
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_batch_candidates(
    *,
    default_batch_size: int,
    fallback_batch_sizes: list[int],
    oom_retry_enabled: bool,
    oom_max_retries: int,
) -> list[int]:
    """Create a deduplicated, ordered list of batch sizes to try."""
    if not oom_retry_enabled:
        return [default_batch_size]

    ordered = [default_batch_size]
    ordered.extend(fallback_batch_sizes)

    unique: list[int] = []
    for bs in ordered:
        if bs > 0 and bs not in unique:
            unique.append(bs)

    max_attempts = max(1, oom_max_retries + 1)
    return unique[:max_attempts]


def estimate_feature_pi_scores(
    data: np.ndarray,
    t_split: int,
    *,
    max_lag: int = 1,
    lag_agg: str = "mean",
) -> np.ndarray:
    """
    Estimate feature-wise PI proxy scores from lagged autocorrelation magnitudes.

    For each feature and lag k in [1, max_lag], compute |corr(x_t, x_{t+k})| on
    the train split and aggregate across lags (mean or max). Scores are then
    normalized to [0, 1] across features.
    """
    train = data[:t_split]
    max_lag = max(1, int(max_lag))
    if lag_agg not in {"mean", "max"}:
        raise ValueError(f"lag_agg must be one of {{'mean', 'max'}}, got {lag_agg!r}")
    if train.shape[0] < max_lag + 2:
        return np.ones(train.shape[1], dtype=np.float32)

    lag_scores = []
    n_features = train.shape[1]
    eps = 1e-8
    for lag in range(1, max_lag + 1):
        x_prev = train[:-lag]
        x_next = train[lag:]
        scores = np.zeros(n_features, dtype=np.float32)
        for feat in range(n_features):
            a = x_prev[:, feat]
            b = x_next[:, feat]
            if np.std(a) < eps or np.std(b) < eps:
                scores[feat] = 0.0
                continue
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            scores[feat] = abs(float(corr))
        lag_scores.append(scores)

    lag_scores_arr = np.stack(lag_scores, axis=0)  # (num_lags, n_features)
    if lag_agg == "max":
        scores = np.max(lag_scores_arr, axis=0)
    else:
        scores = np.mean(lag_scores_arr, axis=0)

    max_val = float(np.max(scores))
    if max_val <= eps:
        return np.ones_like(scores, dtype=np.float32)
    return (scores / max_val).astype(np.float32)


def with_mask_init_from_pi(encoder_params: dict, pi_scores: np.ndarray) -> dict:
    """Attach PI-derived mask initialization values for mask encoders."""
    params = dict(encoder_params)
    if params.get("encoder_type") != "mask_mlp":
        return params
    strategy = params.get("mask_strategy", "")
    if strategy in {"pi_static", "pi_init_learned"}:
        params["mask_init_values"] = pi_scores
    return params


def get_final_mask_stats(model: CPIC, threshold: float) -> tuple[float, str]:
    """Return active-mask fraction and active index signature."""
    if getattr(model.encoder, "encoder_type", None) != "mask_mlp":
        return float("nan"), ""
    mask_tensor = model.encoder.get_feature_mask().detach().cpu().numpy()
    active = np.where(mask_tensor >= threshold)[0]
    active_frac = float(active.size / mask_tensor.size)
    return active_frac, ",".join(map(str, active.tolist()))


def get_mask_probs(model: CPIC) -> np.ndarray | None:
    """Return the deterministic per-feature gate probabilities sigmoid(logits)."""
    if getattr(model.encoder, "encoder_type", None) != "mask_mlp":
        return None
    logits = getattr(getattr(model.encoder, "_mean", None), "mask_logits", None)
    if logits is None:
        return None
    return torch.sigmoid(logits).detach().cpu().numpy()


def get_blob_selection_stats(
    model: CPIC,
    particle_labels: np.ndarray,
    particle_order: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float]:
    """Gate-selection diagnostics for the FeatureMaskMLP mask.

    Returns ``(active_frac, blob_sel_rate, gate_selectivity, blob_base_rate)``.

    - ``blob_sel_rate`` is the PRECISION of the mask: of the features whose gate
      probability clears ``threshold``, the fraction that are blob. It is
      ``NaN`` when no feature clears the threshold — NOT 0.0. The old ``0.0``
      convention conflated three very different states ("gate never became
      confident", "gate at chance", "gate keeps only noise").
    - ``gate_selectivity = mean(prob|blob) - mean(prob|noise)`` is the honest,
      THRESHOLD-FREE and BASE-RATE-FREE signal: >0 means the gate genuinely
      prefers blob features, ~0 means chance, <0 means it prefers noise. Precision
      alone is base-rate-sensitive (it floats up toward 1.0 as the blob fraction
      rises even for a chance gate), so it cannot be read without the base rate.
      Notation: ``prob`` is the per-feature gate probability sigmoid(mask_logit);
      ``mean(prob|blob)`` is that probability averaged over the subset of features
      that are blob (the ``|`` reads "restricted to"), and ``mean(prob|noise)``
      the average over the noise features.
    - ``blob_base_rate`` is the blob feature fraction (the chance level for
      precision), so ``blob_sel_rate`` can be read against it.
    """
    probs = get_mask_probs(model)
    if probs is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Data features are interleaved (x, y) per particle, ordered by particle_order.
    feat_labels = np.repeat(particle_labels[particle_order], 2)
    blob_base_rate = float(np.mean(feat_labels == 1))

    active_idx = np.where(probs >= threshold)[0]
    active_frac = float(active_idx.size / probs.size)
    # NaN (not 0.0) on an empty active set: "nothing cleared the threshold" is not
    # "kept features are all noise".
    blob_sel_rate = (
        float(np.mean(feat_labels[active_idx] == 1)) if active_idx.size > 0 else float("nan")
    )

    blob_probs = probs[feat_labels == 1]
    noise_probs = probs[feat_labels == 0]
    blob_mean = float(np.mean(blob_probs)) if blob_probs.size else float("nan")
    noise_mean = float(np.mean(noise_probs)) if noise_probs.size else float("nan")
    gate_selectivity = blob_mean - noise_mean
    return active_frac, blob_sel_rate, gate_selectivity, blob_base_rate


def save_conv_filter_artifacts(
    model: CPIC,
    out_dir: Path,
    *,
    seed: int,
    num_noise: int,
    encoder_label: str,
    layer_idx: int,
    trajectory_grid: np.ndarray | None = None,
    avg_density: np.ndarray | None = None,
) -> dict[str, str]:
    """Save convolution filter weights and heatmap summaries (physical or panel style)."""
    empty = {
        "filter_weights_path": "",
        "filter_plot_path": "",
        "filter_trajectory_density_path": "",
        "filter_meta_path": "",
    }
    if not hasattr(model.encoder, "get_filters"):
        return empty
    try:
        weights, meta = model.encoder.get_filters(layer_idx=layer_idx)
    except Exception:
        return empty

    enc_type = getattr(model.encoder, "encoder_type", "")
    subdir = "conv_physical" if enc_type == "conv_physical" else "conv"
    target_dir = out_dir / subdir

    safe_label = encoder_label.replace(" ", "_").replace("(", "").replace(")", "")
    stem = f"{safe_label}_noise{num_noise}_seed{seed}_layer{layer_idx}"
    paths = save_encoder_filter_artifacts(
        weights,
        meta,
        out_dir=target_dir,
        stem=stem,
        encoder_type=enc_type,
        encoder_label=encoder_label,
        layer_idx=layer_idx,
        trajectory_grid=trajectory_grid if enc_type == "conv_physical" else None,
        avg_density=avg_density if enc_type == "conv_physical" else None,
    )
    return paths


def build_past_windows_and_gt(
    data: np.ndarray,
    gt_latent: np.ndarray,
    window_size: int,
    model: CPIC,
    device: str,
    t_min: int,
    t_max: int,
    encode_chunk_size: int = 256,
    return_end_times: bool = False,
) -> tuple[np.ndarray, ...]:
    """Encode sliding past windows and align corresponding latent targets.

    The encode is chunked so peak memory scales with ``encode_chunk_size``, not
    the number of windows. This matters for grid-expanding encoders like
    ConvPhysical, where a single forward over all (hundreds/thousands of)
    windows would OOM regardless of the training batch size.
    """
    end_times = np.arange(t_min, t_max)
    past_windows = np.stack([data[t - window_size : t] for t in end_times], axis=0)
    z_chunks = []
    with torch.no_grad():
        for start in range(0, past_windows.shape[0], encode_chunk_size):
            chunk = past_windows[start : start + encode_chunk_size]
            encoded = model.encode(torch.from_numpy(chunk).float().to(device))
            z_chunks.append(encoded[:, -1, :].cpu().numpy())
    z = np.concatenate(z_chunks, axis=0)
    gt = gt_latent[end_times - 1]
    if return_end_times:
        # Callers that probe additional per-time targets (velocity, noise centroid)
        # reuse these encoded latents by aligning their own target arrays with the
        # same `end_times - 1` index, avoiding a second (expensive) encode pass.
        return z, gt, end_times
    return z, gt


def heldout_probe_r2(z_train: np.ndarray, gt_train: np.ndarray, z_test: np.ndarray, gt_test: np.ndarray) -> dict[str, float]:
    """Fit a linear probe on train latents and report held-out R2 metrics."""
    probe = LinearRegression().fit(z_train, gt_train)
    pred_test = probe.predict(z_test)
    return {
        "r2_test_x": float(r2_score(gt_test[:, 0], pred_test[:, 0])),
        "r2_test_y": float(r2_score(gt_test[:, 1], pred_test[:, 1])),
        "r2_test_mean": float(r2_score(gt_test, pred_test, multioutput="uniform_average")),
    }


def build_velocity_gt(gt_latent: np.ndarray) -> np.ndarray:
    """First-order dynamics target: per-step blob-centroid velocity, shape (t_max, 2).

    Position-R2 rewards "observation" mode that models high-variance,
    temporally predictable noise; the orbit's *velocity* (heading + speed) is a
    coherent-blob-only signal — the random-walk noise particles average to no net
    velocity — so velocity-R2 measures whether the latent tracks the dynamics while
    suppressing noise. Derived by finite-differencing the same normalized centroid
    used for position-R2, so it inherits that normalization; row 0 is zero-padded and
    never indexed (windows end at t >= window_size >= 1).
    """
    v = np.zeros_like(gt_latent)
    v[1:] = np.diff(gt_latent, axis=0)
    return v.astype(np.float32)


def build_noise_centroid_gt(positions: np.ndarray, particle_labels: np.ndarray) -> np.ndarray:
    """Noise-cloud centroid target: mean position of the noise particles, shape (t_max, 2).

    Noise-suppression diagnostic. A latent that ignores the random-walk noise
    should decode this *poorly*; the headline contrast is high velocity-R2 with low
    noise-R2. The mean over many AR(1) walkers is near-zero and low-variance, so this
    R2 can sit near the floor for every encoder — that near-floor value is itself the
    evidence the latent is not tracking noise. Uses the noise sub-population mean so it
    stays 2-D and comparable across ``num_noise`` sweep points.
    """
    noise_mask = np.asarray(particle_labels) == 0
    if not noise_mask.any():
        return np.zeros((positions.shape[0], 2), dtype=np.float32)
    return positions[:, noise_mask, :].mean(axis=1).astype(np.float32)


def log_convergence_epoch(
    epoch: int,
    model: CPIC,
    *,
    writer,
    probe_eval_every: int,
    total_epochs: int,
    data: np.ndarray,
    gt_latent: np.ndarray,
    T: int,
    device: str,
    t_split: int,
    t_max: int,
    particle_labels: np.ndarray,
    particle_order: np.ndarray,
    mask_eval_threshold: float,
    encode_chunk_size: int,
) -> None:
    """Per-epoch convergence diagnostics for a single training run.

    Logs two families of curves that training loss alone does not reveal:
      * ``epoch/gate/*`` (mask encoders only) — mean gate prob on blob vs noise
        features, active fraction, and blob-selection precision, so one can see
        whether the gates are still moving at the final epoch (undertrained) or
        have plateaued (genuine behavior).
      * ``epoch/probe_r2/*`` (all encoders, every ``probe_eval_every`` epochs and
        on the final epoch) — held-out linear-probe R2, the reported predictivity
        metric, as a learning curve to confirm asymptotic performance.
    """
    if writer is None:
        return

    probs = get_mask_probs(model)
    if probs is not None:
        feat_labels = np.repeat(particle_labels[particle_order], 2)
        blob = probs[feat_labels == 1]
        noise = probs[feat_labels == 0]
        active = probs >= mask_eval_threshold
        blob_mean = float(blob.mean()) if blob.size else float("nan")
        noise_mean = float(noise.mean()) if noise.size else float("nan")
        writer.add_scalar("epoch/gate/mean_blob", blob_mean, epoch)
        writer.add_scalar("epoch/gate/mean_noise", noise_mean, epoch)
        writer.add_scalar("epoch/gate/active_frac", float(active.mean()), epoch)
        # Threshold-free, base-rate-free selectivity: >0 = genuine blob preference, ~0 = chance.
        writer.add_scalar("epoch/gate/selectivity", blob_mean - noise_mean, epoch)
        blob_sel = float(np.mean(feat_labels[active] == 1)) if active.sum() > 0 else float("nan")
        writer.add_scalar("epoch/gate/blob_sel_rate", blob_sel, epoch)

    if probe_eval_every > 0 and (epoch % probe_eval_every == 0 or epoch == total_epochs - 1):
        z_train, gt_train = build_past_windows_and_gt(
            data, gt_latent, T, model, device, t_min=T, t_max=t_split,
            encode_chunk_size=encode_chunk_size,
        )
        z_test, gt_test = build_past_windows_and_gt(
            data, gt_latent, T, model, device, t_min=t_split, t_max=t_max,
            encode_chunk_size=encode_chunk_size,
        )
        m = heldout_probe_r2(z_train, gt_train, z_test, gt_test)
        writer.add_scalar("epoch/probe_r2/test_mean", m["r2_test_mean"], epoch)
        writer.add_scalar("epoch/probe_r2/test_x", m["r2_test_x"], epoch)
        writer.add_scalar("epoch/probe_r2/test_y", m["r2_test_y"], epoch)


def run_condition(
    *,
    seed: int,
    num_noise: int,
    encoder_spec: EncoderSpec,
    t_max: int,
    train_ratio: float,
    T: int,
    num_blob: int,
    orbit_radius: float,
    trajectory: str,
    semi_major: float | None,
    semi_minor: float | None,
    omega: float,
    sigma_blob: float,
    noise_ar_coeff: float,
    centroid_ar_coeff: float,
    torus_ratio: float,
    torus_r2: float,
    spatial_bounds: float,
    hidden_dim: int,
    ydim: int,
    beta: float,
    critic: str,
    estimator_compress: str,
    compress_ungated: bool,
    consistency_weight: float,
    beta_warmup_epochs: int,
    beta_ramp_epochs: int,
    epochs: int,
    batch_size: int,
    oom_retry_enabled: bool,
    oom_batch_size_fallbacks: list[int],
    oom_max_retries: int,
    oom_skip_on_failure: bool,
    save_filter_viz: bool,
    filter_layer_idx: int,
    filter_out_dir: Path,
    save_probe_viz: bool,
    probe_out_dir: Path,
    enable_tensorboard: bool,
    tensorboard_root_dir: Path,
    mask_eval_threshold: float,
    pi_max_lag: int,
    pi_lag_agg: str,
    probe_eval_every: int,
    lr: float,
    mask_logit_lr: float | None,
    early_stop: int,
    device: str,
) -> dict:
    """Run one seed/noise/encoder condition with optional OOM batch fallback."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    (data, gt_latent, particle_order, particle_labels,
     standardization_mean, standardization_std, positions) = generate_particle_process_timeseries(
        t_max=t_max,
        num_blob=num_blob,
        num_noise=num_noise,
        trajectory=trajectory,
        orbit_radius=orbit_radius,
        semi_major=semi_major,
        semi_minor=semi_minor,
        omega=omega,
        sigma_blob=sigma_blob,
        noise_ar_coeff=noise_ar_coeff,
        centroid_ar_coeff=centroid_ar_coeff,
        torus_ratio=torus_ratio,
        torus_r2=torus_r2,
        spatial_bounds=spatial_bounds,
        seed=seed,
    )
    sigma_noise = float(spatial_bounds * np.sqrt(1 - noise_ar_coeff**2))

    t_split = int(train_ratio * t_max)
    if t_split <= 2 * T or (t_max - t_split) <= 2 * T:
        raise ValueError("train_ratio / t_max / T leaves too few windows.")

    train_data = PastFutureDataset([data[:t_split]], window_size=T)
    pi_scores = estimate_feature_pi_scores(data, t_split, max_lag=pi_max_lag, lag_agg=pi_lag_agg)
    encoder_params = with_mask_init_from_pi(dict(encoder_spec.encoder_params), pi_scores)

    batch_candidates = build_batch_candidates(
        default_batch_size=batch_size,
        fallback_batch_sizes=oom_batch_size_fallbacks,
        oom_retry_enabled=oom_retry_enabled,
        oom_max_retries=oom_max_retries,
    )

    model: CPIC | None = None
    metrics: dict[str, float] | None = None
    vel_metrics: dict[str, float] | None = None
    noise_metrics: dict[str, float] | None = None
    # Captured on the successful attempt so the probe-viz figure can be rendered AFTER
    # the OOM-retry loop (mirrors save_filter_viz), avoiding a plotting error aborting a
    # training run. Holds (z_train, gt_train, z_test, gt_test, vel_train, vel_test).
    probe_viz_data: tuple[np.ndarray, ...] | None = None
    used_batch_size: int | None = None
    status = "success"
    error_type = ""
    error_message = ""
    attempts = 0
    tensorboard_log_dir = ""
    summary_writer_cls = get_summary_writer_cls()
    if enable_tensorboard and summary_writer_cls is None:
        print("[WARN] TensorBoard logging requested but no SummaryWriter backend is installed.")

    for curr_batch_size in batch_candidates:
        attempts += 1
        writer = None
        try:
            model = CPIC(
                ydim=ydim,
                xdim=data.shape[1],
                T=T,
                encoder_params=encoder_params,
                mi_params={
                    "estimator_compress": estimator_compress,
                    "estimator_predictive": "infonce_lower",
                    "critic": critic,
                    "baseline": "constant",
                },
                hidden_dim=hidden_dim,
                beta=beta,
                device=device,
                predictive_space=encoder_spec.predictive_space,
                compress_ungated=compress_ungated,
                consistency_weight=consistency_weight,
                beta_warmup_epochs=beta_warmup_epochs,
                beta_ramp_epochs=beta_ramp_epochs,
            ).to(device)
            if enable_tensorboard and summary_writer_cls is not None:
                safe_label = encoder_spec.label.replace(" ", "_").replace("(", "").replace(")", "")
                tb_dir = tensorboard_root_dir / f"seed{seed}" / f"noise{num_noise}" / safe_label / f"attempt{attempts}"
                tb_dir.mkdir(parents=True, exist_ok=True)
                writer = summary_writer_cls(log_dir=str(tb_dir))
                tensorboard_log_dir = str(tb_dir)

            epoch_callback = None
            if writer is not None:
                epoch_callback = lambda epoch, m: log_convergence_epoch(
                    epoch, m,
                    writer=writer,
                    probe_eval_every=probe_eval_every,
                    total_epochs=epochs,
                    data=data,
                    gt_latent=gt_latent,
                    T=T,
                    device=device,
                    t_split=t_split,
                    t_max=t_max,
                    particle_labels=particle_labels,
                    particle_order=particle_order,
                    mask_eval_threshold=mask_eval_threshold,
                    encode_chunk_size=curr_batch_size,
                )

            model.fit(
                X=train_data,
                epochs=epochs,
                batch_size=curr_batch_size,
                lr=lr,
                early_stop=early_stop,
                writer=writer,
                epoch_callback=epoch_callback,
                mask_logit_lr=mask_logit_lr,
            )

            z_train, gt_train, et_train = build_past_windows_and_gt(
                data, gt_latent, T, model, device, t_min=T, t_max=t_split,
                encode_chunk_size=curr_batch_size, return_end_times=True,
            )
            z_test, gt_test, et_test = build_past_windows_and_gt(
                data, gt_latent, T, model, device, t_min=t_split, t_max=t_max,
                encode_chunk_size=curr_batch_size, return_end_times=True,
            )
            metrics = heldout_probe_r2(z_train, gt_train, z_test, gt_test)
            # Latent-dynamics metrics: reuse the encoded latents (z_train/z_test) and
            # align first-order-dynamics + noise targets with the same end_times index.
            vel_gt = build_velocity_gt(gt_latent)
            noise_gt = build_noise_centroid_gt(positions, particle_labels)
            vel_metrics = heldout_probe_r2(
                z_train, vel_gt[et_train - 1], z_test, vel_gt[et_test - 1]
            )
            noise_metrics = heldout_probe_r2(
                z_train, noise_gt[et_train - 1], z_test, noise_gt[et_test - 1]
            )
            if save_probe_viz:
                probe_viz_data = (
                    z_train, gt_train, z_test, gt_test,
                    vel_gt[et_train - 1], vel_gt[et_test - 1],
                )
            used_batch_size = curr_batch_size
            # A later attempt succeeding must clear any oom_failed status left
            # by earlier (larger-batch) attempts; status is otherwise sticky.
            status = "success"
            error_type = ""
            error_message = ""
            break
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            status = "oom_failed"
            error_type = "oom"
            error_message = str(exc)
            print(
                f"[OOM] seed={seed} num_noise={num_noise} encoder={encoder_spec.label} "
                f"batch_size={curr_batch_size} failed; trying smaller batch if available."
            )
        finally:
            if writer is not None:
                writer.close()
            if metrics is None and model is not None:
                del model
                model = None
            cleanup_memory(device)

    if metrics is None:
        if not oom_skip_on_failure:
            raise RuntimeError(
                f"OOM retries exhausted for seed={seed}, num_noise={num_noise}, "
                f"encoder={encoder_spec.label}. Last error: {error_message}"
            )
        metrics = {"r2_test_x": float("nan"), "r2_test_y": float("nan"), "r2_test_mean": float("nan")}
        used_batch_size = -1
    if vel_metrics is None:
        vel_metrics = {"r2_test_x": float("nan"), "r2_test_y": float("nan"), "r2_test_mean": float("nan")}
    if noise_metrics is None:
        noise_metrics = {"r2_test_x": float("nan"), "r2_test_y": float("nan"), "r2_test_mean": float("nan")}

    mask_active_frac = float("nan")
    mask_active_indices = ""
    blob_sel_rate = float("nan")
    gate_selectivity = float("nan")
    blob_base_rate = float("nan")
    if model is not None and status == "success":
        mask_active_frac, mask_active_indices = get_final_mask_stats(model, mask_eval_threshold)
        _, blob_sel_rate, gate_selectivity, blob_base_rate = get_blob_selection_stats(
            model, particle_labels, particle_order, mask_eval_threshold
        )

    filter_weights_path = ""
    filter_plot_path = ""
    filter_trajectory_density_path = ""
    filter_meta_path = ""
    if model is not None and status == "success" and save_filter_viz:
        trajectory_grid = None
        avg_density = None
        if getattr(model.encoder, "encoder_type", "") == "conv_physical":
            try:
                _, phys_meta = model.encoder.get_filters(layer_idx=filter_layer_idx)
                grid_size = phys_meta["grid_size"]
                enc_spatial_bounds = phys_meta["spatial_bounds"]
                trajectory_grid = compute_trajectory_in_grid_coords(
                    positions, particle_labels, particle_order,
                    standardization_mean, standardization_std,
                    grid_size=grid_size,
                    spatial_bounds=enc_spatial_bounds,
                )
                avg_density = compute_avg_density_grid(
                    data[:t_split], grid_size=grid_size, spatial_bounds=enc_spatial_bounds,
                )
            except Exception as exc:
                print(f"[WARN] Could not compute trajectory/density for conv_physical: {exc}")

        filter_paths = save_conv_filter_artifacts(
            model,
            filter_out_dir,
            seed=seed,
            num_noise=num_noise,
            encoder_label=encoder_spec.label,
            layer_idx=filter_layer_idx,
            trajectory_grid=trajectory_grid,
            avg_density=avg_density,
        )
        filter_weights_path = filter_paths.get("filter_weights_path", "")
        filter_plot_path = filter_paths.get("filter_plot_path", "")
        filter_trajectory_density_path = filter_paths.get("filter_trajectory_density_path", "")
        filter_meta_path = filter_paths.get("filter_meta_path", "")

    # Probe-result spatial viz: position-trajectory overlay + true-vs-predicted velocity
    # field, rendered from the arrays captured on the successful attempt. Wrapped so a
    # plotting failure only drops the figure, never the run's metrics.
    probe_plot_path = ""
    probe_npz_path = ""
    if status == "success" and save_probe_viz and probe_viz_data is not None:
        try:
            z_tr, gt_tr, z_te, gt_te, vel_tr, vel_te = probe_viz_data
            pos_probe = LinearRegression().fit(z_tr, gt_tr)
            vel_probe = LinearRegression().fit(z_tr, vel_tr)
            pred_pos = pos_probe.predict(z_te)
            vel_pred = vel_probe.predict(z_te)
            probe_paths = save_probe_viz_artifacts(
                probe_out_dir,
                seed=seed,
                num_noise=num_noise,
                num_blob=num_blob,
                encoder_label=encoder_spec.label,
                gt_pos=gt_te,
                pred_pos=pred_pos,
                vel_true=vel_te,
                vel_pred=vel_pred,
                r2_pos=float(metrics.get("r2_test_mean", float("nan"))),
                r2_vel=float(vel_metrics.get("r2_test_mean", float("nan"))),
            )
            probe_plot_path = probe_paths.get("probe_plot_path", "")
            probe_npz_path = probe_paths.get("probe_npz_path", "")
        except Exception as exc:
            print(f"[WARN] Could not render probe viz: {exc}")

    # Achieved rate at the final epoch — the honest converged I_compress, an
    # estimator-native compression number logged alongside the R2 families.
    final_I_compress = (
        float(getattr(model, "final_mean_I_compress", float("nan")))
        if model is not None
        else float("nan")
    )
    # Estimator-native latent-quality metric: converged predictive information,
    # reported alongside the linear probe R2 so latent quality is not judged by linear
    # position-decodability alone.
    final_I_predictive = (
        float(getattr(model, "final_mean_I_predictive", float("nan")))
        if model is not None
        else float("nan")
    )
    if model is not None:
        del model
    cleanup_memory(device)

    return {
        "seed": seed,
        "num_noise": num_noise,
        "num_blob": num_blob,
        "ydim": ydim,
        "beta": beta,
        "final_I_compress": final_I_compress,
        "final_I_predictive": final_I_predictive,
        "trajectory": trajectory,
        "sigma_noise": sigma_noise,
        "noise_ar_coeff": noise_ar_coeff,
        "centroid_ar_coeff": centroid_ar_coeff,
        "torus_ratio": torus_ratio,
        "torus_r2": torus_r2,
        "encoder_label": encoder_spec.label,
        "predictive_space": encoder_spec.predictive_space,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "attempts": attempts,
        "used_batch_size": used_batch_size,
        "mask_active_frac": mask_active_frac,
        "mask_active_indices": mask_active_indices,
        "blob_sel_rate": blob_sel_rate,
        "gate_selectivity": gate_selectivity,
        "blob_base_rate": blob_base_rate,
        "pi_mean_score": float(np.mean(pi_scores)),
        "pi_max_lag": int(pi_max_lag),
        "pi_lag_agg": pi_lag_agg,
        "filter_weights_path": filter_weights_path,
        "filter_plot_path": filter_plot_path,
        "filter_trajectory_density_path": filter_trajectory_density_path,
        "filter_meta_path": filter_meta_path,
        "probe_plot_path": probe_plot_path,
        "probe_npz_path": probe_npz_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        # Latent-dynamics metrics: velocity (first-order dynamics; blob-only signal)
        # and noise-cloud-centroid decodability (should be near floor for a noise-suppressing
        # latent). Headline contrast = high velocity-R2 with low noise-R2.
        "r2_velocity_x": vel_metrics["r2_test_x"],
        "r2_velocity_y": vel_metrics["r2_test_y"],
        "r2_velocity_mean": vel_metrics["r2_test_mean"],
        "r2_noise_x": noise_metrics["r2_test_x"],
        "r2_noise_y": noise_metrics["r2_test_y"],
        "r2_noise_mean": noise_metrics["r2_test_mean"],
        **metrics,
    }


def save_csv(rows: list[dict], path: Path) -> None:
    """Write experiment result rows to CSV."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_r2_vs_num_noise(rows: list[dict], out_png: Path) -> None:
    """Plot held-out mean R2 versus number of noise particles by encoder."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df_success = df[df["status"] == "success"].copy()
    agg = (
        df_success.groupby(["encoder_label", "num_noise"], as_index=False)["r2_test_mean"]
        .agg(mean="mean", std="std")
        .sort_values(["encoder_label", "num_noise"])
    )

    # Shared styling from the canonical plotter (single source of truth):
    # color per encoder family, dotted/hollow latent vs solid/filled observation.
    from matplotlib.lines import Line2D

    from plot_particle_results import (
        _base_family_from_label,
        _color_for_label,
        _line_style_for_label,
        _marker_facecolor_for_label,
    )

    # tab10 fallback for any family not in the canonical color map.
    _fallback: dict[str, tuple] = {}
    cmap = plt.get_cmap("tab10")

    def resolve_color(label: str):
        c = _color_for_label(label)
        if c is not None:
            return c
        fam = _base_family_from_label(label)
        if fam not in _fallback:
            _fallback[fam] = cmap(len(_fallback) % 10)
        return _fallback[fam]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]

    for label, sub in agg.groupby("encoder_label"):
        x = sub["num_noise"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["std"].fillna(0.0).to_numpy(dtype=float)
        c = resolve_color(label)
        ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle=_line_style_for_label(label),
                    color=c, markerfacecolor=_marker_facecolor_for_label(label, c),
                    markeredgecolor=c, linewidth=2, markersize=6,
                    capsize=3, elinewidth=1.2, label=label)

    # Explicit proxies so the legend reflects line style / marker fill.
    legend_handles = []
    for lbl in sorted(agg["encoder_label"].unique()):
        c = resolve_color(lbl)
        legend_handles.append(
            Line2D([0], [0], color=c, linestyle=_line_style_for_label(lbl), marker="o",
                   markersize=6, linewidth=2, markerfacecolor=_marker_facecolor_for_label(lbl, c),
                   markeredgecolor=c, label=lbl)
        )

    ax.set_xlabel("Number of noise particles", fontsize=14)
    ax.set_ylabel(r"Test linear-probe $R^2$ (mean over x,y)", fontsize=14)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title(r"$R^2$ vs noise particles", fontsize=14)

    ax2 = axes[1]
    if "mask_active_frac" in df_success.columns:
        mask_df = df_success[df_success["encoder_label"].str.contains("Mask", na=False)]
        if not mask_df.empty:
            mask_agg = (
                mask_df.groupby(["encoder_label", "num_noise"], as_index=False)["mask_active_frac"]
                .agg(mean="mean", std="std")
                .sort_values(["encoder_label", "num_noise"])
            )
            for label, sub in mask_agg.groupby("encoder_label"):
                c = resolve_color(label)
                ax2.errorbar(
                    sub["num_noise"].to_numpy(dtype=float),
                    sub["mean"].to_numpy(dtype=float),
                    yerr=sub["std"].fillna(0.0).to_numpy(dtype=float),
                    fmt="o",
                    linestyle=_line_style_for_label(label),
                    color=c,
                    markerfacecolor=_marker_facecolor_for_label(label, c),
                    markeredgecolor=c,
                    linewidth=2,
                    markersize=6,
                    capsize=3,
                    label=label,
                )
            ax2.set_ylabel("Active mask fraction", fontsize=14)
            ax2.set_title("Mask sparsity vs noise", fontsize=14)
        else:
            ax2.text(0.5, 0.5, "No mask runs available", ha="center", va="center")
    ax2.set_xlabel("Number of noise particles", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Metric families plotted against a sweep axis: (column, panel title).
_R2_FAMILY_PANELS = [
    ("r2_test_mean", "Position R²"),
    ("r2_velocity_mean", "Velocity R²"),
    ("r2_noise_mean", "Noise-cloud R²"),
]
# Estimator-native latent-quality panels (nats): converged predictive information and
# achieved compression rate, plotted against the same sweep axes as the R² families.
_MI_FAMILY_PANELS = [
    ("final_I_predictive", "Predictive info  I_pred (nats)"),
    ("final_I_compress", "Rate  I_compress (nats)"),
]


def plot_r2_families_vs_sweep(
    rows: list[dict],
    x_col: str,
    x_label: str,
    out_png: Path,
    panels: list[tuple[str, str]] = _R2_FAMILY_PANELS,
    ylabel: str = r"Test linear-probe $R^2$",
    logx: bool = False,
) -> None:
    """Plot a row of metric-family panels versus a sweep axis by encoder.

    Generalizes ``plot_r2_vs_num_noise`` to any swept quantity (``num_noise``,
    ``num_blob``, ...) and any panel set (the three latent-quality R² families by
    default, or the MI/PI families via ``panels=_MI_FAMILY_PANELS``), so the same
    house-styled panel row can be produced for each sweep dimension and metric group.
    Each point is the seed mean with a +/-1 std error bar; styling (color per encoder,
    dotted/hollow latent vs solid/filled observation) matches the canonical plotter.
    """
    import pandas as pd
    from matplotlib.lines import Line2D

    from plot_particle_results import (
        _base_family_from_label,
        _color_for_label,
        _line_style_for_label,
        _marker_facecolor_for_label,
    )

    df = pd.DataFrame(rows)
    df = df[df["status"] == "success"].copy()
    panels = [(c, t) for c, t in panels if c in df.columns]
    if df.empty or not panels or x_col not in df.columns:
        return

    _fallback: dict[str, tuple] = {}
    cmap = plt.get_cmap("tab10")

    def resolve_color(label: str):
        c = _color_for_label(label)
        if c is not None:
            return c
        fam = _base_family_from_label(label)
        if fam not in _fallback:
            _fallback[fam] = cmap(len(_fallback) % 10)
        return _fallback[fam]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]
    for ax, (col, title) in zip(axes, panels):
        agg = (
            df.groupby(["encoder_label", x_col], as_index=False)[col]
            .agg(mean="mean", std="std")
            .sort_values(["encoder_label", x_col])
        )
        for label, sub in agg.groupby("encoder_label"):
            c = resolve_color(label)
            ax.errorbar(
                sub[x_col].to_numpy(dtype=float), sub["mean"].to_numpy(dtype=float),
                yerr=sub["std"].fillna(0.0).to_numpy(dtype=float),
                fmt="o", linestyle=_line_style_for_label(label), color=c,
                markerfacecolor=_marker_facecolor_for_label(label, c), markeredgecolor=c,
                linewidth=2, markersize=6, capsize=3, elinewidth=1.2, label=label,
            )
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        if logx:
            ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=11)

    legend_handles = [
        Line2D([0], [0], color=resolve_color(lbl), linestyle=_line_style_for_label(lbl),
               marker="o", markersize=6, linewidth=2,
               markerfacecolor=_marker_facecolor_for_label(lbl, resolve_color(lbl)),
               markeredgecolor=resolve_color(lbl), label=lbl)
        for lbl in sorted(df["encoder_label"].unique())
    ]
    axes[0].legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch particle dynamics encoder sweep using config files.")
    parser.add_argument("--config", type=str, default="particle_circle")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--signature", type=str, default=None)
    args = parser.parse_args()

    if args.config in config_file_dict:
        config_file = config_file_dict[args.config]
    elif Path(args.config).is_file():
        config_file = args.config
    else:
        raise ValueError(
            f"Unknown config {args.config!r}. Pass a registered name "
            f"({list(config_file_dict.keys())}) or a path to an .ini file."
        )

    cfg = MyConf()
    cfg.read(config_file)

    output_dir = Path(cfg.get("Paths", "output_dir"))
    if args.signature is not None:
        output_dir = output_dir / str(args.signature)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(cfg.get("Sweep", "seeds"))
    num_noise_values = parse_int_list(cfg.get("Sweep", "num_noise_values"))

    specs = load_encoder_specs(cfg)

    t_max = cfg.getint("Data", "t_max")
    train_ratio = cfg.getfloat("Data", "train_ratio")
    T = cfg.getint("Data", "T")
    num_blob = cfg.getint("Data", "num_blob")
    # Optional num_blob sweep (mirrors num_noise_values). When [Sweep] num_blob_values
    # is absent, fall back to the single [Data] num_blob so existing configs are
    # unchanged. Sweeping it traces how the coherent signal strength (particles on the
    # orbit) drives position/velocity R2 — the num_blob=0 end is the no-blob control.
    num_blob_values = parse_int_list(
        cfg.get("Sweep", "num_blob_values", fallback=str(num_blob))
    )
    orbit_radius = cfg.getfloat("Data", "orbit_radius")
    trajectory = cfg.get("Data", "trajectory", fallback="circle").strip()
    _semi_major_str = cfg.get("Data", "semi_major", fallback="").strip()
    _semi_minor_str = cfg.get("Data", "semi_minor", fallback="").strip()
    semi_major = float(_semi_major_str) if _semi_major_str else None
    semi_minor = float(_semi_minor_str) if _semi_minor_str else None
    omega = cfg.getfloat("Data", "omega")
    sigma_blob = cfg.getfloat("Data", "sigma_blob")
    noise_ar_coeff = float_or_default(cfg, "Data", "noise_ar_coeff", 0.8)
    centroid_ar_coeff = float_or_default(cfg, "Data", "centroid_ar_coeff", 0.95)
    # Incommensurate-torus params (trajectory=torus). Default ratio = golden ratio
    # (maximally non-closing); torus_r2=0 reduces the torus to the plain circle.
    torus_ratio = float_or_default(cfg, "Data", "torus_ratio", (1.0 + 5.0 ** 0.5) / 2.0)
    torus_r2 = float_or_default(cfg, "Data", "torus_r2", 0.0)
    spatial_bounds = cfg.getfloat("Data", "spatial_bounds")

    hidden_dim = cfg.getint("Model", "hidden_dim")
    # Latent dimensionality. Default 2 (the historical hardcoded value). Increasing it
    # gives the representation more volume; used by the ydim sweep to study how spatial
    # (ydim) and temporal (beta) compression interact.
    ydim = int_or_default(cfg, "Model", "ydim", 2)
    beta = cfg.getfloat("Model", "beta")
    critic = cfg.get("Model", "critic", fallback="concat")
    # Compression (rate) estimator. This term is MINIMIZED, so it needs an UPPER bound
    # on I(X;Z): "infonce_upper" (default, matches CPIC.py) or "vub" (analytic
    # KL(encoder || N(0,I))). A lower bound like "infonce_lower" is degenerate to
    # minimize (variances collapse, bound -> -inf).
    estimator_compress = cfg.get("Model", "estimator_compress", fallback="infonce_upper")
    # Ungated compression rate for the FeatureMaskMLP gate: estimate I(X;Z) from a gate-free
    # encoding + a consistency loss pulling the gated encoding toward it. Absent = off (default);
    # every existing config stays byte-identical.
    compress_ungated = bool_or_default(cfg, "Model", "compress_ungated", False)
    consistency_weight = float_or_default(cfg, "Model", "consistency_weight", 0.0)
    # Rate warm-up: beta=0 for beta_warmup_epochs, then linear ramp to target over beta_ramp_epochs.
    beta_warmup_epochs = int(float_or_default(cfg, "Model", "beta_warmup_epochs", 0))
    beta_ramp_epochs = int(float_or_default(cfg, "Model", "beta_ramp_epochs", 0))

    # Optional beta sweep: if [Sweep] betas is set, loop over those values as the
    # outermost dimension (the "interpretability knob" experiment). Otherwise use
    # the single [Model] beta. Each row records its own beta.
    betas = parse_float_list(cfg.get("Sweep", "betas", fallback=""))
    if not betas:
        betas = [beta]

    epochs = cfg.getint("Training", "epochs")
    batch_size = cfg.getint("Training", "batch_size")
    lr = cfg.getfloat("Training", "lr")
    # Optional dedicated LR for FeatureMaskMLP gate logits (they converge slowly at
    # the base LR). Unset -> None -> single param group (default behavior).
    mask_logit_lr = cfg.getfloat("Training", "mask_logit_lr") if cfg.has_option("Training", "mask_logit_lr") else None
    early_stop = cfg.getint("Training", "early_stop")
    # Conv encoders are the runtime bottleneck and converge in far fewer epochs
    # than the MaskMLP discovery recipe needs. Allow a separate (smaller) budget
    # for conv_* specs; fall back to the global value when not set.
    conv_epochs = int_or_default(cfg, "Training", "conv_epochs", epochs)
    conv_early_stop = int_or_default(cfg, "Training", "conv_early_stop", early_stop)
    oom_retry_enabled = bool_or_default(cfg, "OOM", "oom_retry_enabled", True)
    oom_batch_size_fallbacks = parse_int_list(
        cfg.get("OOM", "oom_batch_size_fallbacks", fallback=str(batch_size))
    )
    oom_max_retries = int_or_default(cfg, "OOM", "oom_max_retries", len(oom_batch_size_fallbacks))
    oom_skip_on_failure = bool_or_default(cfg, "OOM", "oom_skip_on_failure", True)
    save_filter_viz = bool_or_default(cfg, "Analysis", "save_filter_viz", False)
    filter_layer_idx = int_or_default(cfg, "Analysis", "filter_layer_idx", 0)
    filter_out_subdir = cfg.get("Analysis", "filter_out_dir", fallback="filters")
    # Opt-in probe-result spatial figures (position overlay + velocity field). Off by
    # default: it refits two probes and writes a PNG+npz per successful condition.
    save_probe_viz = bool_or_default(cfg, "Analysis", "save_probe_viz", False)
    probe_out_subdir = cfg.get("Analysis", "probe_out_dir", fallback="probe_viz")
    enable_tensorboard = bool_or_default(cfg, "Analysis", "enable_tensorboard", False)
    tensorboard_subdir = cfg.get("Analysis", "tensorboard_dir", fallback="tensorboard")
    mask_eval_threshold = float_or_default(cfg, "Analysis", "mask_eval_threshold", 0.5)
    pi_max_lag = int_or_default(cfg, "Analysis", "pi_max_lag", 1)
    pi_lag_agg = cfg.get("Analysis", "pi_lag_agg", fallback="mean").strip().lower()
    # Per-epoch held-out probe-R2 curve cadence (0 disables; only active with TensorBoard).
    probe_eval_every = int_or_default(cfg, "Analysis", "probe_eval_every", 0)

    use_gpu = cfg.getboolean("Compute", "use_gpu")
    device_pref = args.device or cfg.get("Compute", "device", fallback="auto")
    device = pick_device(device_pref, use_gpu)
    print(f"Device: {device}")

    csv_path = output_dir / cfg.get("Paths", "csv_name")
    plot_path = output_dir / cfg.get("Paths", "plot_name")
    filter_out_dir = output_dir / filter_out_subdir
    probe_out_dir = output_dir / probe_out_subdir
    tensorboard_root_dir = output_dir / tensorboard_subdir
    if save_filter_viz:
        filter_out_dir.mkdir(parents=True, exist_ok=True)
    if save_probe_viz:
        probe_out_dir.mkdir(parents=True, exist_ok=True)
    if enable_tensorboard:
        tensorboard_root_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for beta_val in betas:
        for seed in seeds:
            for num_noise in num_noise_values:
              for num_blob_val in num_blob_values:
                for spec in specs:
                    is_conv = str(spec.encoder_params.get("encoder_type", "")).startswith("conv")
                    spec_epochs = conv_epochs if is_conv else epochs
                    spec_early_stop = conv_early_stop if is_conv else early_stop
                    row = run_condition(
                        seed=seed,
                        num_noise=num_noise,
                        encoder_spec=spec,
                        t_max=t_max,
                        train_ratio=train_ratio,
                        T=T,
                        num_blob=num_blob_val,
                        orbit_radius=orbit_radius,
                        trajectory=trajectory,
                        semi_major=semi_major,
                        semi_minor=semi_minor,
                        omega=omega,
                        sigma_blob=sigma_blob,
                        noise_ar_coeff=noise_ar_coeff,
                        centroid_ar_coeff=centroid_ar_coeff,
                        torus_ratio=torus_ratio,
                        torus_r2=torus_r2,
                        spatial_bounds=spatial_bounds,
                        hidden_dim=hidden_dim,
                        ydim=ydim,
                        beta=beta_val,
                        critic=critic,
                        estimator_compress=estimator_compress,
                        compress_ungated=compress_ungated,
                        consistency_weight=consistency_weight,
                        beta_warmup_epochs=beta_warmup_epochs,
                        beta_ramp_epochs=beta_ramp_epochs,
                        epochs=spec_epochs,
                        batch_size=batch_size,
                        oom_retry_enabled=oom_retry_enabled,
                        oom_batch_size_fallbacks=oom_batch_size_fallbacks,
                        oom_max_retries=oom_max_retries,
                        oom_skip_on_failure=oom_skip_on_failure,
                        save_filter_viz=save_filter_viz,
                        filter_layer_idx=filter_layer_idx,
                        filter_out_dir=filter_out_dir,
                        save_probe_viz=save_probe_viz,
                        probe_out_dir=probe_out_dir,
                        enable_tensorboard=enable_tensorboard,
                        tensorboard_root_dir=tensorboard_root_dir,
                        mask_eval_threshold=mask_eval_threshold,
                        pi_max_lag=pi_max_lag,
                        pi_lag_agg=pi_lag_agg,
                        probe_eval_every=probe_eval_every,
                        lr=lr,
                        mask_logit_lr=mask_logit_lr,
                        early_stop=spec_early_stop,
                        device=device,
                    )
                    rows.append(row)
                    print(
                        f"beta={beta_val:<8g} | seed={seed:>2} | num_noise={num_noise:>3} "
                        f"| num_blob={num_blob_val:>3} | {spec.label:<18} | status={row['status']:<10} "
                        f"| batch={row['used_batch_size']:>4} | epochs={spec_epochs:>3} "
                        f"-> R2_pos={row['r2_test_mean']:.3f} "
                        f"| R2_vel={row['r2_velocity_mean']:.3f} "
                        f"| R2_noise={row['r2_noise_mean']:.3f} "
                        f"| blob_sel={row['blob_sel_rate']:.2f}"
                    )
                    # Flush after every condition so a wall-clock kill never wipes
                    # a multi-hour run (results are otherwise only saved at the end).
                    save_csv(rows, csv_path)

    save_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    # The R2-vs-num_noise sweep plot is only meaningful across multiple noise
    # levels; with a single noise value (e.g. convergence configs) every encoder
    # collapses to one dot at the same x, so skip it. The per-epoch learning
    # curves (plot_convergence_curves.py) are the right diagnostic there.
    if len(num_noise_values) > 1:
        plot_r2_vs_num_noise(rows, plot_path)
        print(f"Wrote {plot_path}")
        families_noise_path = plot_path.with_name("R2_families_vs_num_noise.png")
        plot_r2_families_vs_sweep(rows, "num_noise", "Number of noise particles", families_noise_path)
        print(f"Wrote {families_noise_path}")
        mi_noise_path = plot_path.with_name("MI_families_vs_num_noise.png")
        plot_r2_families_vs_sweep(rows, "num_noise", "Number of noise particles", mi_noise_path,
                                  panels=_MI_FAMILY_PANELS, ylabel="Information (nats)")
        print(f"Wrote {mi_noise_path}")
    else:
        print(
            f"Skipping {plot_path.name}: single noise value "
            f"({num_noise_values[0]}) makes the R2-vs-num_noise sweep plot degenerate."
        )

    # Same panel row against the blob-count axis when num_blob is swept (num_blob=0 is
    # the no-blob control end): position/velocity R² should rise with coherent signal.
    if len(num_blob_values) > 1:
        families_blob_path = plot_path.with_name("R2_families_vs_num_blob.png")
        plot_r2_families_vs_sweep(rows, "num_blob", "Number of blob particles", families_blob_path)
        print(f"Wrote {families_blob_path}")
        mi_blob_path = plot_path.with_name("MI_families_vs_num_blob.png")
        plot_r2_families_vs_sweep(rows, "num_blob", "Number of blob particles", mi_blob_path,
                                  panels=_MI_FAMILY_PANELS, ylabel="Information (nats)")
        print(f"Wrote {mi_blob_path}")

    # Beta axis (log-scaled, geometric ladder): the soft-penalty compression/prediction
    # tradeoff sweep. Same metric-family + MI/PI panel rows against beta.
    if len(betas) > 1:
        families_beta_path = plot_path.with_name("R2_families_vs_beta.png")
        plot_r2_families_vs_sweep(rows, "beta", r"Compression weight $\beta$",
                                  families_beta_path, logx=True)
        print(f"Wrote {families_beta_path}")
        mi_beta_path = plot_path.with_name("MI_families_vs_beta.png")
        plot_r2_families_vs_sweep(rows, "beta", r"Compression weight $\beta$", mi_beta_path,
                                  panels=_MI_FAMILY_PANELS, ylabel="Information (nats)", logx=True)
        print(f"Wrote {mi_beta_path}")
    