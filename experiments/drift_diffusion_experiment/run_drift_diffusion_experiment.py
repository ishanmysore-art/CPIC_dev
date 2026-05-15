#!/usr/bin/env python3
"""
Batch drift-diffusion particle experiment across encoder types and noise levels,
configured via INI files (similar workflow to synthetic/synthetic_experiment.py).

Example
-------
python run_drift_diffusion_experiment.py --config drift_diffusion_cpic_conv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
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
DATA_GEN_PATH = ROOT / "experiments" / "drift_diffusion_experiment"
for folder in (SRC_PATH, DATA_GEN_PATH):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cpic import CPIC
from cpic.utils.data import PastFutureDataset
from filter_visualization import (
    filter_weights_to_panel_magnitudes,
    plot_filter_heatmap_panels,
    plot_physical_filter_heatmaps,
    plot_orbit_density_map,
    compute_orbit_in_grid_coords,
    compute_tangential_arrows,
    compute_avg_density_grid,
)
from generate_drift_diffusion import generate_drift_diffusion_process_timeseries # type: ignore[reportMissingImports]


config_file_dict = {
    "drift_diffusion_cpic_conv": str(Path(__file__).resolve().parent / "config" / "config_drift_diffusion_cpic_conv.ini"),
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


def save_conv_filter_artifacts(
    model: CPIC,
    out_dir: Path,
    *,
    seed: int,
    num_noise: int,
    encoder_label: str,
    layer_idx: int,
    orbit_grid: np.ndarray | None = None,
    avg_density: np.ndarray | None = None,
) -> tuple[str, str]:
    """Save convolution filter weights and a compact heatmap summary."""
    if not hasattr(model.encoder, "get_filters"):
        return "", ""
    try:
        weights, meta = model.encoder.get_filters(layer_idx=layer_idx)
    except Exception:
        return "", ""

    safe_label = encoder_label.replace(" ", "_").replace("(", "").replace(")", "")
    stem = f"{safe_label}_noise{num_noise}_seed{seed}_layer{layer_idx}"
    npy_path = out_dir / f"{stem}.npy"
    png_path = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}.json"

    np.save(npy_path, weights)

    is_physical = getattr(model.encoder, "encoder_type", "") == "conv_physical"
    if is_physical:
        fig, _ = plot_physical_filter_heatmaps(
            weights,
            n_cols=8,
            suptitle=f"{encoder_label} layer {layer_idx}",
        )
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if orbit_grid is not None and avg_density is not None:
            arrows = compute_tangential_arrows(orbit_grid)
            density_png = out_dir / f"{stem}_orbit_density.png"
            fig2, _ = plot_orbit_density_map(
                avg_density,
                orbit_grid,
                arrows=arrows,
                spatial_bounds=meta.get("spatial_bounds", 3.0),
                suptitle=f"{encoder_label} — orbit + density (noise={num_noise}, seed={seed})",
            )
            fig2.savefig(density_png, dpi=150, bbox_inches="tight")
            plt.close(fig2)
    else:
        w_mag = filter_weights_to_panel_magnitudes(weights)
        fig, _ = plot_filter_heatmap_panels(
            w_mag,
            gap_rows=1,
            filters_per_panel=8,
            suptitle=f"{encoder_label} layer {layer_idx}",
        )
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    with meta_path.open("w") as f:
        json.dump({"shape": list(weights.shape), "meta": meta}, f)
    return str(npy_path), str(png_path)


def build_past_windows_and_gt(
    data: np.ndarray,
    gt_latent: np.ndarray,
    window_size: int,
    model: CPIC,
    device: str,
    t_min: int,
    t_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode sliding past windows and align corresponding latent targets."""
    end_times = np.arange(t_min, t_max)
    past_windows = np.stack([data[t - window_size : t] for t in end_times], axis=0)
    with torch.no_grad():
        encoded = model.encode(torch.from_numpy(past_windows).float().to(device))
        z = encoded[:, -1, :].cpu().numpy()
    gt = gt_latent[end_times - 1]
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


def run_condition(
    *,
    seed: int,
    num_noise: int,
    sigma_noise: float,
    encoder_spec: EncoderSpec,
    t_max: int,
    train_ratio: float,
    T: int,
    num_blob: int,
    orbit_radius: float,
    omega: float,
    sigma_blob: float,
    noise_ar_coeff: float,
    spatial_bounds: float,
    hidden_dim: int,
    beta: float,
    critic: str,
    epochs: int,
    batch_size: int,
    oom_retry_enabled: bool,
    oom_batch_size_fallbacks: list[int],
    oom_max_retries: int,
    oom_skip_on_failure: bool,
    save_filter_viz: bool,
    filter_layer_idx: int,
    filter_out_dir: Path,
    enable_tensorboard: bool,
    tensorboard_root_dir: Path,
    mask_eval_threshold: float,
    pi_max_lag: int,
    pi_lag_agg: str,
    lr: float,
    early_stop: int,
    device: str,
) -> dict:
    """Run one seed/noise/encoder condition with optional OOM batch fallback."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    (data, gt_latent, particle_order, particle_labels,
     standardization_mean, standardization_std, positions) = generate_drift_diffusion_process_timeseries(
        t_max=t_max,
        num_blob=num_blob,
        num_noise=num_noise,
        orbit_radius=orbit_radius,
        omega=omega,
        sigma_blob=sigma_blob,
        sigma_noise=sigma_noise,
        noise_ar_coeff=noise_ar_coeff,
        spatial_bounds=spatial_bounds,
        seed=seed,
    )

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
                ydim=2,
                xdim=data.shape[1],
                T=T,
                encoder_params=encoder_params,
                mi_params={
                    "estimator_compress": "infonce_lower",
                    "estimator_predictive": "infonce_lower",
                    "critic": critic,
                    "baseline": "constant",
                },
                hidden_dim=hidden_dim,
                beta=beta,
                device=device,
                predictive_space=encoder_spec.predictive_space,
            ).to(device)
            if enable_tensorboard and summary_writer_cls is not None:
                safe_label = encoder_spec.label.replace(" ", "_").replace("(", "").replace(")", "")
                tb_dir = tensorboard_root_dir / f"seed{seed}" / f"noise{num_noise}" / safe_label / f"attempt{attempts}"
                tb_dir.mkdir(parents=True, exist_ok=True)
                writer = summary_writer_cls(log_dir=str(tb_dir))
                tensorboard_log_dir = str(tb_dir)
            model.fit(
                X=train_data,
                epochs=epochs,
                batch_size=curr_batch_size,
                lr=lr,
                early_stop=early_stop,
                writer=writer,
            )

            z_train, gt_train = build_past_windows_and_gt(data, gt_latent, T, model, device, t_min=T, t_max=t_split)
            z_test, gt_test = build_past_windows_and_gt(data, gt_latent, T, model, device, t_min=t_split, t_max=t_max)
            metrics = heldout_probe_r2(z_train, gt_train, z_test, gt_test)
            used_batch_size = curr_batch_size
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

    mask_active_frac = float("nan")
    mask_active_indices = ""
    if model is not None and status == "success":
        mask_active_frac, mask_active_indices = get_final_mask_stats(model, mask_eval_threshold)

    filter_weights_path = ""
    filter_plot_path = ""
    if model is not None and status == "success" and save_filter_viz:
        orbit_grid = None
        avg_density = None
        if getattr(model.encoder, "encoder_type", "") == "conv_physical":
            try:
                _, phys_meta = model.encoder.get_filters(layer_idx=filter_layer_idx)
                grid_size = phys_meta["grid_size"]
                enc_spatial_bounds = phys_meta["spatial_bounds"]
                orbit_grid = compute_orbit_in_grid_coords(
                    positions, particle_labels, particle_order,
                    standardization_mean, standardization_std,
                    grid_size=grid_size,
                    spatial_bounds=enc_spatial_bounds,
                )
                avg_density = compute_avg_density_grid(
                    data[:t_split], grid_size=grid_size, spatial_bounds=enc_spatial_bounds,
                )
            except Exception as exc:
                print(f"[WARN] Could not compute orbit/density for conv_physical: {exc}")

        filter_weights_path, filter_plot_path = save_conv_filter_artifacts(
            model,
            filter_out_dir,
            seed=seed,
            num_noise=num_noise,
            encoder_label=encoder_spec.label,
            layer_idx=filter_layer_idx,
            orbit_grid=orbit_grid,
            avg_density=avg_density,
        )

    if model is not None:
        del model
    cleanup_memory(device)

    return {
        "seed": seed,
        "num_noise": num_noise,
        "sigma_noise": sigma_noise,
        "noise_ar_coeff": noise_ar_coeff,
        "encoder_label": encoder_spec.label,
        "predictive_space": encoder_spec.predictive_space,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "attempts": attempts,
        "used_batch_size": used_batch_size,
        "mask_active_frac": mask_active_frac,
        "mask_active_indices": mask_active_indices,
        "pi_mean_score": float(np.mean(pi_scores)),
        "pi_max_lag": int(pi_max_lag),
        "pi_lag_agg": pi_lag_agg,
        "filter_weights_path": filter_weights_path,
        "filter_plot_path": filter_plot_path,
        "tensorboard_log_dir": tensorboard_log_dir,
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]

    for label, sub in agg.groupby("encoder_label"):
        x = sub["num_noise"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["std"].fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=yerr, fmt='-o', linewidth=2, markersize=5, capsize=3, elinewidth=1.2, label=label)

    ax.set_xlabel("Number of noise particles", fontsize=14)
    ax.set_ylabel(r"Held-out linear probe $R^2$ (mean over x,y)", fontsize=14)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
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
                ax2.errorbar(
                    sub["num_noise"].to_numpy(dtype=float),
                    sub["mean"].to_numpy(dtype=float),
                    yerr=sub["std"].fillna(0.0).to_numpy(dtype=float),
                    fmt="-o",
                    linewidth=2,
                    markersize=5,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch drift-diffusion encoder sweep using config files.")
    parser.add_argument("--config", type=str, default="drift_diffusion_cpic_conv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--signature", type=str, default=None)
    args = parser.parse_args()

    if args.config in config_file_dict:
        config_file = config_file_dict[args.config]
    else:
        raise ValueError(f"Unknown config {args.config!r}. Available: {list(config_file_dict.keys())}")

    cfg = MyConf()
    cfg.read(config_file)

    output_dir = Path(cfg.get("Paths", "output_dir"))
    if args.signature is not None:
        output_dir = output_dir / str(args.signature)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(cfg.get("Sweep", "seeds"))
    num_noise_values = parse_int_list(cfg.get("Sweep", "num_noise_values"))
    sigma_noise = cfg.getfloat("Sweep", "sigma_noise")

    specs = load_encoder_specs(cfg)

    t_max = cfg.getint("Data", "t_max")
    train_ratio = cfg.getfloat("Data", "train_ratio")
    T = cfg.getint("Data", "T")
    num_blob = cfg.getint("Data", "num_blob")
    orbit_radius = cfg.getfloat("Data", "orbit_radius")
    omega = cfg.getfloat("Data", "omega")
    sigma_blob = cfg.getfloat("Data", "sigma_blob")
    noise_ar_coeff = float_or_default(cfg, "Data", "noise_ar_coeff", 0.8)
    spatial_bounds = cfg.getfloat("Data", "spatial_bounds")

    hidden_dim = cfg.getint("Model", "hidden_dim")
    beta = cfg.getfloat("Model", "beta")
    critic = cfg.get("Model", "critic", fallback="concat")

    epochs = cfg.getint("Training", "epochs")
    batch_size = cfg.getint("Training", "batch_size")
    lr = cfg.getfloat("Training", "lr")
    early_stop = cfg.getint("Training", "early_stop")
    oom_retry_enabled = bool_or_default(cfg, "OOM", "oom_retry_enabled", True)
    oom_batch_size_fallbacks = parse_int_list(
        cfg.get("OOM", "oom_batch_size_fallbacks", fallback=str(batch_size))
    )
    oom_max_retries = int_or_default(cfg, "OOM", "oom_max_retries", len(oom_batch_size_fallbacks))
    oom_skip_on_failure = bool_or_default(cfg, "OOM", "oom_skip_on_failure", True)
    save_filter_viz = bool_or_default(cfg, "Analysis", "save_filter_viz", False)
    filter_layer_idx = int_or_default(cfg, "Analysis", "filter_layer_idx", 0)
    filter_out_subdir = cfg.get("Analysis", "filter_out_dir", fallback="filters")
    enable_tensorboard = bool_or_default(cfg, "Analysis", "enable_tensorboard", False)
    tensorboard_subdir = cfg.get("Analysis", "tensorboard_dir", fallback="tensorboard")
    mask_eval_threshold = float_or_default(cfg, "Analysis", "mask_eval_threshold", 0.5)
    pi_max_lag = int_or_default(cfg, "Analysis", "pi_max_lag", 1)
    pi_lag_agg = cfg.get("Analysis", "pi_lag_agg", fallback="mean").strip().lower()

    use_gpu = cfg.getboolean("Compute", "use_gpu")
    device_pref = args.device or cfg.get("Compute", "device", fallback="auto")
    device = pick_device(device_pref, use_gpu)
    print(f"Device: {device}")

    csv_path = output_dir / cfg.get("Paths", "csv_name")
    plot_path = output_dir / cfg.get("Paths", "plot_name")
    filter_out_dir = output_dir / filter_out_subdir
    tensorboard_root_dir = output_dir / tensorboard_subdir
    if save_filter_viz:
        filter_out_dir.mkdir(parents=True, exist_ok=True)
    if enable_tensorboard:
        tensorboard_root_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in seeds:
        for num_noise in num_noise_values:
            for spec in specs:
                row = run_condition(
                    seed=seed,
                    num_noise=num_noise,
                    sigma_noise=sigma_noise,
                    encoder_spec=spec,
                    t_max=t_max,
                    train_ratio=train_ratio,
                    T=T,
                    num_blob=num_blob,
                    orbit_radius=orbit_radius,
                    omega=omega,
                    sigma_blob=sigma_blob,
                    noise_ar_coeff=noise_ar_coeff,
                    spatial_bounds=spatial_bounds,
                    hidden_dim=hidden_dim,
                    beta=beta,
                    critic=critic,
                    epochs=epochs,
                    batch_size=batch_size,
                    oom_retry_enabled=oom_retry_enabled,
                    oom_batch_size_fallbacks=oom_batch_size_fallbacks,
                    oom_max_retries=oom_max_retries,
                    oom_skip_on_failure=oom_skip_on_failure,
                    save_filter_viz=save_filter_viz,
                    filter_layer_idx=filter_layer_idx,
                    filter_out_dir=filter_out_dir,
                    enable_tensorboard=enable_tensorboard,
                    tensorboard_root_dir=tensorboard_root_dir,
                    mask_eval_threshold=mask_eval_threshold,
                    pi_max_lag=pi_max_lag,
                    pi_lag_agg=pi_lag_agg,
                    lr=lr,
                    early_stop=early_stop,
                    device=device,
                )
                rows.append(row)
                print(
                    f"seed={seed:>2} | num_noise={num_noise:>3} | {spec.label:<16} "
                    f"| status={row['status']:<10} | batch={row['used_batch_size']:>4} "
                    f"-> R2_test_mean={row['r2_test_mean']:.3f}"
                )

    save_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    plot_r2_vs_num_noise(rows, plot_path)
    print(f"Wrote {plot_path}")
    