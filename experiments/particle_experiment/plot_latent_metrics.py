"""Latent-quality metrics: position vs velocity vs noise-cloud decodability.

The premise is that linear-probe R2 on the blob *position* is the wrong yardstick for
latent space — it rewards "observation" mode that models high-variance, temporally
predictable noise. This script plots, against a sweep axis (number of noise or blob
particles), three linear-probe R2 families written by ``run_particle_experiment.py``:

  1. POSITION  R2 (``r2_test_mean``)    — the legacy metric (blob centroid position).
  2. VELOCITY  R2 (``r2_velocity_mean``) — first-order dynamics; a coherent-blob-only
     signal (the random-walk noise particles average to no net velocity), so this is
     the honest "does the latent track the orbit" number.
  3. NOISE     R2 (``r2_noise_mean``)   — decodability of the noise-cloud centroid; a
     noise-suppressing latent should sit near the floor here.

The story is the CONTRAST: high velocity-R2 with low noise-R2 means the latent tracks
dynamics while suppressing noise, even where position-R2 (obs-inflated) disagrees. A 4th
panel plots the estimator-native latent-quality number ``final_I_predictive``.

Styling follows house convention: FeatureMaskMLP green, observation = solid line +
filled marker, latent = dotted + hollow, legend outside the axes, error bars = +/-1 std
over seeds.

Usage (from the repo root):
    PYTHONPATH=src uv run python experiments/particle_experiment/plot_latent_metrics.py \
        --res-root experiments/particle_experiment/res/noise_sweep --x-col num_noise
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MASK_GREEN = "#2ca02c"
LATENT_LINESTYLE = ":"
OBS_LINESTYLE = "-"

# (column, axis title, y-label). Panels rendered left-to-right in this order.
METRIC_PANELS = [
    ("r2_test_mean", "Position R²", "linear-probe R²  (position)"),
    ("r2_velocity_mean", "Velocity R²", "linear-probe R²  (velocity)"),
    ("r2_noise_mean", "Noise-cloud R²", "linear-probe R²  (noise centroid)"),
    ("final_I_predictive", "Predictive info", "I_predictive  (nats)"),
]


def _space_style(predictive_space: str) -> tuple[str, bool]:
    """observation -> (solid, filled marker); latent -> (dotted, hollow)."""
    is_latent = str(predictive_space).strip().lower().startswith("lat")
    return (LATENT_LINESTYLE if is_latent else OBS_LINESTYLE, not is_latent)


def _space_suffix(predictive_space: str) -> str:
    return "(L)" if str(predictive_space).strip().lower().startswith("lat") else "(O)"


def collect(res_root: Path, x_col: str) -> pd.DataFrame:
    """Glob every ``*_runs.csv`` under ``res_root`` (recursively); keep successes."""
    paths = sorted(glob.glob(str(res_root / "**" / "*_runs.csv"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(str(res_root / "*_runs.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No run CSVs found under {res_root} (looked for **/*_runs.csv then *_runs.csv)."
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source"] = p
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.lower() == "success"].copy()
    combined = combined.dropna(subset=[x_col])
    combined = combined.sort_values(x_col).reset_index(drop=True)
    print(f"Collected {len(combined)} successful run(s) across {len(paths)} CSV file(s).")
    return combined


# Human-readable axis labels for the supported sweep axes.
_X_AXES = {
    "num_blob": "number of blob particles",
    "num_noise": "number of noise particles",
}


def plot(df: pd.DataFrame, out_dir: Path, x_col: str = "num_noise") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [p for p in METRIC_PANELS if p[0] in df.columns]
    if not panels:
        raise ValueError(
            "None of the latent-metric columns are present — was the CSV written by an "
            "older runner? Expected any of: " + ", ".join(c for c, _, _ in METRIC_PANELS)
        )
    if x_col not in df.columns:
        raise ValueError(f"x-axis column {x_col!r} not in CSV; have: {list(df.columns)}")
    x_label = _X_AXES.get(x_col, x_col)
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), squeeze=False)
    axes = axes[0]

    spaces = sorted(df.get("predictive_space", pd.Series(["observation"])).unique())
    for ax, (col, title, ylabel) in zip(axes, panels):
        for space in spaces:
            sub = df[df["predictive_space"] == space] if "predictive_space" in df.columns else df
            agg = (
                sub.groupby(x_col, as_index=False)
                .agg(val=(col, "mean"), val_std=(col, "std"))
                .sort_values(x_col)
            )
            agg["val_std"] = agg["val_std"].fillna(0.0)
            linestyle, filled = _space_style(space)
            mfc = MASK_GREEN if filled else "none"
            ax.errorbar(
                agg[x_col], agg["val"], yerr=agg["val_std"],
                marker="o", linestyle=linestyle, color=MASK_GREEN,
                markerfacecolor=mfc, markeredgecolor=MASK_GREEN,
                linewidth=2, markersize=7, capsize=3, ecolor=MASK_GREEN, elinewidth=1,
                label=f"FeatureMaskMLP {_space_suffix(space)}",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")

    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / f"latent_metrics_vs_{x_col}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {out_path}")
    return out_path


def main() -> None:
    default_root = Path(__file__).resolve().parent / "res" / "noise_sweep"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-root", type=Path, default=default_root,
                    help="Directory holding the run CSVs (searched recursively).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the figure (default: --res-root).")
    ap.add_argument("--x-col", type=str, default="num_noise",
                    choices=sorted(_X_AXES.keys()),
                    help="Sweep axis for the x-axis (default: num_noise).")
    args = ap.parse_args()

    out_dir = args.out_dir or args.res_root
    df = collect(args.res_root, args.x_col)

    keep = [c for c in [
        "num_blob", "num_noise", "predictive_space", "encoder_label",
        "seed", "r2_test_mean", "r2_velocity_mean", "r2_noise_mean", "final_I_predictive",
        "final_I_compress", "blob_sel_rate", "__source",
    ] if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n" + df[keep].to_string(index=False))

    plot(df, out_dir, x_col=args.x_col)


if __name__ == "__main__":
    main()
