from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from filter_visualization import regenerate_physical_filter_plots


VALID_METRICS = {"r2_test_mean", "r2_test_x", "r2_test_y", "mask_active_frac"}
METRIC_TO_YLABEL = {
    "r2_test_mean": r"Test $R^2$ (linear-probe)",
    "r2_test_x": r"Test $R^2$ (linear-probe, x-dim)",
    "r2_test_y": r"Test $R^2$ (linear-probe, y-dim)",
    "mask_active_frac": "Active mask fraction",
}

# Flip True/False to quickly control which curves appear when no CLI
# include/exclude filters are provided.
DEFAULT_ENCODER_PLOT_ENABLED = {
    "MLP (L)": True,
    "MLP (O)": True,
    "ConvSpatial (L)": True,
    "ConvSpatial (O)": True,
    "ConvParticle (L)": True,
    "ConvParticle (O)": True,
    "ConvPhysical (L)": True,
    "ConvPhysical (O)": True,
    "MaskRandom (L)": False,
    "MaskRandom (O)": False,
    "MaskPIStatic (L)": False,
    "MaskPIStatic (O)": False,
    "MaskPIInitLearned (L)": False,
    "MaskPIInitLearned (O)": False,
}

# For mask-specific auxiliary panels, keep one predictive-space variant per
# mask strategy to avoid redundant overlapping curves.
PREFERRED_MASK_SUFFIX = "(O)"

# Stable, simplified colors: one color per encoder family.
ENCODER_BASE_COLOR_MAP = {
    "MLP": "#1f77b4",
    "ConvSpatial": "#ff7f0e",
    "ConvParticle": "#2ca02c",
    "ConvPhysical": "#17becf",
    "MaskRandom": "#d62728",
    "MaskPIStatic": "#9467bd",
    "MaskPIInitLearned": "#8c564b",
}

# Predictive-space style:
# - observation: filled markers + solid lines
# - latent: unfilled markers + dotted lines
LATENT_LINESTYLE = ":"
OBS_LINESTYLE = "-"


def parse_label_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _mask_base_label(label: str) -> str:
    if label.endswith(" (L)") or label.endswith(" (O)"):
        return label[:-4]
    return label


def _line_style_for_label(label: str) -> str:
    if label.endswith(" (L)"):
        return LATENT_LINESTYLE
    if label.endswith(" (O)"):
        return OBS_LINESTYLE
    return "-"


def _is_observation_label(label: str) -> bool:
    return label.endswith(" (O)")


def _base_family_from_label(label: str) -> str:
    base = _mask_base_label(label)
    if " (" in base:
        base = base.split(" (", 1)[0]
    return base


def _color_for_label(label: str) -> str | None:
    family = _base_family_from_label(label)
    return ENCODER_BASE_COLOR_MAP.get(family, None)


def _marker_facecolor_for_label(label: str, color: str | None) -> str:
    if _is_observation_label(label):
        return color if color is not None else "auto"
    return "none"


def select_relevant_mask_variants(mask_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one predictive-space variant per mask strategy.

    Preference order: observation label "(O)" first, then latent "(L)".
    """
    if mask_df.empty:
        return mask_df

    keep_labels: list[str] = []
    for _, group in mask_df.groupby(mask_df["encoder_label"].map(_mask_base_label)):
        labels = sorted(group["encoder_label"].dropna().unique().tolist())
        preferred = [lbl for lbl in labels if lbl.endswith(PREFERRED_MASK_SUFFIX)]
        if preferred:
            keep_labels.append(preferred[0])
        else:
            keep_labels.append(labels[0])
    return mask_df[mask_df["encoder_label"].isin(keep_labels)].copy()


def compute_mask_target_delta(
    mask_df: pd.DataFrame,
    *,
    num_blob: int,
) -> pd.DataFrame:
    """Compute target-selection lift over random expectation for mask encoders.

    target_delta = observed_target_frac - expected_target_frac_if_random
    where each particle contributes 2 dimensions.
    """
    target_dims = 2 * int(num_blob)
    rows: list[dict[str, float | str]] = []
    for _, row in mask_df.iterrows():
        idx_text = row.get("mask_active_indices", "")
        if not isinstance(idx_text, str) or not idx_text.strip():
            continue
        try:
            active = [int(x) for x in idx_text.split(",") if x.strip()]
        except ValueError:
            continue
        if not active:
            continue

        target_count = sum(1 for i in active if i < target_dims)
        observed_target_frac = target_count / len(active)
        total_dims = target_dims + 2 * int(float(row["num_noise"]))
        expected_random_frac = target_dims / total_dims if total_dims > 0 else 0.0
        rows.append(
            {
                "encoder_label": row["encoder_label"],
                "num_noise": float(row["num_noise"]),
                "target_delta": observed_target_frac - expected_random_frac,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["encoder_label", "num_noise", "mean", "std"])

    delta_df = pd.DataFrame(rows)
    return (
        delta_df.groupby(["encoder_label", "num_noise"], as_index=False)["target_delta"]
        .agg(mean="mean", std="std")
        .sort_values(["encoder_label", "num_noise"])
    )


def plot_from_csv(
    csv_path: Path,
    out_path: Path,
    metric: str,
    title: str | None,
    std: bool = False,
    include_encoders: list[str] | None = None,
    exclude_encoders: list[str] | None = None,
    include_mask_panel: bool = False,
    include_mask_target_panel: bool = False,
    num_blob: int = 30,
) -> None:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got {metric!r}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"encoder_label", "num_noise", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    if not include_encoders and not exclude_encoders:
        include_encoders = [label for label, enabled in DEFAULT_ENCODER_PLOT_ENABLED.items() if enabled]

    include_set = set(include_encoders or [])
    exclude_set = set(exclude_encoders or [])
    available_labels = sorted(df["encoder_label"].dropna().unique().tolist())

    unknown_includes = sorted(include_set - set(available_labels))
    if unknown_includes:
        raise ValueError(
            f"Unknown encoder labels in --include-encoders: {unknown_includes}. "
            f"Available: {available_labels}"
        )
    unknown_excludes = sorted(exclude_set - set(available_labels))
    if unknown_excludes:
        raise ValueError(
            f"Unknown encoder labels in --exclude-encoders: {unknown_excludes}. "
            f"Available: {available_labels}"
        )

    if include_set:
        df = df[df["encoder_label"].isin(include_set)].copy()
    if exclude_set:
        df = df[~df["encoder_label"].isin(exclude_set)].copy()
    if df.empty:
        raise ValueError("No rows left after encoder filtering. Check include/exclude encoder arguments.")

    agg = (
        df.groupby(["encoder_label", "num_noise"], as_index=False)[metric]
        .agg(mean="mean", std="std")
        .sort_values(["encoder_label", "num_noise"])
    )

    panel_count = 1 + int(include_mask_panel) + int(include_mask_target_panel)
    if panel_count > 1:
        fig, axes = plt.subplots(1, panel_count, figsize=(6.5 * panel_count, 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        ax = axes[0]
        ax2 = axes[1] if include_mask_panel else None
        ax3 = axes[-1] if include_mask_target_panel else None
    else:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax2 = None
        ax3 = None

    for label, sub in agg.groupby("encoder_label"):
        x = sub["num_noise"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["std"].fillna(0.0).to_numpy(dtype=float)
        color = _color_for_label(label)
        linestyle = _line_style_for_label(label)
        markerfacecolor = _marker_facecolor_for_label(label, color)
        
        if std:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                linestyle=linestyle,
                color=color,
                markerfacecolor=markerfacecolor,
                linewidth=2,
                markersize=5,
                capsize=3,
                elinewidth=1.2,
                label=label,
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                linestyle=linestyle,
                color=color,
                markerfacecolor=markerfacecolor,
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Number of noise particles", fontsize=14)
    ax.set_ylabel(METRIC_TO_YLABEL.get(metric, metric), fontsize=14)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title(title or f"Particle orbit: {metric} vs noise particles", fontsize=14)

    if include_mask_panel and ax2 is not None:
        if "mask_active_frac" in df.columns:
            mask_df = df[df["encoder_label"].str.contains("Mask", na=False)]
            mask_df = select_relevant_mask_variants(mask_df)
            if not mask_df.empty:
                mask_agg = (
                    mask_df.groupby(["encoder_label", "num_noise"], as_index=False)["mask_active_frac"]
                    .agg(mean="mean", std="std")
                    .sort_values(["encoder_label", "num_noise"])
                )

                markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
                # Small deterministic x-jitter to separate overlapping curves visually.
                jitter = np.linspace(-1.5, 1.5, num=max(1, mask_agg["encoder_label"].nunique()))

                for idx, (label, sub) in enumerate(mask_agg.groupby("encoder_label")):
                    x = sub["num_noise"].to_numpy(dtype=float)
                    y = sub["mean"].to_numpy(dtype=float)
                    yerr = sub["std"].fillna(0.0).to_numpy(dtype=float)
                    x_plot = x + jitter[idx]
                    marker = markers[idx % len(markers)]
                    linestyle = _line_style_for_label(label)
                    color = _color_for_label(label)
                    markerfacecolor = _marker_facecolor_for_label(label, color)
                    if std:
                        ax2.errorbar(
                            x_plot,
                            y,
                            yerr=yerr,
                            fmt=marker,
                            linestyle=linestyle,
                            color=color,
                            markerfacecolor=markerfacecolor,
                            linewidth=2,
                            markersize=5,
                            capsize=3,
                            elinewidth=1.2,
                            label=label,
                        )
                    else:
                        ax2.plot(
                            x_plot,
                            y,
                            marker=marker,
                            linestyle=linestyle,
                            color=color,
                            markerfacecolor=markerfacecolor,
                            linewidth=2,
                            label=label,
                        )
                ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
                ax2.set_ylabel("Active mask fraction", fontsize=14)
                ax2.set_title("Mask sparsity vs noise", fontsize=14)
            else:
                ax2.text(0.5, 0.5, "No mask runs available", ha="center", va="center")
        else:
            ax2.text(0.5, 0.5, "mask_active_frac not found in CSV", ha="center", va="center")
        ax2.set_xlabel("Number of noise particles", fontsize=14)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.set_ylim(-0.05, 1.05)

    if include_mask_target_panel and ax3 is not None:
        if {"mask_active_indices", "num_noise", "encoder_label"}.issubset(df.columns):
            mask_df = df[df["encoder_label"].str.contains("Mask", na=False)].copy()
            mask_df = select_relevant_mask_variants(mask_df)
            delta_agg = compute_mask_target_delta(mask_df, num_blob=num_blob)
            if not delta_agg.empty:
                for label, sub in delta_agg.groupby("encoder_label"):
                    x = sub["num_noise"].to_numpy(dtype=float)
                    y = sub["mean"].to_numpy(dtype=float)
                    yerr = sub["std"].fillna(0.0).to_numpy(dtype=float)
                    color = _color_for_label(label)
                    linestyle = _line_style_for_label(label)
                    markerfacecolor = _marker_facecolor_for_label(label, color)
                    if std:
                        ax3.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="o",
                            linestyle=linestyle,
                            color=color,
                            markerfacecolor=markerfacecolor,
                            linewidth=2,
                            markersize=5,
                            capsize=3,
                            elinewidth=1.2,
                            label=label,
                        )
                    else:
                        ax3.plot(
                            x,
                            y,
                            marker="o",
                            linestyle=linestyle,
                            color=color,
                            markerfacecolor=markerfacecolor,
                            linewidth=2,
                            label=label,
                        )
                ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            else:
                ax3.text(0.5, 0.5, "No mask indices available", ha="center", va="center")
        else:
            ax3.text(0.5, 0.5, "mask_active_indices not found in CSV", ha="center", va="center")
        ax3.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        ax3.set_xlabel("Number of noise particles", fontsize=14)
        ax3.set_ylabel("Target selection lift vs random", fontsize=14)
        ax3.set_title("Mask target-vs-noise selection", fontsize=14)
        ax3.tick_params(axis="both", labelsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def _parse_optional_int_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    return vals or None


def export_physical_filter_gallery(
    csv_path: Path,
    gallery_dir: Path,
    *,
    encoder_labels: list[str] | None = None,
    noise_levels: list[int] | None = None,
    seeds: list[int] | None = None,
    regenerate: bool = False,
) -> list[dict[str, str]]:
    """
    Collect or regenerate ConvPhysical filter PNGs from a sweep CSV.

    Copies existing ``filter_plot_path`` / ``filter_orbit_density_path`` files into
    ``gallery_dir``, or rebuilds PNGs from ``filter_weights_path`` when
    ``regenerate=True``.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    mask = df["encoder_label"].astype(str).str.contains("ConvPhysical", na=False)
    if "status" in df.columns:
        mask &= df["status"].astype(str) == "success"
    subset = df[mask].copy()
    if encoder_labels:
        subset = subset[subset["encoder_label"].isin(encoder_labels)]
    if noise_levels is not None:
        subset = subset[subset["num_noise"].isin(noise_levels)]
    if seeds is not None:
        subset = subset[subset["seed"].isin(seeds)]

    gallery_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for _, row in subset.iterrows():
        label = str(row["encoder_label"])
        seed = int(row["seed"])
        num_noise = int(row["num_noise"])
        stem = f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_noise{num_noise}_seed{seed}"

        weights_path = str(row.get("filter_weights_path", "") or "").strip()
        plot_path = str(row.get("filter_plot_path", "") or "").strip()
        orbit_path = str(row.get("filter_orbit_density_path", "") or "").strip()
        meta_path = str(row.get("filter_meta_path", "") or "").strip()

        if regenerate and weights_path and Path(weights_path).exists():
            meta_candidate = meta_path or str(Path(weights_path).with_suffix(".json"))
            paths = regenerate_physical_filter_plots(
                weights_path,
                meta_candidate if Path(meta_candidate).exists() else None,
            )
            plot_path = paths.get("filter_plot_path", plot_path)
            orbit_path = paths.get("filter_orbit_density_path", orbit_path)

        entry = {
            "encoder_label": label,
            "seed": str(seed),
            "num_noise": str(num_noise),
            "gallery_filter_plot": "",
            "gallery_orbit_density": "",
        }

        if plot_path and Path(plot_path).exists():
            dst = gallery_dir / f"{stem}_filters.png"
            shutil.copy2(plot_path, dst)
            entry["gallery_filter_plot"] = str(dst)

        if orbit_path and Path(orbit_path).exists():
            dst_orbit = gallery_dir / f"{stem}_orbit_density.png"
            shutil.copy2(orbit_path, dst_orbit)
            entry["gallery_orbit_density"] = str(dst_orbit)

        if entry["gallery_filter_plot"] or entry["gallery_orbit_density"]:
            manifest.append(entry)

    if manifest:
        manifest_path = gallery_dir / "physical_filter_gallery_manifest.csv"
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        print(f"Wrote {len(manifest)} gallery entries to {gallery_dir}")
        print(f"Manifest: {manifest_path}")
    else:
        print("No ConvPhysical filter artifacts found for the given filters.")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot particle-orbit results from CSV.")
    parser.add_argument("--csv", type=Path, default="experiments/particle_orbit_experiment/res/particle_orbit_probe_runs.csv", help="Path to particle_orbit_probe_runs.csv")
    parser.add_argument("--out", type=Path, default="experiments/particle_orbit_experiment/res/R2_vs_num_noise_particles.png", help="Output image path (e.g., .png)")
    parser.add_argument(
        "--metric",
        type=str,
        default="r2_test_mean",
        choices=sorted(VALID_METRICS),
        help="Which metric column to visualize.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="$R^2$ vs noise particles",
        help="Optional custom title. If omitted, a default title is used.",
    )
    parser.add_argument(
        "--include-encoders",
        type=str,
        default=None,
        help="Comma-separated encoder labels to include (exact match). "
        "Example: 'MLP (O),ConvSpatial (O)'.",
    )
    parser.add_argument(
        "--exclude-encoders",
        type=str,
        default=None,
        help="Comma-separated encoder labels to exclude (exact match).",
    )
    parser.add_argument(
        "--include-mask-panel",
        action="store_true",
        help="If set, add a second panel with mask_active_frac vs num_noise for mask encoders.",
    )
    parser.add_argument(
        "--include-mask-target-panel",
        action="store_true",
        help="If set, add panel for target-selection lift over random expectation using mask indices.",
    )
    parser.add_argument(
        "--num-blob",
        type=int,
        default=30,
        help="Number of target particles used to compute target-vs-noise selection lift (default: 30).",
    )
    parser.add_argument(
        "--export-physical-filters",
        action="store_true",
        help="Copy or regenerate ConvPhysical filter PNGs from CSV paths into a gallery folder.",
    )
    parser.add_argument(
        "--physical-filter-gallery-dir",
        type=Path,
        default=None,
        help="Output directory for --export-physical-filters (default: <csv_parent>/physical_filter_gallery).",
    )
    parser.add_argument(
        "--physical-filter-encoders",
        type=str,
        default=None,
        help="Comma-separated ConvPhysical labels to include (default: all ConvPhysical rows).",
    )
    parser.add_argument(
        "--physical-filter-noise",
        type=str,
        default=None,
        help="Comma-separated num_noise values to include.",
    )
    parser.add_argument(
        "--physical-filter-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to include.",
    )
    parser.add_argument(
        "--regenerate-physical-filters",
        action="store_true",
        help="Rebuild PNGs from saved .npy weights before copying to the gallery.",
    )
    parser.add_argument(
        "--skip-r2-plot",
        action="store_true",
        help="Skip the R2 curve plot (useful with --export-physical-filters only).",
    )
    args = parser.parse_args()

    include_encoders = parse_label_list(args.include_encoders)
    exclude_encoders = parse_label_list(args.exclude_encoders)

    if args.export_physical_filters:
        gallery_dir = args.physical_filter_gallery_dir
        if gallery_dir is None:
            gallery_dir = args.csv.resolve().parent / "physical_filter_gallery"
        export_physical_filter_gallery(
            args.csv,
            gallery_dir,
            encoder_labels=parse_label_list(args.physical_filter_encoders) or None,
            noise_levels=_parse_optional_int_list(args.physical_filter_noise),
            seeds=_parse_optional_int_list(args.physical_filter_seeds),
            regenerate=args.regenerate_physical_filters,
        )

    if not args.skip_r2_plot:
        plot_from_csv(
            args.csv,
            args.out,
            args.metric,
            args.title,
            std=True,
            include_encoders=include_encoders,
            exclude_encoders=exclude_encoders,
            include_mask_panel=args.include_mask_panel,
            include_mask_target_panel=args.include_mask_target_panel,
            num_blob=args.num_blob,
        )
        print(f"Wrote {args.out}")