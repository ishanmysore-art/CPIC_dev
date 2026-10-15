"""Compare saved CPIC M1 runs without recomputing or inventing measurements.

Example:
    python plot_mlp_comparison.py \
        --input Linear=res/m1_scaling_benchmark/linear.pkl \
        --input MLP=res/m1_scaling_benchmark/mlp.pkl \
        --output-dir plots/mlp_comparison

Accepts current result dictionaries and older ``[r2, mi]`` pickles. Mask plots
are made only when both mask-replicate arrays and their missingness labels exist.
For each missingness, points are individual independent mask replicates and the
error bar is mean ± sample SD across those replicates (SD=0 for one replicate).
Saved ``mi`` values are from the best-loss epoch, whereas optional diagnostic
snapshots are first-batch values at the named epochs.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_run(spec):
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
    path = Path(raw_path).expanduser()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        result = payload
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        result = {"r2": payload[0], "mi": payload[1]}
    else:
        raise ValueError(f"Unsupported CPIC result format: {path}")
    return {"label": label, "path": path, "result": result}


def masked_values(run, key, dim_index, offset_index, t_index):
    result = run["result"]
    if key not in result or "eval_missing_ps" not in result.get("masking", {}):
        return None
    values = np.asarray(result[key], dtype=float)
    if values.ndim != 5:
        raise ValueError(f"{run['path']}: {key} must have axes [dim, missingness, replicate, offset, T]")
    try:
        selected = values[dim_index, :, :, offset_index, t_index]
    except IndexError as exc:
        raise ValueError(f"{run['path']}: selection out of range for {key} shape {values.shape}") from exc
    missingness = np.asarray(result["masking"]["eval_missing_ps"], dtype=float)
    if selected.shape[0] != len(missingness):
        raise ValueError(f"{run['path']}: {key} missingness axis disagrees with eval_missing_ps")
    if selected.shape[1] < 1 or not np.isfinite(selected).all():
        raise ValueError(f"{run['path']}: {key} has no replicates or contains non-finite values")
    order = np.argsort(missingness)
    return missingness[order], selected[order]


def mean_sd(values):
    return values.mean(axis=1), values.std(axis=1, ddof=1 if values.shape[1] > 1 else 0)


def draw_replicates(ax, x, values, color):
    # All replicates use the exact missingness location; no random jitter is added.
    for replicate in range(values.shape[1]):
        ax.scatter(x, values[:, replicate], s=11, alpha=0.22, color=color, zorder=2)


def draw_summary(ax, x, values, label, color):
    mean, sd = mean_sd(values)
    draw_replicates(ax, x, values, color)
    ax.errorbar(x, mean, yerr=sd, marker="o", markersize=4, linewidth=1.6,
                capsize=3, color=color, label=label, zorder=3)


def finish_axis(ax, ylabel, symlog):
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    ax.set_xlabel("Missing neurons (%)")
    ax.set_ylabel(ylabel)
    if symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.text(0.02, 0.02, "Symmetric log R² axis; linear within ±1",
                transform=ax.transAxes, fontsize=7, color="0.35")
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def save_figure(fig, output_dir, stem):
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        with path.open("xb") as handle:
            fig.savefig(handle, format=extension, bbox_inches="tight")
    plt.close(fig)


def predictive_value(run, dim_index, t_index):
    if "mi" not in run["result"]:
        return None
    values = np.asarray(run["result"]["mi"], dtype=float)
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError(f"{run['path']}: mi must have axes [dim, T, compression/prediction]")
    try:
        value = values[dim_index, t_index, 1]
    except IndexError as exc:
        raise ValueError(f"{run['path']}: selection out of range for mi shape {values.shape}") from exc
    return float(value) if np.isfinite(value) else None


def diagnostic_rows(run, dim_index, t_index):
    result = run["result"]
    rows = result.get("mlp_diagnostics", [])
    t_values = result.get("config", {}).get("T")
    selected_t = t_values[t_index] if t_values is not None else None
    selected_dim = result.get("ydim") if dim_index == 0 else None
    return sorted((row for row in rows
                   if (selected_t is None or row.get("T", selected_t) == selected_t)
                   and (selected_dim is None or row.get("ydim", selected_dim) == selected_dim)
                   and np.isfinite(row.get("I_predictive", np.nan))),
                  key=lambda row: row["epoch"])


def make_plots(runs, output_dir, dim_index=0, offset_index=0, t_index=0,
               symlog=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    t_values = {tuple(run["result"].get("config", {}).get("T", [])) for run in runs}
    common_t = next(iter(t_values)) if len(t_values) == 1 else ()
    selected_t = common_t[t_index] if 0 <= t_index < len(common_t) else None
    context = (f"T={selected_t}" if selected_t is not None else f"T index {t_index}")
    context += f"; decoding offset index {offset_index}"
    masked = []
    for run in runs:
        frozen = masked_values(run, "frozen_mask_eval_r2", dim_index, offset_index, t_index)
        refit = masked_values(run, "mask_eval_r2", dim_index, offset_index, t_index)
        if frozen is not None and refit is not None:
            if not np.array_equal(frozen[0], refit[0]):
                raise ValueError(f"{run['path']}: refit/frozen missingness values differ")
            masked.append((run["label"], frozen, refit))
        else:
            print(f"Skipping mask plots for {run['label']}: mask replicate arrays unavailable")

    colors = plt.get_cmap("tab10")
    stems = []
    if masked:
        stems += ["frozen_r2_vs_missingness", "refit_vs_frozen_r2"]
    predictive = [(run["label"], predictive_value(run, dim_index, t_index),
                   diagnostic_rows(run, dim_index, t_index)) for run in runs]
    if any(value is not None or rows for _, value, rows in predictive):
        stems.append("predictive_information")
    if not stems:
        raise ValueError("No requested metrics are available in the supplied pickle files")
    for stem in stems:
        for extension in ("pdf", "png"):
            path = output_dir / f"{stem}.{extension}"
            if path.exists():
                raise FileExistsError(f"Plot already exists: {path}")

    if masked:
        fig, ax = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
        for index, (label, frozen, _) in enumerate(masked):
            draw_summary(ax, frozen[0] * 100, frozen[1], label, colors(index % 10))
        finish_axis(ax, "Frozen clean-decoder R²", symlog)
        ax.set_title(context)
        ax.legend(frameon=False, title="Encoder")
        save_figure(fig, output_dir, "frozen_r2_vs_missingness")

        fig, axes = plt.subplots(len(masked), 1, figsize=(6.8, 3.3 * len(masked)),
                                 squeeze=False, layout="constrained")
        for ax, (label, frozen, refit) in zip(axes[:, 0], masked):
            draw_summary(ax, refit[0] * 100, refit[1], "Refit on masked latents", colors(0))
            draw_summary(ax, frozen[0] * 100, frozen[1], "Frozen clean decoder", colors(1))
            finish_axis(ax, f"{label}: behavioral R²", symlog)
            ax.legend(frameon=False)
        fig.suptitle(context)
        save_figure(fig, output_dir, "refit_vs_frozen_r2")

    if "predictive_information" in stems:
        has_diagnostics = any(rows for _, _, rows in predictive)
        has_saved_mi = any(value is not None for _, value, _ in predictive)
        fig, axes = plt.subplots(1, int(has_saved_mi) + int(has_diagnostics),
                                 figsize=(9.2 if has_saved_mi and has_diagnostics else 6.2, 3.8),
                                 squeeze=False, layout="constrained")
        if has_saved_mi:
            left = axes[0, 0]
            for index, (label, value, _) in enumerate(predictive):
                if value is not None:
                    left.scatter(index, value, s=45, color=colors(index % 10), zorder=3)
            left.set_xticks(range(len(predictive)), [label for label, _, _ in predictive], rotation=25, ha="right")
            left.set_ylabel("Predictive InfoNCE bound (nats)")
            left.set_title("Saved best-loss epoch")
            left.axhline(0, color="0.6", linewidth=0.8)
            left.grid(axis="y", color="0.9", linewidth=0.6)
        if has_diagnostics:
            right = axes[0, int(has_saved_mi)]
            for index, (label, _, rows) in enumerate(predictive):
                if rows:
                    right.plot([row["epoch"] for row in rows],
                               [row["I_predictive"] for row in rows],
                               marker="o", markersize=4, color=colors(index % 10), label=label)
            right.axhline(0, color="0.6", linewidth=0.8)
            right.set_xlabel("Training epoch")
            right.set_ylabel("Predictive InfoNCE bound (nats)")
            right.set_title("Diagnostic first batch")
            right.grid(axis="y", color="0.9", linewidth=0.6)
            right.legend(frameon=False)
        fig.suptitle(context)
        save_figure(fig, output_dir, "predictive_information")
    return stems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH",
                        help="Trusted CPIC result pickle; repeat for each condition")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim-index", type=int, default=0)
    parser.add_argument("--offset-index", type=int, default=0,
                        help="Behavioral decoding offset index (runner uses 0,1,2 for 5,10,15)")
    parser.add_argument("--t-index", type=int, default=0)
    parser.add_argument("--symlog", action="store_true",
                        help="Use a symmetric log R² axis, linear within ±1, for extreme negative values")
    args = parser.parse_args(argv)
    runs = [load_run(spec) for spec in args.input]
    labels = [run["label"] for run in runs]
    if len(labels) != len(set(labels)):
        parser.error("Each --input label must be unique")
    stems = make_plots(runs, args.output_dir, args.dim_index,
                       args.offset_index, args.t_index, args.symlog)
    print("Created: " + ", ".join(str(args.output_dir / stem) + ".{pdf,png}" for stem in stems))


if __name__ == "__main__":
    main()
