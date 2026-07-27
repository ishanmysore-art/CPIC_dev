"""Complexity curve: predictivity vs compression complexity I(X;Z), traced over the beta sweep.

The information-bottleneck trade-off figure. For each encoder we run the soft-penalty
objective ``L = beta*I_compress - beta1*I_predictive`` at several beta values; increasing
beta squeezes the latent, lowering the compression complexity ``I(X;Z)`` (x-axis) at some
cost to predictivity (y-axis). Plotting predictivity against the ACHIEVED complexity
(``final_I_compress``) rather than against beta gives the honest curve: a good encoder reaches
the same predictive ceiling at a fraction of the complexity.

Each point is a beta value; points are joined in beta order so the line traces the sweep.
x and y are means over seeds with +/-1 std error bars.

Reads the columns written by ``run_particle_experiment.py``:
``final_I_compress`` (complexity, x), and predictivity (y) = ``r2_test_mean`` (default,
linear-probe R2) or ``final_I_predictive`` (``--y-col final_I_predictive``), grouped by
(encoder x predictive-space x beta).

Styling (house convention): FeatureMaskMLP green, ConvPhysical blue, MLP gray; observation =
solid line + filled marker, latent = dotted + hollow; legend outside the axes.

Usage (from the repo root):
    PYTHONPATH=src uv run python experiments/particle_experiment/plot_complexity_curve.py \
        --res-root experiments/particle_experiment/res/poster_headline
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Encoder colors (house convention). FeatureMaskMLP green, ConvPhysical blue, MLP gray.
_MMLP_GREEN = "#2ca02c"
_CNN_BLUE = "#0072b2"
_MLP_GRAY = "#6e6e6e"

# y-axis metric -> axis label.
_Y_LABELS = {
    "r2_test_mean": "linear-probe R²  (position)",
    "final_I_predictive": "predictive info  I_pred  (nats)",
}


def _encoder_display(label: str) -> tuple[str, str]:
    """Map a raw encoder_label (e.g. 'MaskUniformLearned (L)') to (display_name, color)."""
    base = str(label).split(" (")[0].strip().lower()
    if "mask" in base or "feature" in base:
        return "FeatureMaskMLP", _MMLP_GREEN
    if "conv" in base or "physical" in base:
        return "ConvPhysical", _CNN_BLUE
    return "MLP", _MLP_GRAY


def _space_of(row_space: str, label: str) -> str:
    """Return 'latent' or 'observation' from the predictive_space column, falling back to the
    (L)/(O) suffix in the encoder label."""
    s = str(row_space).strip().lower()
    if s.startswith("lat"):
        return "latent"
    if s.startswith("obs"):
        return "observation"
    return "latent" if "(l)" in str(label).lower() else "observation"


def collect(res_root: Path) -> pd.DataFrame:
    """Glob every ``*_runs.csv`` under ``res_root`` (recursively); keep successful rows."""
    paths = sorted(glob.glob(str(res_root / "**" / "*_runs.csv"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(str(res_root / "*_runs.csv")))
    if not paths:
        raise FileNotFoundError(f"No run CSVs found under {res_root}.")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source"] = p
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.lower() == "success"].copy()
    print(f"Collected {len(combined)} successful run(s) across {len(paths)} CSV file(s).")
    return combined


def plot(df: pd.DataFrame, out_dir: Path, y_col: str = "r2_test_mean", logx: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_col = "final_I_compress"
    for col in (x_col, y_col, "beta", "encoder_label"):
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} not in CSV; have: {list(df.columns)}")

    df = df.copy()
    df["_disp"] = df["encoder_label"].map(lambda l: _encoder_display(l)[0])
    df["_color"] = df["encoder_label"].map(lambda l: _encoder_display(l)[1])
    df["_space"] = [
        _space_of(sp, lab)
        for sp, lab in zip(df.get("predictive_space", ""), df["encoder_label"])
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    # One series per (encoder display, predictive space); a series is the beta-ordered curve.
    for (disp, space), sub in df.groupby(["_disp", "_space"]):
        color = sub["_color"].iloc[0]
        agg = (
            sub.groupby("beta", as_index=False)
            .agg(x=(x_col, "mean"), x_sd=(x_col, "std"),
                 y=(y_col, "mean"), y_sd=(y_col, "std"))
            .sort_values("beta")
        )
        agg[["x_sd", "y_sd"]] = agg[["x_sd", "y_sd"]].fillna(0.0)
        is_latent = space == "latent"
        linestyle = ":" if is_latent else "-"
        mfc = "none" if is_latent else color
        suffix = "(L)" if is_latent else "(O)"
        ax.errorbar(
            agg["x"], agg["y"], xerr=agg["x_sd"], yerr=agg["y_sd"],
            marker="o", linestyle=linestyle, color=color,
            markerfacecolor=mfc, markeredgecolor=color,
            linewidth=2, markersize=7, capsize=3, elinewidth=1,
            label=f"{disp} {suffix}",
        )

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("compression complexity  I(X;Z)  (nats)")
    ax.set_ylabel(_Y_LABELS.get(y_col, y_col))
    ax.set_title("Predictivity vs complexity (β sweep)")
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"complexity_curve_{y_col}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {out_path}")
    return out_path


def main() -> None:
    default_root = Path(__file__).resolve().parent / "res" / "poster_headline"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-root", type=Path, default=default_root,
                    help="Directory holding the beta-sweep run CSVs (searched recursively).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the figure (default: --res-root).")
    ap.add_argument("--y-col", type=str, default="r2_test_mean",
                    choices=sorted(_Y_LABELS.keys()),
                    help="Predictivity metric for the y-axis (default: r2_test_mean).")
    ap.add_argument("--logx", action="store_true",
                    help="Log-scale the complexity axis (only valid if all I(X;Z) > 0).")
    args = ap.parse_args()

    out_dir = args.out_dir or args.res_root
    df = collect(args.res_root)

    n_beta = df["beta"].nunique() if "beta" in df.columns else 0
    if n_beta < 2:
        print(f"[WARN] Only {n_beta} beta value(s) present — the complexity curve needs a beta "
              "sweep ([Sweep] betas) to be meaningful.")

    plot(df, out_dir, y_col=args.y_col, logx=args.logx)


if __name__ == "__main__":
    main()
