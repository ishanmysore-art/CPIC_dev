#!/usr/bin/env python3
"""
Render static learning-curve figures from the Step-1 convergence run's
TensorBoard event files (config ``particle_convergence``).

Reads ``<res>/convergence/tensorboard/seed*/noise*/<Encoder>/attempt1`` and writes
individual PNGs to ``<res>/convergence/``:

  * convergence_probe_r2.png   -- held-out linear-probe R2 vs epoch (all runs)
  * convergence_loss.png       -- training objective vs epoch (all runs)
  * convergence_gates_featuremaskmlp.png -- FeatureMaskMLP blob vs noise gate probs

Styling follows the presentation convention: MLP gray, ConvPhysical blue,
FeatureMaskMLP green; observation = solid + filled marker, latent = dotted +
hollow marker; legend outside the axes.

Usage
-----
python plot_convergence_curves.py [--res-dir experiments/particle_experiment/res]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Display-name + style maps keyed by the sanitized TensorBoard subdir name.
FAMILY_COLOR = {"MLP": "#7f7f7f", "ConvPhysical": "#1f77b4", "MaskUniformLearned": "#2ca02c"}
FAMILY_DISPLAY = {"MLP": "MLP", "ConvPhysical": "ConvPhysical", "MaskUniformLearned": "FeatureMaskMLP"}


def parse_run_dirname(name: str) -> tuple[str, str]:
    """Split a sanitized run dir like 'ConvPhysical_O' into (family, space)."""
    family, _, suffix = name.rpartition("_")
    return family, suffix  # suffix in {"L", "O"}


def style_for(space: str) -> dict:
    """Line/marker style for latent (L, dotted/hollow) vs observation (O, solid/filled)."""
    if space == "O":
        return {"linestyle": "-", "fill": True, "marker": "o"}
    return {"linestyle": ":", "fill": False, "marker": "o"}


def load_runs(tb_root: Path) -> dict[str, EventAccumulator]:
    runs: dict[str, EventAccumulator] = {}
    for d in sorted(glob.glob(str(tb_root / "seed*" / "noise*" / "*" / "attempt*"))):
        label = Path(d).parents[0].name
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        runs[label] = ea
    return runs


def curve(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])


def _display_label(name: str) -> str:
    family, space = parse_run_dirname(name)
    return f"{FAMILY_DISPLAY.get(family, family)} ({space})"


def _plot_scalar(runs, tag, ylabel, title, out_path, hline=None, symlog="auto",
                 figsize=(7.5, 4.5), linthresh=1.0, linscale=1.0):
    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    plotted = 0
    maxabs = 0.0
    for name in sorted(runs):
        family, space = parse_run_dirname(name)
        if tag not in runs[name].Tags()["scalars"]:
            continue
        st, v = curve(runs[name], tag)
        plotted += 1
        finite = v[np.isfinite(v)]
        if finite.size:
            maxabs = max(maxabs, float(np.abs(finite).max()))
        c = FAMILY_COLOR.get(family, "black")
        sty = style_for(space)
        ax.plot(st, v, color=c, linestyle=sty["linestyle"], linewidth=2,
                marker=sty["marker"], markevery=max(1, len(st) // 12), markersize=6,
                markerfacecolor=(c if sty["fill"] else "white"), markeredgecolor=c)
        handles.append(Line2D([0], [0], color=c, linestyle=sty["linestyle"], marker="o",
                              markersize=6, linewidth=2,
                              markerfacecolor=(c if sty["fill"] else "white"),
                              markeredgecolor=c, label=_display_label(name)))
    if plotted == 0:
        plt.close(fig)
        print(f"Skipped {out_path} (no runs log '{tag}')")
        return
    # Detonating runs (e.g. FeatureMaskMLP I_compress under infonce_lower) span many
    # orders of magnitude -> symlog keeps the bounded curves readable while still
    # showing the blow-up. "auto" only switches when something actually blows up, so
    # bounded vub / infonce_upper figures stay on a plain linear axis.
    use_symlog = symlog is True or (symlog == "auto" and maxabs > 100.0)
    if use_symlog:
        # linscale > 1 widens the linear band around 0 so the "0" tick stops
        # colliding with the +/-10^0 log ticks (the crammed-symlog problem when a
        # curve spans ~20+ decades). linthresh sets where the log region begins.
        ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_gates(runs, out_path):
    """FeatureMaskMLP blob vs noise mean-gate probability over epochs (L and O)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    handles = []
    green = FAMILY_COLOR["MaskUniformLearned"]
    for name in [n for n in sorted(runs) if n.startswith("MaskUniformLearned")]:
        _, space = parse_run_dirname(name)
        sty = style_for(space)
        if "epoch/gate/mean_blob" not in runs[name].Tags()["scalars"]:
            continue
        sb, vb = curve(runs[name], "epoch/gate/mean_blob")
        sn, vn = curve(runs[name], "epoch/gate/mean_noise")
        # blob = green, noise = gray, line style encodes latent/observation.
        ax.plot(sb, vb, color=green, linestyle=sty["linestyle"], linewidth=2)
        ax.plot(sn, vn, color="#7f7f7f", linestyle=sty["linestyle"], linewidth=2)
        handles.append(Line2D([0], [0], color=green, linestyle=sty["linestyle"],
                              linewidth=2, label=f"blob gate ({space})"))
        handles.append(Line2D([0], [0], color="#7f7f7f", linestyle=sty["linestyle"],
                              linewidth=2, label=f"noise gate ({space})"))
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Mean gate probability", fontsize=13)
    ax.set_title("FeatureMaskMLP gate learning curves", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="experiments/particle_experiment/res")
    ap.add_argument("--subdir", default="convergence",
                    help="convergence-run subdirectory under --res-dir")
    args = ap.parse_args()

    conv_dir = Path(args.res_dir) / args.subdir
    tb_root = conv_dir / "tensorboard"
    runs = load_runs(tb_root)
    if not runs:
        raise SystemExit(f"No runs found under {tb_root}")

    _plot_scalar(runs, "epoch/probe_r2/test_mean",
                 r"Held-out linear-probe $R^2$", r"Predictivity vs epoch",
                 conv_dir / "convergence_probe_r2.png", hline=0.0)
    # Taller figure + wider linear band: the bounded MLP/ConvPhysical curves sit near 0
    # while FeatureMaskMLP plunges ~20 decades, so the default height crams every symlog
    # decade together. More vertical room + linscale spreads them out.
    _plot_scalar(runs, "epoch/loss/mean",
                 "Training loss (objective)", "Loss vs epoch",
                 conv_dir / "convergence_loss.png",
                 figsize=(7.5, 6.5), linscale=1.6)
    # Rate-estimate curve: the direct "FeatureMaskMLP is diverging" signal. Under
    # infonce_lower the mask rate detonates (auto-symlog) while MLP/ConvPhysical stay
    # bounded; under vub / infonce_upper it stays bounded for everyone.
    _plot_scalar(runs, "epoch/I_compress/mean",
                 "I_compress  (rate estimate)", "Rate estimate vs epoch",
                 conv_dir / "convergence_i_compress.png", hline=0.0,
                 figsize=(7.5, 6.5), linscale=2.0)
    plot_gates(runs, conv_dir / "convergence_gates_featuremaskmlp.png")
