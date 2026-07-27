"""Probing-result visualizations: position-trajectory overlay + velocity field.

The linear probe fit on the CPIC latents produces two kinds of decoded output that
are worth seeing spatially, not just as scalar R2:

  * POSITION trajectory — ground-truth blob-centroid orbit vs the probe's predicted
    orbit (the overlay used in ``particle_dynamics.ipynb``'s ``evaluate_probe_train_test``).
  * VELOCITY field — the first-order dynamics. At each ground-truth orbit position we
    draw the true velocity vector (blue) and the probe-predicted velocity vector (red)
    as a quiver, so one can see whether the latent recovers the local flow (heading +
    speed) around the orbit rather than just static position.

These functions take already-computed arrays (positions, true/predicted velocities) so
they can be called both from ``run_particle_experiment.py`` (opt-in ``save_probe_viz``)
and imported into a notebook. ``save_probe_viz_artifacts`` also dumps a small ``.npz``
so the figure can be re-rendered / restyled without retraining the model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GT_COLOR = "#1f5fbf"   # blue — ground truth
PRED_COLOR = "#c0392b"  # red — probe prediction


def plot_trajectory_probe(ax, gt_pos: np.ndarray, pred_pos: np.ndarray, r2: float, title: str) -> None:
    """Overlay the ground-truth vs probe-predicted centroid trajectory on ``ax``."""
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], "--", color=GT_COLOR, linewidth=2, label="ground truth")
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], "-", color=PRED_COLOR, linewidth=1.5,
            alpha=0.9, label=f"predicted (R²={r2:.2f})")
    ax.set_title(title)
    ax.set_xlabel("centroid x")
    ax.set_ylabel("centroid y")
    ax.axis("equal")
    ax.legend(loc="upper right", fontsize=8)


def plot_velocity_field_probe(
    ax,
    gt_pos: np.ndarray,
    vel_true: np.ndarray,
    vel_pred: np.ndarray,
    r2: float,
    title: str,
    max_arrows: int = 60,
) -> None:
    """Quiver of true (blue) vs predicted (red) velocity vectors along the orbit.

    Arrows are anchored at the ground-truth positions ``gt_pos`` and drawn in data
    coordinates with a single shared amplification factor (chosen from the true-speed
    median) so the two fields are directly comparable. The path is subsampled to at
    most ``max_arrows`` anchors to keep the field legible.
    """
    n = len(gt_pos)
    stride = max(1, n // max_arrows)
    p = gt_pos[::stride]
    vt = vel_true[::stride]
    vp = vel_pred[::stride]

    # Shared amplification: make a typical true arrow span ~12% of the plot extent.
    extent = float(np.ptp(gt_pos, axis=0).max()) or 1.0
    med_speed = float(np.median(np.linalg.norm(vt, axis=1))) or 1e-9
    amp = 0.12 * extent / med_speed

    ax.plot(gt_pos[:, 0], gt_pos[:, 1], "-", color="0.8", linewidth=1, zorder=0)
    qk = dict(angles="xy", scale_units="xy", scale=1.0 / amp, width=0.005)
    ax.quiver(p[:, 0], p[:, 1], vt[:, 0], vt[:, 1], color=GT_COLOR, label="true velocity", **qk)
    ax.quiver(p[:, 0], p[:, 1], vp[:, 0], vp[:, 1], color=PRED_COLOR,
              label=f"predicted (R²={r2:.2f})", **qk)
    ax.set_title(title)
    ax.set_xlabel("centroid x")
    ax.set_ylabel("centroid y")
    ax.axis("equal")
    ax.legend(loc="upper right", fontsize=8)


def save_probe_viz_artifacts(
    out_dir: Path,
    *,
    seed: int,
    num_noise: int,
    num_blob: int,
    encoder_label: str,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    vel_true: np.ndarray,
    vel_pred: np.ndarray,
    r2_pos: float,
    r2_vel: float,
) -> dict[str, str]:
    """Render the 2-panel probe figure (position overlay + velocity field) and dump a
    companion ``.npz`` so it can be re-rendered without retraining. All arrays are the
    held-out TEST windows. ``encoder_label`` already carries the predictive-space tag
    (e.g. ``"FeatureMaskMLP (O)"``), so it is used verbatim. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{encoder_label}_seed{seed}_nn{num_noise}_nb{num_blob}".replace(" ", "")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_trajectory_probe(axes[0], gt_pos, pred_pos, r2_pos,
                          f"Position probe — {encoder_label}")
    plot_velocity_field_probe(axes[1], gt_pos, vel_true, vel_pred, r2_vel,
                              f"Velocity field probe — {encoder_label}")
    fig.suptitle(
        f"Held-out linear probe  |  num_blob={num_blob}, num_noise={num_noise}, seed={seed}",
        y=1.02,
    )
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path, gt_pos=gt_pos, pred_pos=pred_pos, vel_true=vel_true, vel_pred=vel_pred,
        r2_pos=np.float32(r2_pos), r2_vel=np.float32(r2_vel),
    )
    return {"probe_plot_path": str(png_path), "probe_npz_path": str(npz_path)}
