from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _pad_filter_magnitudes(w_mag: np.ndarray, gap_rows: int = 1):
    """Pad filter magnitudes with NaN rows for visual separation."""
    c_out, k_h = w_mag.shape
    pad_h = c_out + (c_out - 1) * gap_rows
    pad = np.full((pad_h, k_h), np.nan, dtype=np.float64)
    for i in range(c_out):
        pad[i * (1 + gap_rows), :] = w_mag[i, :]
    return pad, c_out, k_h, gap_rows


def filter_weights_to_panel_magnitudes(weights: np.ndarray) -> np.ndarray:
    """
    Convert Conv2d-like weights into per-filter 1D magnitudes for panel plotting.

    Expected shapes:
    - conv_spatial: (C_out, C_in, K_h, 1) -> abs(weights[:, 0, :, 0])
    - conv_particle2d: (C_out, C_in, K_particles, K_coords) -> sum abs over coord dim
    """
    if weights.ndim != 4:
        raise ValueError(f"Expected 4D conv weights, got shape {weights.shape}")
    if weights.shape[1] < 1:
        raise ValueError(f"Expected at least one input channel, got shape {weights.shape}")

    w_mag = np.abs(weights[:, 0, :, :])
    if weights.shape[-1] == 1:
        return w_mag[:, :, 0]
    return w_mag.sum(axis=2)


def plot_filter_heatmap_panels(
    w_mag: np.ndarray,
    gap_rows: int = 1,
    filters_per_panel: int = 12,
    suptitle: str | None = None,
    cell_inches: float = 0.28,
):
    """
    Plot filter magnitude heatmaps in panel chunks.
    """
    c_out, k_h = w_mag.shape
    chunks = []
    for start in range(0, c_out, filters_per_panel):
        end = min(start + filters_per_panel, c_out)
        sub = w_mag[start:end, :]
        pad, c_sub, kh, g = _pad_filter_magnitudes(sub, gap_rows=gap_rows)
        chunks.append((pad, start, end, c_sub, kh, g))

    n_panels = len(chunks)
    max_pad_h = max(pad.shape[0] for pad, *_ in chunks)
    fig_w = n_panels * (k_h * cell_inches) + max(0, n_panels - 1)
    fig_h = max_pad_h * cell_inches
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = np.array([axes])

    ims = []
    for ax, (pad, start, end, c_sub, kh, g) in zip(axes.flat, chunks):
        im = ax.imshow(pad, aspect="equal", cmap="magma")
        ims.append(im)
        ax.set_xticks(np.arange(kh))
        ax.set_xticklabels([str(i) for i in range(kh)])
        ax.set_yticks([i * (1 + g) for i in range(c_sub)])
        ax.set_yticklabels([str(start + i) for i in range(c_sub)])
        ax.set_title(f"filters {start}-{end - 1}", fontsize=10)

    axes.flat[0].set_ylabel("Output channel")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("|weight|")
    return fig, axes


# ---------------------------------------------------------------------------
# ConvPhysicalEncoder: 2D filter heatmaps and trajectory path overlay
# ---------------------------------------------------------------------------

def plot_physical_filter_heatmaps(
    weights: np.ndarray,
    n_cols: int = 8,
    suptitle: str | None = None,
) -> tuple:
    """
    Plot ConvPhysicalEncoder filter weights as 2D spatial heatmaps.

    Parameters
    ----------
    weights : np.ndarray, shape (C_out, 1, K, K)
        Conv2d filter weights from get_filters().
    n_cols : int
        Panels per row.
    suptitle : str, optional
        Figure suptitle.

    Returns
    -------
    fig, axes
    """
    if weights.ndim != 4 or weights.shape[1] != 1:
        raise ValueError(f"Expected weights shape (C_out, 1, K, K), got {weights.shape}")

    C_out = weights.shape[0]
    n_rows = max(1, (C_out + n_cols - 1) // n_cols)

    # Extra width reserves space for the shared colorbar (avoids overlap on right columns).
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.55 + 0.9, n_rows * 1.55),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    vmax = float(np.abs(weights[:, 0]).max()) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = None
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        if idx < C_out:
            im = ax.imshow(weights[idx, 0], cmap="RdBu_r", norm=norm, aspect="equal", origin="lower")
            ax.set_title(f"f{idx}", fontsize=7)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_visible(False)

    if im is not None:
        fig.colorbar(
            im,
            ax=axes,
            location="right",
            shrink=0.85,
            aspect=25,
            label="weight",
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    return fig, axes


def compute_trajectory_in_grid_coords(
    positions: np.ndarray,
    particle_labels: np.ndarray,
    particle_order: np.ndarray,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    grid_size: int,
    spatial_bounds: float,
) -> np.ndarray:
    """
    Compute the blob centroid path in the encoder's 2D grid coordinate space.

    Applies the exact per-feature standardization from training to the blob
    particle positions, then maps to grid coordinates [0, grid_size].

    Parameters
    ----------
    positions : np.ndarray, shape (t_max, N, 2)
        Raw particle positions in original simulation order.
    particle_labels : np.ndarray, shape (N,)
        1 = blob, 0 = noise, in simulation order.
    particle_order : np.ndarray, shape (N,)
        Sort-by-x permutation from _positions_to_particle_timeseries.
    standardization_mean : np.ndarray, shape (N*2,)
        Per-feature mean used during standardization.
    standardization_std : np.ndarray, shape (N*2,)
        Per-feature std used during standardization.
    grid_size : int
        Encoder grid resolution.
    spatial_bounds : float
        Encoder grid range (standardized units).

    Returns
    -------
    trajectory_grid : np.ndarray, shape (t_max, 2)
        Blob centroid path in grid coordinates [0, grid_size].
    """
    t_max, N, _ = positions.shape

    # Sort positions by particle_order to align with standardized feature columns
    pos_sorted = positions[:, particle_order, :]  # (t_max, N, 2)

    # Standardize each particle coordinate using stored per-feature arrays
    # Feature column 2*i = x of sorted particle i; 2*i+1 = y of sorted particle i
    pos_std = np.zeros_like(pos_sorted, dtype=np.float32)
    for i in range(N):
        pos_std[:, i, 0] = (pos_sorted[:, i, 0] - standardization_mean[2 * i]) / standardization_std[2 * i]
        pos_std[:, i, 1] = (pos_sorted[:, i, 1] - standardization_mean[2 * i + 1]) / standardization_std[2 * i + 1]

    # Identify blob particles in sorted order
    sorted_labels = particle_labels[particle_order]
    blob_sorted_idx = np.where(sorted_labels == 1)[0]

    # Centroid of blob particles in standardized space
    centroid_std = pos_std[:, blob_sorted_idx, :].mean(axis=1)  # (t_max, 2)

    # Map standardized coords to grid coords [0, grid_size]
    scale = grid_size / (2.0 * spatial_bounds)
    trajectory_grid = (centroid_std + spatial_bounds) * scale
    trajectory_grid = np.clip(trajectory_grid, 0.0, grid_size - 1)
    return trajectory_grid


def compute_tangential_arrows(
    trajectory_grid: np.ndarray,
    n_arrows: int = 12,
    arrow_length: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample evenly-spaced tangential velocity arrows along the trajectory path.

    Parameters
    ----------
    trajectory_grid : np.ndarray, shape (t_max, 2)
        Trajectory path in grid coordinates.
    n_arrows : int
        Number of arrows to place.
    arrow_length : float
        Display length of each arrow in grid units.

    Returns
    -------
    positions : np.ndarray, shape (n_arrows, 2) — arrow base positions
    directions : np.ndarray, shape (n_arrows, 2) — unit tangent vectors scaled to arrow_length
    """
    t_max = len(trajectory_grid)
    indices = np.linspace(0, t_max - 1, n_arrows, dtype=int)
    positions = trajectory_grid[indices]

    # Central-difference tangent; wrap around endpoints
    prev_idx = np.clip(indices - 1, 0, t_max - 1)
    next_idx = np.clip(indices + 1, 0, t_max - 1)
    tangents = trajectory_grid[next_idx] - trajectory_grid[prev_idx]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = (tangents / norms) * arrow_length
    return positions, directions


def overlay_trajectory_on_ax(
    ax,
    trajectory_grid: np.ndarray,
    arrows: tuple[np.ndarray, np.ndarray] | None = None,
    trajectory_color: str = "lime",
    arrow_color: str = "white",
) -> None:
    """
    Draw the blob trajectory path and optional tangential arrows onto an existing axis.

    Coordinates must already be in the same units as the axis (grid cells).
    """
    ax.plot(trajectory_grid[:, 0], trajectory_grid[:, 1], color=trajectory_color, linewidth=1.0, alpha=0.85, zorder=5)
    if arrows is not None:
        pos, dirs = arrows
        ax.quiver(
            pos[:, 0], pos[:, 1], dirs[:, 0], dirs[:, 1],
            color=arrow_color, scale=1.0, scale_units="xy", angles="xy",
            width=0.006, headwidth=4, headlength=4, zorder=6,
        )


def compute_avg_density_grid(
    data: np.ndarray,
    grid_size: int,
    spatial_bounds: float,
) -> np.ndarray:
    """
    Compute the time-averaged particle density grid from standardized timeseries data.

    Parameters
    ----------
    data : np.ndarray, shape (t_max, N*2)
        Standardized interleaved particle coordinates.
    grid_size : int
        Grid resolution.
    spatial_bounds : float
        Coordinate range for binning (standardized units).

    Returns
    -------
    avg_density : np.ndarray, shape (grid_size, grid_size), float32
        Mean particle count per cell over time.
    """
    t_max, feat_dim = data.shape
    N = feat_dim // 2
    scale = grid_size / (2.0 * spatial_bounds)

    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    for t in range(t_max):
        coords = data[t].reshape(N, 2)
        xs = np.clip(((coords[:, 0] + spatial_bounds) * scale).astype(int), 0, grid_size - 1)
        ys = np.clip(((coords[:, 1] + spatial_bounds) * scale).astype(int), 0, grid_size - 1)
        np.add.at(density, (ys, xs), 1.0)
    density /= t_max
    return density


# -----------------------------------------------------------------------------
# Trajectory-filter alignment metric (used by the notebook interpretability analysis)
# -----------------------------------------------------------------------------

def trajectory_filter_alignment(weights, trajectory_grid, avg_density, grid_size):
    """
    Per-filter trajectory selectivity: fraction of the filter's response on the density
    map that lands on the trajectory path.

    Applies each |filter| as a spatial detector via conv2d on avg_density, then measures
    how much of the resulting response map overlaps the trajectory path mask. NOTE: this
    metric only measures response *overlap* with the path — it is structurally blind to
    filter *shape* (an isotropic blob overlaps a circle and an ellipse about equally).
    """
    traj_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    oi = np.clip(trajectory_grid[:, 1].astype(int), 0, grid_size - 1)
    oj = np.clip(trajectory_grid[:, 0].astype(int), 0, grid_size - 1)
    traj_mask[oi, oj] = 1.0
    traj_mask_norm = traj_mask / (traj_mask.sum() + 1e-8)

    K = weights.shape[2]
    pad = K // 2
    density_t = torch.from_numpy(avg_density).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    scores = []
    for c in range(weights.shape[0]):
        kernel = torch.from_numpy(np.abs(weights[c:c + 1])).float()  # (1,1,K,K)
        response = F.conv2d(density_t, kernel, padding=pad).squeeze().numpy()  # (H,W)
        response_norm = response / (response.sum() + 1e-8)
        scores.append(float(np.sum(response_norm * traj_mask_norm)))
    return np.array(scores)


def plot_trajectory_density_map(
    avg_density: np.ndarray,
    trajectory_grid: np.ndarray,
    arrows: tuple[np.ndarray, np.ndarray] | None = None,
    spatial_bounds: float = 3.0,
    suptitle: str | None = None,
) -> tuple:
    """
    Plot average particle density with the blob trajectory path and velocity arrows overlaid.

    Parameters
    ----------
    avg_density : np.ndarray, shape (grid_size, grid_size)
        Mean particle count per cell.
    trajectory_grid : np.ndarray, shape (t_max, 2)
        Trajectory path in grid coordinates.
    arrows : tuple, optional
        (positions, directions) from compute_tangential_arrows.
    spatial_bounds : float
        Used for axis tick labeling in standardized units.
    suptitle : str, optional

    Returns
    -------
    fig, ax
    """
    grid_size = avg_density.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(avg_density, cmap="magma", origin="lower", aspect="equal",
                   extent=[0, grid_size, 0, grid_size])
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean particle count")

    overlay_trajectory_on_ax(ax, trajectory_grid, arrows=arrows)

    tick_positions = np.linspace(0, grid_size, 5)
    tick_labels = [f"{v:.1f}" for v in np.linspace(-spatial_bounds, spatial_bounds, 5)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("x (std units)")
    ax.set_ylabel("y (std units)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig, ax


def save_encoder_filter_artifacts(
    weights: np.ndarray,
    meta: dict,
    *,
    out_dir: Path | str,
    stem: str,
    encoder_type: str,
    encoder_label: str,
    layer_idx: int = 0,
    trajectory_grid: np.ndarray | None = None,
    avg_density: np.ndarray | None = None,
) -> dict[str, str]:
    """
    Save learned conv filter weights and visualization PNGs for a sweep run.

    For ``conv_physical``, writes 2D filter heatmaps and an optional trajectory+density
    overlay. For other conv encoders, writes panel magnitude heatmaps.

    Returns
    -------
    dict
        Keys: ``filter_weights_path``, ``filter_plot_path``, ``filter_trajectory_density_path``
        (empty when not applicable), ``filter_meta_path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{stem}.npy"
    png_path = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}.json"
    trajectory_density_path = ""

    np.save(npy_path, weights)

    suptitle = f"{encoder_label} layer {layer_idx}"
    if encoder_type == "conv_physical":
        fig, _ = plot_physical_filter_heatmaps(weights, n_cols=8, suptitle=suptitle)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if trajectory_grid is not None and avg_density is not None:
            trajectory_density_path = str(out_dir / f"{stem}_trajectory_density.png")
            arrows = compute_tangential_arrows(trajectory_grid)
            fig2, _ = plot_trajectory_density_map(
                avg_density,
                trajectory_grid,
                arrows=arrows,
                spatial_bounds=float(meta.get("spatial_bounds", 3.0)),
                suptitle=f"{encoder_label} — trajectory + density",
            )
            fig2.savefig(trajectory_density_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            np.save(out_dir / f"{stem}_trajectory_grid.npy", trajectory_grid)
            np.save(out_dir / f"{stem}_avg_density.npy", avg_density)
    else:
        w_mag = filter_weights_to_panel_magnitudes(weights)
        fig, _ = plot_filter_heatmap_panels(
            w_mag,
            gap_rows=1,
            filters_per_panel=8,
            suptitle=suptitle,
        )
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    payload = {
        "encoder_type": encoder_type,
        "encoder_label": encoder_label,
        "layer_idx": layer_idx,
        "shape": list(weights.shape),
        "meta": meta,
        "filter_weights_path": str(npy_path),
        "filter_plot_path": str(png_path),
        "filter_trajectory_density_path": trajectory_density_path,
    }
    if trajectory_grid is not None:
        payload["trajectory_grid_path"] = str(out_dir / f"{stem}_trajectory_grid.npy")
    if avg_density is not None:
        payload["avg_density_path"] = str(out_dir / f"{stem}_avg_density.npy")

    with meta_path.open("w") as f:
        json.dump(payload, f, indent=2)

    payload["filter_meta_path"] = str(meta_path)
    return payload


def regenerate_physical_filter_plots(
    weights_path: Path | str,
    meta_path: Path | str | None = None,
    *,
    out_dir: Path | str | None = None,
    stem: str | None = None,
) -> dict[str, str]:
    """Rebuild ConvPhysical filter PNGs from saved ``.npy`` / ``.json`` artifacts."""
    weights_path = Path(weights_path)
    out_dir = Path(out_dir or weights_path.parent)
    stem = stem or weights_path.stem

    weights = np.load(weights_path)
    meta: dict = {}
    encoder_label = "ConvPhysical"
    layer_idx = 0
    if meta_path is not None and Path(meta_path).exists():
        with Path(meta_path).open() as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        encoder_label = payload.get("encoder_label", encoder_label)
        layer_idx = int(payload.get("layer_idx", 0))

    trajectory_grid_path = out_dir / f"{stem}_trajectory_grid.npy"
    avg_density_path = out_dir / f"{stem}_avg_density.npy"
    trajectory_grid = np.load(trajectory_grid_path) if trajectory_grid_path.exists() else None
    avg_density = np.load(avg_density_path) if avg_density_path.exists() else None

    return save_encoder_filter_artifacts(
        weights,
        meta,
        out_dir=out_dir,
        stem=stem,
        encoder_type="conv_physical",
        encoder_label=encoder_label,
        layer_idx=layer_idx,
        trajectory_grid=trajectory_grid,
        avg_density=avg_density,
    )
