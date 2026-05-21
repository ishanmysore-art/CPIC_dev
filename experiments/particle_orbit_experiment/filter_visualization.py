from __future__ import annotations

import numpy as np
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

    This mirrors the notebook visualization style used in particle_orbit.ipynb.
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
# ConvPhysicalEncoder: 2D filter heatmaps and orbit vector field overlay
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
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
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, label="weight")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig, axes


def compute_orbit_in_grid_coords(
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
    orbit_grid : np.ndarray, shape (t_max, 2)
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
    orbit_grid = (centroid_std + spatial_bounds) * scale
    orbit_grid = np.clip(orbit_grid, 0.0, grid_size - 1)
    return orbit_grid


def compute_tangential_arrows(
    orbit_grid: np.ndarray,
    n_arrows: int = 12,
    arrow_length: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample evenly-spaced tangential velocity arrows along the orbit path.

    Parameters
    ----------
    orbit_grid : np.ndarray, shape (t_max, 2)
        Orbit path in grid coordinates.
    n_arrows : int
        Number of arrows to place.
    arrow_length : float
        Display length of each arrow in grid units.

    Returns
    -------
    positions : np.ndarray, shape (n_arrows, 2) — arrow base positions
    directions : np.ndarray, shape (n_arrows, 2) — unit tangent vectors scaled to arrow_length
    """
    t_max = len(orbit_grid)
    indices = np.linspace(0, t_max - 1, n_arrows, dtype=int)
    positions = orbit_grid[indices]

    # Central-difference tangent; wrap around endpoints
    prev_idx = np.clip(indices - 1, 0, t_max - 1)
    next_idx = np.clip(indices + 1, 0, t_max - 1)
    tangents = orbit_grid[next_idx] - orbit_grid[prev_idx]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = (tangents / norms) * arrow_length
    return positions, directions


def overlay_orbit_on_ax(
    ax,
    orbit_grid: np.ndarray,
    arrows: tuple[np.ndarray, np.ndarray] | None = None,
    orbit_color: str = "lime",
    arrow_color: str = "white",
) -> None:
    """
    Draw the blob orbit path and optional tangential arrows onto an existing axis.

    Coordinates must already be in the same units as the axis (grid cells).
    """
    ax.plot(orbit_grid[:, 0], orbit_grid[:, 1], color=orbit_color, linewidth=1.0, alpha=0.85, zorder=5)
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


def plot_orbit_density_map(
    avg_density: np.ndarray,
    orbit_grid: np.ndarray,
    arrows: tuple[np.ndarray, np.ndarray] | None = None,
    spatial_bounds: float = 3.0,
    suptitle: str | None = None,
) -> tuple:
    """
    Plot average particle density with the blob orbit path and velocity arrows overlaid.

    Parameters
    ----------
    avg_density : np.ndarray, shape (grid_size, grid_size)
        Mean particle count per cell.
    orbit_grid : np.ndarray, shape (t_max, 2)
        Orbit path in grid coordinates.
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

    overlay_orbit_on_ax(ax, orbit_grid, arrows=arrows)

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
