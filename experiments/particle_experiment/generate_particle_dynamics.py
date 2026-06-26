"""
Spatial particle-dynamics generator for conv encoders (+ MLP).

Design:
- Coherent blob: particles move together on a parametric trajectory mu(t) with per-particle
  diffusion and observation noise. Supported trajectories: "circle", "ellipse".
- Random noise: mean-reverting AR(1) particles that cluster near the origin.
- Output: timeseries (t_max, N*2) ready for CPIC; interleaved particle coordinates
  [x0,y0,x1,y1,...] after sorting particles by x at t=0 (label-agnostic, geometric order).

Ground truth for evaluation only (never passed to CPIC): (t_max, 2) blob centroid
trajectory normalized to [-1, 1] per axis, for R² / alignment evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# -----------------------------------------------------------------------------
# Trajectory factory
# -----------------------------------------------------------------------------

def _make_centroid(t_max, trajectory, omega, orbit_radius, semi_major, semi_minor, rng=None, centroid_ar_coeff=0.95):
    """Return (centroid_x, centroid_y) arrays of shape (t_max,) for the chosen trajectory."""
    t = np.arange(t_max, dtype=np.float64)
    if trajectory == "circle":
        return orbit_radius * np.cos(omega * t), orbit_radius * np.sin(omega * t)
    elif trajectory == "ellipse":
        if semi_major is None or semi_minor is None:
            raise ValueError("semi_major and semi_minor must be specified for trajectory='ellipse'")
        return semi_major * np.cos(omega * t), semi_minor * np.sin(omega * t)
    elif trajectory == "random_walk":
        if rng is None:
            raise ValueError("rng must be provided for trajectory='random_walk'")
        if not (0.0 <= centroid_ar_coeff < 1.0):
            raise ValueError(f"centroid_ar_coeff must be in [0, 1), got {centroid_ar_coeff}")
        sigma_c = orbit_radius * np.sqrt(1 - centroid_ar_coeff ** 2)
        centroid = np.zeros((t_max, 2))
        for t_idx in range(1, t_max):
            centroid[t_idx] = centroid_ar_coeff * centroid[t_idx - 1] + rng.normal(0, sigma_c, 2)
        return centroid[:, 0], centroid[:, 1]
    else:
        raise ValueError(f"Unknown trajectory {trajectory!r}. Choose 'circle', 'ellipse', or 'random_walk'.")


# -----------------------------------------------------------------------------
# Particle simulation
# -----------------------------------------------------------------------------

def _run_blob(t_max, centroid_x, centroid_y, num_particles, sigma_blob, rng):
    """
    Blob positions (t_max, num_particles, 2): centroid[t] + i.i.d. offset each t (no divergence).

    Offset from centroid is redrawn every timestep (add sigma_blob per timestep), so the
    blob stays a fixed-size cloud around the trajectory and does not spread out over time.
    """
    positions = np.zeros((t_max, num_particles, 2))
    for t in range(t_max):
        positions[t] = np.column_stack([centroid_x[t], centroid_y[t]]) + rng.normal(0, sigma_blob, (num_particles, 2))
    return positions


def _run_random_walk(t_max, num_particles, sigma_noise, spatial_bounds, rng, noise_ar_coeff=0.8):
    """
    Positions (t_max, num_particles, 2): init uniform in [-B,B]^2, then AR(1) dynamics.

    AR(1): x_t = alpha * x_{t-1} + eps_t. With alpha=0.8, particles mean-revert toward origin.
    sigma_noise is scaled so the stationary distribution fills [-spatial_bounds, spatial_bounds]^2.
    """
    positions = np.zeros((t_max, num_particles, 2))
    positions[0] = rng.uniform(-spatial_bounds, spatial_bounds, (num_particles, 2))
    if not (0.0 <= noise_ar_coeff < 1.0):
        raise ValueError(f"noise_ar_coeff must be in [0, 1), got {noise_ar_coeff}")
    for t in range(1, t_max):
        positions[t] = noise_ar_coeff * positions[t - 1] + rng.normal(0, sigma_noise, (num_particles, 2))
    return positions


def generate_particle_positions(
    t_max,
    num_blob=30,
    num_noise=50,
    trajectory="circle",
    orbit_radius=3.0,
    semi_major=None,
    semi_minor=None,
    omega=0.05,
    sigma_blob=0.1,
    noise_ar_coeff=0.8,
    centroid_ar_coeff=0.95,
    spatial_bounds=10.0,
    seed=None,
):
    """
    Generate 2D particle positions: coherent blob (parametric trajectory + diffusion) + AR(1) noise.

    Parameters
    ----------
    t_max : int
        Number of time steps.
    num_blob, num_noise : int
        Particle counts for blob and noise.
    trajectory : str
        "circle"  — blob centroid traces r*(cos, sin).
        "ellipse" — blob centroid traces (a*cos, b*sin); requires semi_major and semi_minor.
        "random_walk" — blob centroid traces an AR(1) random walk.
    orbit_radius : float
        Radius for circular trajectory.
    semi_major, semi_minor : float or None
        Semi-axes for ellipse trajectory (a along x, b along y).
    omega : float
        Angular speed (rad/timestep).
    sigma_blob : float
        Per-particle diffusion std around centroid.
    noise_ar_coeff : float
        AR(1) coefficient for noise particles (0 ≤ α < 1).
    centroid_ar_coeff : float
        AR(1) coefficient for the random-walk centroid trajectory (0 ≤ α < 1).
        Stationary std ≈ orbit_radius; ignored for 'circle' and 'ellipse'.
    spatial_bounds : float
        Half-extent of the simulation plane: [-B, B]^2.
    seed : int or None
        Random seed.

    Returns
    -------
    positions : np.ndarray, shape (t_max, num_blob+num_noise, 2)
    centroid_x, centroid_y : np.ndarray, shape (t_max,)
        True blob centroid trajectory (for ground truth latent construction).
    """
    rng = np.random.default_rng(seed)

    centroid_x, centroid_y = _make_centroid(
        t_max, trajectory, omega, orbit_radius, semi_major, semi_minor,
        rng=rng, centroid_ar_coeff=centroid_ar_coeff,
    )
    positions_blob = _run_blob(t_max, centroid_x, centroid_y, num_blob, sigma_blob, rng)

    sigma_noise = spatial_bounds * np.sqrt(1 - noise_ar_coeff ** 2)
    positions_noise = _run_random_walk(t_max, num_noise, sigma_noise, spatial_bounds, rng, noise_ar_coeff)

    positions = np.concatenate([positions_blob, positions_noise], axis=1)
    return positions, centroid_x, centroid_y


# -----------------------------------------------------------------------------
# Representation: interleaved particle coordinates (t_max, N*2)
# -----------------------------------------------------------------------------

def _positions_to_particle_timeseries(positions, seed_positions=None):
    """
    Convert (t_max, N, 2) particle positions into a standardized (t_max, N*2) timeseries.

    Particles are ordered by increasing x-coordinate at t=0 (purely geometric; no use of
    blob/noise labels). Flattening is interleaved: after sorting, particle i occupies
    columns 2*i and 2*i+1 as [x_i, y_i].

    Centering is per-coordinate, but each particle's (x, y) share a single isotropic
    scale (the RMS of their two stds). Independent per-axis scaling would divide out a
    trajectory's aspect ratio — e.g. an axis-aligned ellipse becomes a circle in encoder
    space — so a shared scale is used to preserve trajectory geometry. For a circular
    orbit std_x ≈ std_y, so this matches the old per-axis behavior.
    """
    t_max, N, _ = positions.shape
    if seed_positions is None:
        x0 = positions[0, :, 0]
    else:
        x0 = np.asarray(seed_positions, dtype=np.float64)
        if x0.shape != (N,):
            raise ValueError(f"seed_positions must have shape ({N},), got {x0.shape}")

    particle_order = np.argsort(x0)
    pos_sorted = positions[:, particle_order, :]
    flat = pos_sorted.reshape(t_max, N * 2).astype(np.float64)

    mean = flat.mean(axis=0, keepdims=True)            # (1, N*2) per-coordinate center
    std = flat.std(axis=0, keepdims=True)              # (1, N*2) per-coordinate std

    # Isotropic per-particle scale: share one factor across each particle's (x, y)
    # so axis-aligned shapes (ellipses) keep their aspect ratio in encoder space.
    std_pp = std.reshape(1, N, 2)
    shared = np.sqrt((std_pp ** 2).mean(axis=2, keepdims=True))      # (1, N, 1)
    std = np.broadcast_to(shared, (1, N, 2)).reshape(1, N * 2).copy()
    std[std == 0] = 1.0

    data = np.float32((flat - mean) / std)
    return data, particle_order, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


# -----------------------------------------------------------------------------
# Ground truth latent
# -----------------------------------------------------------------------------

def _ground_truth_from_centroid(centroid_x, centroid_y):
    """
    Normalize the blob centroid trajectory to [-1, 1] using a single shared scale,
    preserving the aspect ratio of the trajectory.

    Works for any trajectory shape — circle, ellipse, or arbitrary path.
    """
    gt = np.column_stack([centroid_x, centroid_y]).astype(np.float32)
    scale = max(float(np.abs(gt).max()), 1e-8)
    return gt / scale


# -----------------------------------------------------------------------------
# Main timeseries generator
# -----------------------------------------------------------------------------

def generate_particle_process_timeseries(
    t_max=200,
    num_blob=30,
    num_noise=50,
    trajectory="circle",
    orbit_radius=3.0,
    semi_major=None,
    semi_minor=None,
    omega=0.05,
    sigma_blob=0.1,
    noise_ar_coeff=0.8,
    centroid_ar_coeff=0.95,
    spatial_bounds=10.0,
    seed=None,
):
    """
    Generate particle-dynamics timeseries ready for CPIC.

    1) Simulates 2D particles (coherent blob on parametric trajectory + AR(1) noise).
    2) Converts positions to interleaved, x-sorted, standardized coordinates.
    3) Derives ground truth latent from the blob centroid (normalized to [-1, 1]).

    Returns
    -------
    data : np.ndarray, float32, shape (t_max, N*2)
    ground_truth_latent : np.ndarray, float32, shape (t_max, 2)
        Normalized blob centroid trajectory for R² evaluation.
    particle_order : np.ndarray, shape (N,)
    particle_labels : np.ndarray, shape (N,), int8
        1 = blob, 0 = noise, in original simulation order.
    standardization_mean : np.ndarray, shape (N*2,), float32
    standardization_std : np.ndarray, shape (N*2,), float32
    positions : np.ndarray, shape (t_max, N, 2), float64
        Raw particle positions in original simulation order.
    """
    positions, centroid_x, centroid_y = generate_particle_positions(
        t_max=t_max,
        num_blob=num_blob, num_noise=num_noise,
        trajectory=trajectory,
        orbit_radius=orbit_radius,
        semi_major=semi_major, semi_minor=semi_minor,
        omega=omega, sigma_blob=sigma_blob,
        noise_ar_coeff=noise_ar_coeff,
        centroid_ar_coeff=centroid_ar_coeff,
        spatial_bounds=spatial_bounds,
        seed=seed,
    )
    N = num_blob + num_noise
    particle_labels = np.zeros(N, dtype=np.int8)
    particle_labels[:num_blob] = 1

    data, particle_order, standardization_mean, standardization_std = _positions_to_particle_timeseries(positions)
    ground_truth_latent = _ground_truth_from_centroid(centroid_x, centroid_y)

    return (
        data, ground_truth_latent, particle_order, particle_labels,
        standardization_mean, standardization_std, positions,
    )


# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------

def animate_particle_dynamics(
    t_max=500,
    num_blob=30,
    num_noise=50,
    trajectory="circle",
    orbit_radius=3.0,
    semi_major=None,
    semi_minor=None,
    omega=0.05,
    sigma_blob=0.1,
    noise_ar_coeff=0.8,
    centroid_ar_coeff=0.95,
    spatial_bounds=10.0,
    seed=42,
    interval=50,
):
    """Animate the particle dynamics: blob (orange) + noise (gray)."""
    positions, _, _ = generate_particle_positions(
        t_max=t_max, num_blob=num_blob, num_noise=num_noise,
        trajectory=trajectory, orbit_radius=orbit_radius,
        semi_major=semi_major, semi_minor=semi_minor,
        omega=omega, sigma_blob=sigma_blob,
        noise_ar_coeff=noise_ar_coeff, centroid_ar_coeff=centroid_ar_coeff,
        spatial_bounds=spatial_bounds, seed=seed,
    )

    particle_labels = np.zeros(num_blob + num_noise, dtype=np.int8)
    particle_labels[:num_blob] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-spatial_bounds, spatial_bounds)
    ax.set_ylim(-spatial_bounds, spatial_bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    colors = np.where(particle_labels, "C1", "gray")
    scat = ax.scatter(
        positions[0, :, 0], positions[0, :, 1],
        s=40, c=colors, alpha=0.85, edgecolors="k", linewidths=0.3,
    )

    def init():
        scat.set_offsets(positions[0])
        return (scat,)

    def update(frame):
        scat.set_offsets(positions[frame])
        ax.set_title(f"t={frame}  trajectory={trajectory}  (orange=blob, gray=noise)")
        return (scat,)

    return animation.FuncAnimation(fig, update, frames=t_max, init_func=init, interval=interval, blit=False)


# -----------------------------------------------------------------------------
# 3D verification plot (x, y, t)
# -----------------------------------------------------------------------------

def plot_verification_3d(positions, num_blob, num_noise, path=None):
    """3D (x, y, t) trajectory plot: blob (orange) and noise (gray)."""
    t_max, N, _ = positions.shape
    t_axis = np.arange(t_max)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        positions[:, :num_blob, 0].ravel(order="F"),
        positions[:, :num_blob, 1].ravel(order="F"),
        np.repeat(t_axis, num_blob),
        c="C1", s=2, alpha=0.6, label="coherent blob",
    )
    ax.scatter(
        positions[:, num_blob:, 0].ravel(order="F"),
        positions[:, num_blob:, 1].ravel(order="F"),
        np.repeat(t_axis, num_noise),
        c="gray", s=2, alpha=0.2, label="random noise",
    )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
    ax.legend()
    fig.subplots_adjust(left=0, right=0.86, bottom=0.07, top=0.95)

    if path is not None:
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return None
    return fig, ax


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    ani = animate_particle_dynamics(
        t_max=500, num_blob=30, num_noise=80,
        trajectory="circle", orbit_radius=3.0, semi_major=2.5, semi_minor=1.5, centroid_ar_coeff=0.95,
        omega=0.05, sigma_blob=0.1,
        noise_ar_coeff=0.8, spatial_bounds=10.0,
        seed=42, interval=50,
    )
    ani.save(os.path.join(_dir, "particle_circle_process.gif"), fps=20, writer="pillow")
    plt.show()
