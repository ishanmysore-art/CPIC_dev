"""
Spatial particle-orbit generator for conv encoders (+ MLP).

Design:
- Coherent blob: particles move together on a circular orbit mu(t) = (r cos(omega t), r sin(omega t))
  with per-particle diffusion and observation noise.
- Random noise: random walk with AR(1) coefficient ("a discount factor of the previous position's noise").
- Output: timeseries (t_max, N*2) ready for CPIC; interleaved particle coordinates
  [x0,y0,x1,y1,...] after sorting particles by x at t=0 (label-agnostic, geometric order).

Ground truth for evaluation only (never passed to CPIC): (t_max, 2) circular trajectory (cos, sin) for R2 / alignment.

Filter / kernel interpretability (particle representation)
---------------------------------------------------------
The feature axis is x-sorted at t=0 and interleaved as [x0,y0,x1,y1,...], so index j maps to a
known spatial ordering along the x-axis in the plane (analogous to electrodes left-to-right).
Per-particle importance can be summarized by summing absolute filter weights across output channels
and visualized as a scatter of particles at their t=0 positions colored by importance.
For a spatiotemporal kernel with shape (k_feat, k_time), the heatmap along the time axis encodes
which temporal fragments of the dynamics (e.g. segments of the circular orbit) the filter responds to.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# -----------------------------------------------------------------------------
# Particle simulation
# -----------------------------------------------------------------------------

def _run_blob(t_max, centroid_x, centroid_y, num_particles, sigma_blob, rng):
    """
    Blob positions (t_max, num_particles, 2): centroid[t] + i.i.d. offset each t (no divergence).

    Offset from centroid is redrawn every timestep (add sigma_blob per timestep), so the
    blob stays a fixed-size cloud around the orbit and does not spread out over time.

    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    centroid_x : np.ndarray, shape (t_max,)
        x-coordinates of the centroid.
    centroid_y : np.ndarray, shape (t_max,)
        y-coordinates of the centroid.
    num_particles : int
        Number of particles in the coherent blob.
    sigma_blob : float
        Std of offset from centroid per particle per timestep.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    positions : np.ndarray, shape (t_max, num_particles, 2)
        Positions of the particles.
    """
    positions = np.zeros((t_max, num_particles, 2))
    for t in range(t_max):
        positions[t] = np.column_stack([centroid_x[t], centroid_y[t]]) + rng.normal(0, sigma_blob, (num_particles, 2)) 
    return positions


def _run_random_walk(t_max, num_particles, sigma_noise, spatial_bounds, rng, noise_ar_coeff=0.8):
    """
    Positions (t_max, num_particles, 2): init uniform in [-B,B]^2, then AR(1) dynamics.
    
    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    num_particles : int
        Number of random noise particles.
    sigma_noise : float
        Standard deviation of the noise.
    spatial_bounds : float
        Half-extent of the plane: [-spatial_bounds, spatial_bounds]^2.
    rng : np.random.Generator
        Random number generator.
    noise_ar_coeff : float, optional
        AR(1) coefficient for noise particles. Should satisfy 0 <= coeff < 1 for
        stable mean-reverting dynamics. coeff=1 corresponds to a random walk. Default is 0.8.

    Returns
    -------
    positions : np.ndarray, shape (t_max, num_particles, 2)
        Positions of the particles.
    """
    positions = np.zeros((t_max, num_particles, 2))
    positions[0] = rng.uniform(-spatial_bounds, spatial_bounds, (num_particles, 2))
    if not (0.0 <= noise_ar_coeff < 1.0):
        raise ValueError(f"noise_ar_coeff must be in [0, 1), got {noise_ar_coeff}")
    for t in range(1, t_max):
        positions[t] = noise_ar_coeff * positions[t - 1] + rng.normal(0, sigma_noise, (num_particles, 2))
    return positions


def generate_particle_orbit_positions(
    t_max,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    noise_ar_coeff=0.8,
    spatial_bounds=10.0,
    seed=None):
    """
    Generate 2D particle positions: coherent blob (circular orbit + diffusion) + random noise.

    Smaller sigma_blob keeps the blob tighter and easier to discern from noise;
    larger value make the coherent signal harder to see (useful for harder CPIC tasks).

    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    num_blob : int
        Number of particles in the coherent blob.
    num_noise : int
        Number of random noise particles.
    orbit_radius : float
        Radius of the circular orbit.
    omega : float
        Angular speed of the circular orbit.
    sigma_blob : float
        Standard deviation of the Gaussian random walk.
    noise_ar_coeff : float
        AR(1) coefficient for noise particles. The step noise std is derived as
        spatial_bounds * sqrt(1 - noise_ar_coeff**2) so the stationary distribution
        fills [-spatial_bounds, spatial_bounds]^2.
    spatial_bounds : float
        Half-extent of the plane: [-spatial_bounds, spatial_bounds]^2.
    seed : int, optional
        Random seed.

    Returns
    -------
    positions : np.ndarray, shape (t_max, 80, 2) -> (t_max - T, T, 160)
        (x, y) for all N = num_blob + num_noise particles.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    t_axis = np.arange(t_max, dtype=np.float64)
    centroid_x = orbit_radius * np.cos(omega * t_axis)
    centroid_y = orbit_radius * np.sin(omega * t_axis)
    positions_blob = _run_blob(t_max, centroid_x, centroid_y, num_blob, sigma_blob, rng)

    sigma_noise = spatial_bounds * np.sqrt(1 - noise_ar_coeff**2)
    positions_noise = _run_random_walk(t_max, num_noise, sigma_noise, spatial_bounds, rng, noise_ar_coeff=noise_ar_coeff)
    
    positions = np.concatenate([positions_blob, positions_noise], axis=1)
    
    return positions


# -----------------------------------------------------------------------------
# Representation: interleaved particle coordinates (t_max, N*2)
# -----------------------------------------------------------------------------

def _positions_to_particle_timeseries(positions, seed_positions=None):
    """
    Convert (t_max, N, 2) particle positions into a standardized (t_max, N*2) timeseries.

    Particles are ordered by increasing x-coordinate at t=0 (purely geometric; no use of blob/noise
    labels). Flattening is interleaved: after sorting, particle i occupies columns 2*i and
    2*i+1 as [x_i, y_i]. Each column is standardized independently across time.

    Parameters
    ----------
    positions : np.ndarray, shape (t_max, N, 2)
        Particle positions (x, y) over time in original simulation order.
    seed_positions : np.ndarray of shape (N,) optional
        x-coordinates at t=0 used to define the sort order (e.g. training-set x at t=0). If None,
        uses positions[0, :, 0]. Pass the same seed_positions on held-out data to apply the
        identical column ordering as on the reference run.

    Returns
    -------
    data : np.ndarray, shape (t_max, N*2), dtype float32
        Standardized interleaved coordinates.
    particle_order : np.ndarray, shape (N,), dtype int
        Permutation indices such that sorted particle index i is original particle particle_order[i].
        Feature columns 2*i:2*i+2 correspond to that particle after sorting.
    standardization_mean : np.ndarray, shape (N*2,), dtype float32
        Per-feature mean used for standardization.
    standardization_std : np.ndarray, shape (N*2,), dtype float32
        Per-feature std used for standardization.
    """
    t_max, N, _ = positions.shape
    if seed_positions is None:
        x0 = positions[0, :, 0] # x-coordinates at t=0
    else:
        x0 = np.asarray(seed_positions, dtype=np.float64)
        if x0.shape != (N,):
            raise ValueError(f"seed_positions must have shape ({N},), got {x0.shape}")

    particle_order = np.argsort(x0)
    pos_sorted = positions[:, particle_order, :]
    # C-order reshape: (t_max, N, 2) -> (t_max, N*2) as x0,y0,x1,y1,...
    flat = pos_sorted.reshape(t_max, N * 2).astype(np.float64)

    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    data = np.float32((flat - mean) / std)
    return data, particle_order, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


# -----------------------------------------------------------------------------
# Time series generation
# -----------------------------------------------------------------------------

def _generate_ground_truth_latent(t_max, omega=0.05, scale=1.0):
    """
    Ground truth 2D latent (cos, sin) for evaluation (e.g. R2, linear alignment).

    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    omega : float
        Angular speed of the circular orbit.
    scale : float
        Amplitude of the signal.

    Returns
    -------
    ground_truth_latent : np.ndarray, shape (t_max, 2)
    """
    t = np.arange(t_max, dtype=float)
    return np.column_stack([scale * np.cos(omega * t), scale * np.sin(omega * t)])


def generate_particle_orbit_process_timeseries(
    t_max=200,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    noise_ar_coeff=0.8,
    spatial_bounds=10.0,
    seed=None):
    """
    Generate particle-orbit blob timeseries ready for CPIC.

    1) Simulates 2D particles (coherent blob + random noise).
    2) Converts positions to interleaved particle coordinates after x-sort at t=0.
    3) Standardizes the interleaved particle coordinates.
    4) Generates ground truth latent (cos, sin) for evaluation.

    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    num_blob, num_noise : int
        Particles in coherent blob and random noise.
    orbit_radius, omega : float
        Blob orbit (r cos(\omega t), r sin(\omega t)).
    sigma_blob : float
        Blob spread (per-particle diffusion std).
    noise_ar_coeff : float
        AR(1) coefficient for noise particles. Step noise std is derived as
        spatial_bounds * sqrt(1 - noise_ar_coeff**2) to fill the domain.
    spatial_bounds : float
        Plane [-spatial_bounds, spatial_bounds]^2.
    seed : int, optional
        Random seed.

    Returns
    -------
    data : np.ndarray, float32, shape (t_max, N*2)
        Standardized interleaved particle coordinates after x-sort at t=0; N = num_blob + num_noise.
    ground_truth_latent : np.ndarray, shape (t_max, 2)
        Circular (cos, sin) trajectory for evaluation.
    particle_order : np.ndarray, shape (N,)
        Sort indices (by x at t=0). Columns 2*i:2*i+2 refer to original particle particle_order[i].
    particle_labels : np.ndarray, shape (N,), int8
        1 = coherent blob, 0 = noise, in **original** simulation order (before sorting).
        particle_labels[particle_order] gives labels aligned with the sorted feature columns
        (pair [2*i, 2*i+1] corresponds to label particle_labels[particle_order[i]]).
    standardization_mean : np.ndarray, shape (N*2,), float32
        Per-feature mean used for standardization (for inverse-transforming to grid coords).
    standardization_std : np.ndarray, shape (N*2,), float32
        Per-feature std used for standardization.
    positions : np.ndarray, shape (t_max, N, 2), float64
        Raw particle positions before standardization (original simulation order).
    """
    positions = generate_particle_orbit_positions(
        t_max=t_max,
        num_blob=num_blob, num_noise=num_noise,
        orbit_radius=orbit_radius, omega=omega,
        sigma_blob=sigma_blob,
        noise_ar_coeff=noise_ar_coeff,
        spatial_bounds=spatial_bounds,
        seed=seed)
    N = num_blob + num_noise
    particle_labels = np.zeros(N, dtype=np.int8)
    particle_labels[:num_blob] = 1
    data, particle_order, standardization_mean, standardization_std = _positions_to_particle_timeseries(positions)
    ground_truth_latent = _generate_ground_truth_latent(t_max, omega=omega, scale=1.0)
    return data, ground_truth_latent, particle_order, particle_labels, standardization_mean, standardization_std, positions


# -----------------------------------------------------------------------------
# Animation of the particle-orbit process
# -----------------------------------------------------------------------------

def animate_particle_orbit_process(
    t_max=500,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    sigma_noise=0.5,
    noise_ar_coeff=0.8,
    spatial_bounds=10.0,
    seed=42,
    interval=50,
):
    """
    Animate the particle-orbit blob system in 2D: blob (orange), random noise (gray),
    structured noise (orange).

    Parameters
    ----------
    particle_positions : (t_max, N, 2) float
        Full particle trajectories.
    num_blob, num_noise, n_structured : int
        Counts for the three groups (first num_blob = blob, next num_noise = random, rest = structured).
    spatial_bounds : float
        Axis limits [-spatial_bounds, spatial_bounds] for x and y.
    interval : int
        Delay between frames in ms.
    save_path : str, optional
        If set, save the animation to this path (e.g. .gif with writer='pillow').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Use plt.show() to display, or ani.save(...) to save elsewhere.
    """
    positions = generate_particle_orbit_positions(
        t_max=t_max,
        num_blob=num_blob,
        num_noise=num_noise,
        orbit_radius=orbit_radius,
        omega=omega,
        sigma_blob=sigma_blob,
        sigma_noise=sigma_noise,
        noise_ar_coeff=noise_ar_coeff,
        spatial_bounds=spatial_bounds,
        seed=seed
        )

    particle_labels = np.zeros(num_blob + num_noise, dtype=np.int8)
    particle_labels[:num_blob] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-spatial_bounds, spatial_bounds)
    ax.set_ylim(-spatial_bounds, spatial_bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

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
        ax.set_title(f"t = {frame}  (orange = coherent blob, gray = random noise)")
        return (scat,)

    return animation.FuncAnimation(fig, update, frames=t_max, init_func=init, interval=interval, blit=False)


# -----------------------------------------------------------------------------
# 3D verification plot (x, y, t)
# -----------------------------------------------------------------------------

def plot_verification_3d(positions, num_blob, num_noise, path=None):
    """
    3D (x, y, t) trajectory plot: coherent blob (one color) and random noise (another).
    The blob should appear as a helix/tube through the cloud.

    Parameters
    ----------
    positions : np.ndarray, shape (t_max, N, 2)
        Particle positions (x, y) over time; first num_blob are blob, rest are noise.
    num_blob, num_noise : int
        Counts for blob and noise particles.
    path : str, optional
        If set, save the figure to this path and close; otherwise return (fig, ax).

    Returns
    -------
    fig, ax : matplotlib figure and 3D axes, or None if path was set.
    """
    t_max, N, _ = positions.shape
    t_axis = np.arange(t_max)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    blob_end = num_blob
    ax.scatter(
        positions[:, :blob_end, 0].ravel(order="F"),
        positions[:, :blob_end, 1].ravel(order="F"),
        np.repeat(t_axis, blob_end),
        c="C1",
        s=2,
        alpha=0.6,
        label="coherent blob",
    )
    ax.scatter(
        positions[:, blob_end:, 0].ravel(order="F"),
        positions[:, blob_end:, 1].ravel(order="F"),
        np.repeat(t_axis, num_noise),
        c="gray",
        s=2,
        alpha=0.2,
        label="random noise",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.legend()
    fig.subplots_adjust(left=0, right=0.86, bottom=0.07, top=0.95)

    if path is not None:
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return None
    return fig, ax


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    ani = animate_particle_orbit_process(
        t_max=500,
        num_blob=30,
        num_noise=50,
        orbit_radius=3.0,
        omega=0.05,
        sigma_blob=0.7,
        sigma_noise=0.5,
        noise_ar_coeff=0.8,
        spatial_bounds=10.0,
        seed=42,
        interval=50,
    )
    #ani.save(os.path.join(_dir, "particle_orbit_process.gif"), fps=20, writer="pillow")
    plt.show()
