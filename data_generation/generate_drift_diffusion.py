"""
Spatial drift-diffusion generator for ConvSpatial2DEncoder (+ MLP).

Design:
- Coherent blob: particles move together on a circular orbit mu(t) = (r cos(omega t), r sin(omega t))
  with per-particle diffusion and observation noise.
- Random noise: pure 2D random walks.
- Output: timeseries (t_max, GxG) ready for CPIC (Gaussian splatting grid representation).

Ground truth for evaluation only (never passed to CPIC): (t_max, 2) circular trajectory (cos, sin) for R2 / alignment.
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


def _run_random_walk(t_max, num_particles, sigma_noise, spatial_bounds, rng):
    """
    Positions (t_max, num_particles, 2): init uniform in [-B,B]^2, then 2D random walk.
    
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

    Returns
    -------
    positions : np.ndarray, shape (t_max, num_particles, 2)
        Positions of the particles.
    """
    positions = np.zeros((t_max, num_particles, 2))
    positions[0] = rng.uniform(-spatial_bounds, spatial_bounds, (num_particles, 2))
    for t in range(1, t_max):
        positions[t] = positions[t - 1] + rng.normal(0, sigma_noise, (num_particles, 2))
    return positions


def generate_drift_diffusion_2d_positions(
    t_max,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    sigma_noise=0.5,
    spatial_bounds=10.0,
    seed=None,
    ):
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
    sigma_noise : float
        Standard deviation of the noise.
    spatial_bounds : float
        Half-extent of the plane: [-spatial_bounds, spatial_bounds]^2.
    seed : int, optional
        Random seed.

    Returns
    -------
    positions : np.ndarray, shape (t_max, N, 2)
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

    positions_noise = _run_random_walk(t_max, num_noise, sigma_noise, spatial_bounds, rng)
    
    positions = np.concatenate([positions_blob, positions_noise], axis=1)
    
    return positions


# -----------------------------------------------------------------------------
# Representation: Spatial grid occupancy with Gaussian splatting (t_max, G*G)
# -----------------------------------------------------------------------------

def _positions_to_grid_timeseries(positions, spatial_bounds=10.0, G=32, sigma_splat=0.5):
    """
    Convert continuous particle positions into a z-scored spatio‑temporal grid representation.

    At each timestep, this function:
    1) Lays down a GxG grid covering [-spatial_bounds, spatial_bounds}]^2 
       and uses the cell centers as grid coordinates.
    2) For every particle, evaluates an isotropic 2D Gaussian (width sigma_splat) at every grid cell center 
       and sums contributions across particles, producing a smooth “density map” over the grid.
    3) Flattens each GxG map to length G * G and z-scores each grid cell
       across time (subtract mean over t_max, divide by std; degenerate cells get std = 1).

    Parameters
    ----------
    positions : np.ndarray, shape (t_max, N, 2)
        Continuous 2D positions (x, y) for N particles over t_max.
    spatial_bounds : float
        Half-extent of the spatial domain: the grid spans [-spatial_bounds, spatial_bounds}]^2.
    G : int
        Number of grid cells per spatial dimension (total cells = G * G).
    sigma_splat : float
        Standard deviation of the Gaussian kernel used to “splat” each particle onto the grid.
    
    Returns
    -------
    data : np.ndarray, shape (t_max, G*G), dtype float32
        Z-scored soft occupancy values per grid cell over time; this is the grid
        representation used as CPIC input.
    """
    t_max, N, _ = positions.shape

    # uniform grid spacing so that G cells cover [-spatial_bounds, spatial_bounds]
    cell_size = (2 * spatial_bounds) / G

    # 1D coordinates of grid cell centers along x and y
    xs = -spatial_bounds + (np.arange(G) + 0.5) * cell_size
    ys = -spatial_bounds + (np.arange(G) + 0.5) * cell_size

    # broadcast to full GxG grid of (x, y) centers
    cell_x = np.broadcast_to(xs[np.newaxis, :], (G, G))
    cell_y = np.broadcast_to(ys[:, np.newaxis], (G, G))
    cells = np.stack([cell_x, cell_y], axis=-1) # (G, G, 2)

    grid_flat = np.zeros((t_max, G * G), dtype=np.float64)
    for t in range(t_max):
        # squared Euclidean distance from each grid cell center to each particle
        diff = positions[t] - cells[:, :, np.newaxis, :]
        d2 = (diff ** 2).sum(axis=-1) # (G, G, N)

        # add up Gaussian contributions from all N particles at every grid cell
        M = np.exp(-d2 / (2 * sigma_splat**2)).sum(axis=-1) # (G, G)

        # store flattened grid for timestep t
        grid_flat[t] = M.ravel()
    grid_flat = grid_flat.astype(np.float32)

    mean = grid_flat.mean(axis=0, keepdims=True)
    std = grid_flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return np.float32((grid_flat - mean) / std)


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


def generate_drift_diffusion_process_timeseries(
    t_max=200,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    sigma_noise=0.5,
    spatial_bounds=10.0,
    grid_G=32,
    sigma_splat=0.5,
    seed=None,
):
    """
    Generate drift-diffusion blob timeseries ready for CPIC.

    1) Simulates 2D particles (coherent blob + random noise).
    2) Applies the chosen representation and z-scores.

    Parameters
    ----------
    t_max : int
        Number of time steps in the trajectory.
    num_blob, num_noise : int
        Particles in coherent blob and random noise.
    orbit_radius, omega : float
        Blob orbit (r cos(ωt), r sin(ωt)).
    sigma_blob, sigma_noise : float
        Blob spread, and noise-particle step std.
    spatial_bounds : float
        Plane [-spatial_bounds, spatial_bounds]^2.
    grid_G : int
        Grid side length; total cells = grid_G * grid_G.
    sigma_splat : float
        Gaussian splat width.
    seed : int, optional
        Random seed.

    Returns
    -------
    data : np.ndarray, float32
        (t_max, G*G) soft Gaussian splat, z-scored per cell.
    ground_truth_latent : np.ndarray, shape (t_max, 2)
        Circular (cos, sin) trajectory for evaluation.
    """
    positions = generate_drift_diffusion_2d_positions(
        t_max=t_max,
        num_blob=num_blob,
        num_noise=num_noise,
        orbit_radius=orbit_radius,
        omega=omega,
        sigma_blob=sigma_blob,
        sigma_noise=sigma_noise,
        spatial_bounds=spatial_bounds,
        seed=seed,
    )
    data = _positions_to_grid_timeseries(positions, spatial_bounds=spatial_bounds, G=grid_G, sigma_splat=sigma_splat)
    ground_truth_latent = _generate_ground_truth_latent(t_max, omega=omega, scale=1.0)

    return data, ground_truth_latent


# -----------------------------------------------------------------------------
# Animation of the drift-diffusion process
# -----------------------------------------------------------------------------

def animate_drift_diffusion_process(
    t_max=500,
    num_blob=30,
    num_noise=50,
    orbit_radius=3.0,
    omega=0.05,
    sigma_blob=0.7,
    sigma_noise=0.5,
    spatial_bounds=10.0,
    seed=42,
    interval=50,
):
    """
    Animate the drift-diffusion blob system in 2D: blob (orange), random noise (gray),
    structured noise (orange). Same style as drift_diffusion_spatial_quadrant.

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
    positions = generate_drift_diffusion_2d_positions(
        t_max=t_max,
        num_blob=num_blob,
        num_noise=num_noise,
        orbit_radius=orbit_radius,
        omega=omega,
        sigma_blob=sigma_blob,
        sigma_noise=sigma_noise,
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
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return None
    return fig, ax


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    ani = animate_drift_diffusion_process(
        t_max=500,
        num_blob=30,
        num_noise=50,
        orbit_radius=3.0,
        omega=0.05,
        sigma_blob=0.7,
        sigma_noise=0.5,
        spatial_bounds=10.0,
        seed=42,
        interval=50,
    )
    #ani.save(os.path.join(_dir, "drift_diffusion_process.gif"), fps=20, writer="pillow")
    plt.show()
