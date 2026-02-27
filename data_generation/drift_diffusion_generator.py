import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def generate_drift_diffusion(
        T=1000, # number of timesteps
        N=20, # number of blobs
        drift_rate=0.08, # how fast/far group drifts
        rotation_speed=0.05, # angular speed of drift direction
        cohesion_strength=0.1, # how strongly individuals follow the group
        dt=1.0, # displacement at time t
        diffusion_std=0.3, # individual noise
        seed=42
    ):

    np.random.seed(seed)

    positions = np.zeros((T, N, 2))
    positions[0] = np.random.normal(0, 0.1, (N, 2))  # start near origin

    for t in range(1, T):
        centroid = positions[t-1].mean(axis=0)
        
        drift_vector = drift_rate * np.array([np.cos(rotation_speed*t), np.sin(rotation_speed*t)])
        cohesion_term = cohesion_strength * np.sin(centroid - positions[t-1])
        random_diffusion = np.random.normal(0, diffusion_std, (N, 2))

        positions[t] = positions[t-1] + (drift_vector + cohesion_term) * dt + random_diffusion

    return positions


def animate_drift_diffusion(
        T=1000,
        N=20,
        drift_rate=0.08,
        rotation_speed=0.05,
        cohesion_strength=0.1,
        dt=1.0,
        diffusion_std=0.3,
        x_max=4,
        y_max=6
    ):

    positions = generate_drift_diffusion(T,
                                         N,
                                         drift_rate,
                                         rotation_speed,
                                         cohesion_strength,
                                         dt,
                                         diffusion_std,
                                        )

    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=80, color='orange', alpha=0.8)
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        scat.set_offsets(positions[frame])
        ax.set_title(f"t = {frame}")
        return scat,

    return animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=50, blit=True)


if __name__ == "__main__":
    ani = animate_drift_diffusion(T=1000,
                                  N=20,
                                  drift_rate=0.2,
                                  rotation_speed=0.05,
                                  cohesion_strength=0.1,
                                  dt=1.0,
                                  diffusion_std=0.3,
                                  x_max=4,
                                  y_max=6
                                )
    # ani.save("drift_diffusion.gif", fps=30, writer='pillow')
    plt.show()
