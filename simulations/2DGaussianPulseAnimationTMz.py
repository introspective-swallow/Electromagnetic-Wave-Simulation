import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
ANIMATIONS = "/animations/"


# Constants
c = 3e8  # Speed of light (m/s)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
epsilon_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)
sigma = 0.0  # Electrical conductivity (S/m)
sigma_star = 0.0  # Magnetic conductivity (S/m)

# Simulation parameters
nx, ny = 100, 100  # Grid size
dx, dy = 1e-2, 1e-2  # Spatial steps
dt = dx / (2 * c)  # Time step (satisfying CFL condition)
time_steps = 160  # Number of time steps

# Medium coefficients
Ca = np.ones((nx, ny))
Cb = dt / (epsilon_0 * dx) * np.ones((nx, ny))
Da = np.ones((nx, ny))
Db = dt / (mu_0 * dx) * np.ones((nx, ny))

# Fields
Ez = np.zeros((nx, ny))  # Electric field (z-direction)
Hx = np.zeros((nx, ny))  # Magnetic field (x-direction)
Hy = np.zeros((nx, ny))  # Magnetic field (y-direction)

# Source parameters
source_frequency = 1e9  # Source frequency (Hz)
source_position = (nx // 2, ny // 2)  # Center of the grid

# Create temporal profile of the source
t = np.arange(time_steps) * dt
source_temporal = np.sin(2 * np.pi * source_frequency * t)

# Create spatial profile of the source
x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
source_position = (nx // 2, ny // 2)
source_sigma = 1.0  # Controls the spatial extent of the source
spatial_envelope = np.exp(
    -((x - source_position[0]) ** 2 + (y - source_position[1]) ** 2) / (2 * source_sigma**2)
)

# Create the source field
source_field = np.zeros((time_steps, nx, ny))
for n in range(time_steps):
    source_field[n] = spatial_envelope * source_temporal[n]

# Plot source at its peak
# plt.figure()
# plt.imshow(source_field[-1], cmap="hot", origin="lower")
# plt.colorbar()
# plt.title("Source field at its peak")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Preallocate storage for fields at each time step
Ez_storage = np.zeros((time_steps, nx, ny))
Hx_storage = np.zeros((time_steps, nx, ny))
Hy_storage = np.zeros((time_steps, nx, ny))

# Simulation loop with progress bar
print("Running simulation...")
for n in tqdm(range(time_steps), desc="Simulating"):
    # Update Hx
    Hx[:, :-1] = Da[:, :-1] * Hx[:, :-1] + Db[:, :-1] * (Ez[:, :-1] - Ez[:, 1:])

    # Update Hy
    Hy[:-1, :] = Da[:-1, :] * Hy[:-1, :] - Db[:-1, :] * (Ez[:-1, :] - Ez[1:, :])

    # Update Ez
    Ez[1:-1, 1:-1] = (
        Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1]
        + Cb[1:-1, 1:-1]
        * (
            (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1])
            - (Hx[1:-1, 1:-1] - Hx[1:-1, :-2])
        )
    )

    # Apply the source
    Ez += source_field[n]

    # Store fields
    Ez_storage[n] = Ez
    Hx_storage[n] = Hx
    Hy_storage[n] = Hy

print("Simulation complete.")

def animate_wavefields():
    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Add a text annotation for simulation time at the center bottom of the figure
    axtext = fig.add_axes([0.1, 0.02, 0.8, 0.07])
    axtext.axis("off")
    time = axtext.text(0.5,0.5, str(0), ha="center", va="bottom", fontsize=12)

    fields = [Ez_storage, Hx_storage, Hy_storage]
    titles = [r"$E_z$", r"$H_x$", r"$H_y$"]

    # Use 90th percentile to set color range
    color_ranges = [
        (np.percentile(field, 1), np.percentile(field, 99)) for field in fields
    ]
    
    
    images = []
    colorbars = []

    # Initialize plots
    for i, ax in enumerate(axs):
        img = ax.imshow(
            fields[i][0],
            vmin=color_ranges[i][0],
            vmax=color_ranges[i][1],
            cmap="RdBu",
            interpolation="nearest",
        )
        images.append(img)
        ax.set_title(titles[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = plt.colorbar(img, ax=ax, orientation="vertical")
        colorbars.append(cbar)

    # Update function for animation
    def update(n):
        for i, img in enumerate(images):
            img.set_data(fields[i][n])
            img.set_clim(vmin=color_ranges[i][0], vmax=color_ranges[i][1])
        time.set_text(f"Time: {n*dt*1e9:.2f} ns")
        return [*images, time]

    # Animation with progress bar
    print("Generating animation...")
    ani = FuncAnimation(fig, update, frames=range(time_steps), interval=30, blit=True)
    # ani.save(ANIMATIONS+"2DGaussianPulse.gif", fps=30)
    # print("Animation saved as 2DGaussianPulse.gif.")

    plt.show()

# Run the animation
animate_wavefields()