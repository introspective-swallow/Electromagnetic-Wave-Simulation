import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

ANIMATIONS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/animations/"
FIGS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/figs/"

# Acoustic simulation parameters
s66 = 1.5 # Strain coefficient
rho = 7.4 # Density (kg/m^3)

# Simulation parameters
c = 1/np.sqrt(rho*s66)  # Speed of light (m/s)
mu_0 = rho  # Permeability of free space (H/m)
epsilon_0 = s66  # Permittivity of free space (F/m)
sigma = 0.0  # Conductivity (S/m)
sigma_star = 0.0  # Magnetic conductivity (S/m)

L = 1.0  # Length of the simulation domain (m)
nx = 1000  # Number of spatial grid points
dx = L / nx  # Spatial step

time_steps = 2000  # Number of time steps
dt = dx / c  # Time step satisfying the CFL condition

# Source parameters
source_position = nx//2  # Center of the spatial grid
source_width = 5  # Width of the Gaussian envelope
source_peak_timestep = 30  # Timestep at which the source reaches its peak

# Fields and currents
Ez = np.zeros(nx)  # Electric field (z-direction)
Hy = np.zeros(nx)  # Magnetic field (y-direction)

# Precompute coefficients for FDTD update equations
ce1 = (dt / epsilon_0) / (1 + dt * sigma / (2 * epsilon_0))
ce2 = (1 - dt * sigma / (2 * epsilon_0)) / (1 + dt * sigma / (2 * epsilon_0))

ch1 = (dt / mu_0) / (1 + dt * sigma_star / (2 * mu_0))
ch2 = (1 - dt * sigma_star / (2 * mu_0)) / (1 + dt * sigma_star / (2 * mu_0))

# Gaussian pulse source
tspace = np.arange(time_steps)
source = np.sin(2 * np.pi * 5 * tspace * dt)

# Arrays to store simulation data for plotting
Ez_history = []
Hy_history = []

# Start simulation timer
start_time = time.time()

# Time-stepping loop
for n in range(time_steps):

    Hy[-1] = Hy[-2]  # PEC boundary condition
    # Update Hy field
    for i in range(nx - 1):
        Hy[i] = ch1 * ((Ez[i + 1] - Ez[i]) / dx) + ch2 * Hy[i]

    Ez[0] = Ez[1]  # PEC boundary condition
    # Update Ez field
    for i in range(1, nx):
        Ez[i] = ce1 * ((Hy[i] - Hy[i - 1]) / dx) + ce2 * Ez[i]

    # Apply source (sinusoidal current density with Gaussian envelope)
    Ez[source_position] += source[n]

    # Store fields for plotting
    if n % 10 == 0:
        Ez_history.append(Ez.copy())
        Hy_history.append(Hy.copy())

# End simulation timer
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

# Animation setup
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

Ez_line, = ax.plot([], [], label=r'$T_6$')
Hy_line, = ax.plot([], [], label=r'$v_y$', color='orange')

ax.set_xlim(0, nx)
ax.set_ylabel("Field amplitude")
ax.legend("upper right", bbox_to_anchor=(1.1, 1))

# Compute y-axis limits for each field
Ez_min, Ez_max = np.min(Ez_history), np.max(Ez_history)
Hy_min, Hy_max = np.min(Hy_history), np.max(Hy_history)
Total_min = min(Ez_min, Hy_min)
Total_max = max(Ez_max, Hy_max)

# Add some padding to the y-axis limits
Total_min -= 0.1 * abs(Total_min)
Total_max += 0.1 * abs(Total_max)
print(Total_min, Total_max)
ax.set_ylim(Total_min, Total_max)


# Animation function
def update(frame):
    Ez_line.set_data(range(nx), Ez_history[frame])
    Hy_line.set_data(range(nx), -Hy_history[frame])

    # Set the title to show the time
    ax.set_title(f"Time: {frame * dt:.2e} s")
    return Ez_line, Hy_line

ani = animation.FuncAnimation(fig, update, frames=len(Ez_history), interval=100, blit=False)
plt.show()

# Save animation as GIF
#ani.save(ANIMATIONS+"1DYeesAlgorithm.gif", writer="pillow", fps=20)
#print("Animation saved as fdtd_simulation.gif.")

