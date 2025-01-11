import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

ANIMATIONS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/animations/"
FIGS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/figs/"

# Simulation parameters
c = 1  # Speed of light (m/s)
mu_0 = 377  # Permeability of free space (H/m)
epsilon_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)
sigma = 0.0  # Conductivity (S/m)
sigma_star = 0.0  # Magnetic conductivity (S/m)

L = 1.0  # Length of the simulation domain (m)
nx = 200  # Number of spatial grid points
dx = L / nx  # Spatial step

time_steps = 500  # Number of time steps
S = 0.5 # Courant number
dt = S*dx/c  # Time step satisfying the CFL condition


# Source parameters
source_position = 50  # Center of the spatial grid
source_width = 5  # Width of the Gaussian envelope
source_peak_timestep = 30  # Timestep at which the source reaches its peak

# Fields and currents
Ez = np.zeros(nx)  # Electric field (z-direction)
Hy = np.zeros(nx-1)  # Magnetic field (y-direction)

# Precompute coefficients for FDTD update equations
ce1 = (dt / epsilon_0) / (1 + dt * sigma / (2 * epsilon_0))
ce2 = (1 - dt * sigma / (2 * epsilon_0)) / (1 + dt * sigma / (2 * epsilon_0))

ch1 = (dt / mu_0) / (1 + dt * sigma_star / (2 * mu_0))
ch2 = (1 - dt * sigma_star / (2 * mu_0)) / (1 + dt * sigma_star / (2 * mu_0))

# Gaussian pulse source
tspace = np.arange(time_steps)
source = np.exp(-((tspace - source_peak_timestep) ** 2) / (2 * source_width ** 2))

# Arrays to store simulation data for plotting
Ez_history = []
Hy_history = []

# Start simulation timer
start_time = time.time()

S_prima = S/(np.sqrt(mu_0*epsilon_0)*c)

ABC_constants = {
    "E-2[0]": 0,
    "E-2[1]": 0,
    "E-2[2]": 0,
    "E-1[0]": 0,
    "E-1[1]": 0,
    "E-1[2]": 0,
    "E-2[-1]": 0,
    "E-2[-2]": 0,
    "E-2[-3]": 0,
    "E-1[-1]": 0,
    "E-1[-2]": 0,
    "E-1[-3]": 0
}

# Time-stepping loop
for n in range(time_steps):
    # Update Hy field
    for i in range(nx - 1):
        Hy[i] = ch1 * ((Ez[i + 1] - Ez[i]) / dx) + ch2 * Hy[i]

    # Update Ez field
    for i in range(1, nx-1):
        Ez[i] = ce1 * ((Hy[i] - Hy[i - 1]) / dx) + ce2 * Ez[i]

    # Apply source (sinusoidal current density with Gaussian envelope)
    Ez[source_position] += source[n]

    # Second order ABC
    Ez[0] = -1 / (1/S_prima  + 2 + S_prima) * (
        (1/S_prima -2 + S_prima)*(Ez[2] + ABC_constants["E-2[0]"]) +
        2*(S_prima - 1/S_prima)*(ABC_constants["E-1[0]"]+ABC_constants["E-1[2]"]-Ez[1]-ABC_constants["E-2[1]"]) - 
        4*(1/S_prima +S_prima)*ABC_constants["E-1[1]"]) - ABC_constants["E-2[2]"]
    
    Ez[-1] = -1 / (1/S_prima  + 2 + S_prima) * (
        (1/S_prima -2 + S_prima)*(Ez[-3] + ABC_constants["E-2[-1]"]) +
        2*(S_prima - 1/S_prima)*(ABC_constants["E-1[-1]"]+ABC_constants["E-1[-3]"]-Ez[-2]-ABC_constants["E-2[-2]"]) - 
        4*(1/S_prima +S_prima)*ABC_constants["E-1[-2]"]) - ABC_constants["E-2[-3]"]

    # Old 1 order ABC
    #Ez[-1] =  ABC_constants["E-1[-2]"] + (S_prima - 1)/(S_prima + 1) * (Ez[-2]-Ez[-1])  

    # Update ABC constants
    ABC_constants["E-2[2]"] = ABC_constants["E-1[2]"]
    ABC_constants["E-2[1]"] = ABC_constants["E-1[1]"]
    ABC_constants["E-2[0]"] = ABC_constants["E-1[0]"]
    ABC_constants["E-1[0]"] = Ez[0]
    ABC_constants["E-1[1]"] = Ez[1]
    ABC_constants["E-1[2]"] = Ez[2]
    ABC_constants["E-2[-1]"] = ABC_constants["E-1[-1]"]
    ABC_constants["E-2[-2]"] = ABC_constants["E-1[-2]"]
    ABC_constants["E-2[-3]"] = ABC_constants["E-1[-3]"]
    ABC_constants["E-1[-1]"] = Ez[-1]
    ABC_constants["E-1[-2]"] = Ez[-2]
    ABC_constants["E-1[-3]"] = Ez[-3]

    # Store fields for plotting
    if n % 10 == 0:
        Ez_history.append(Ez.copy())
        Hy_history.append(Hy.copy())

# End simulation timer
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

# Animation setup
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

Ez_line, = axes[0].plot([], [], label='Ez')
Hy_line, = axes[1].plot([], [], label='Hy', color='orange')

axes[0].set_xlim(0, nx)
axes[0].set_ylabel("Field amplitude")
axes[0].legend()

axes[1].set_xlim(0, nx-1)
axes[1].set_xlabel("Grid index")
axes[1].set_ylabel("Field amplitude")
axes[1].legend()

# Add some space between the subplots to fit the time (title of the second subplot)
plt.subplots_adjust(hspace=0.5)


# Compute y-axis limits for each field
Ez_min, Ez_max = np.min(Ez_history), np.max(Ez_history)
Hy_min, Hy_max = np.min(Hy_history), np.max(Hy_history)

# Add some padding to the y-axis limits
Ez_min, Ez_max = 1.1 * Ez_min, 1.1 * Ez_max
Hy_min, Hy_max = 1.1 * Hy_min, 1.1 * Hy_max

axes[0].set_ylim(Ez_min, Ez_max)
axes[1].set_ylim(Hy_min, Hy_max)

# Animation function
def update(frame):
    Ez_line.set_data(range(nx), Ez_history[frame])
    Hy_line.set_data(range(nx-1), Hy_history[frame])

    # Set the title to show the time
    axes[0].set_title(f"Time: {frame * dt:.2e} s")
    
    return Ez_line, Hy_line

ani = animation.FuncAnimation(fig, update, frames=len(Ez_history), interval=200, blit=False)
plt.show()

# Save animation as GIF
#file_name = "1DYeesAlgorithm2orderABC"
#ani.save(ANIMATIONS+file_name, writer="pillow", fps=20)
#print(f"Animation saved as {file_name}.")

