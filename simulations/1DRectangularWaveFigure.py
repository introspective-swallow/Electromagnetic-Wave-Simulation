import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from scipy.interpolate import interp1d
import os

ANIMATIONS = "/animations/"
FIGS = "/figs/"

def fdtd_1d(c, dt, dx, nx, nt, initial_conditions):
    """
    Simulate 1D Maxwell's equations using the explicit second-order FDTD method.

    Parameters:
        c (float): Wave propagation speed.
        dt (float): Time step.
        dx (float): Spatial step.
        nx (int): Number of spatial grid points.
        nt (int): Number of time steps.
        initial_condition (callable): Function defining the initial condition

    Returns:
        np.ndarray: 2D array with the wavefield at all time steps.
    """
    # Courant condition check
    assert c * dt / dx <= 1, "Stability condition violated: c * dt / dx must be <= 1."

    # Start counting the simulation time
    start_time = time.time()

    # Initialize fields
    u = np.zeros((nt, nx))

    # Initialize rectangular source
    u[0], u[1] = initial_conditions(nx, dt)

    # Time stepping loop
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = (
                (c * dt / dx) ** 2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
                + 2 * u[n, i]
                - u[n - 1, i]
            )

        

    # Print the simulation time
    print(f"Simulation time: {time.time() - start_time:.2f} seconds")
    return u


# Define an initial condition for t=0,1
def initial_conditions(nx, dt):
    t0 = np.zeros(nx)
    t0[(nx // 2-20):(nx // 2+20)] = np.ones(40)
    t1 = np.zeros(nx)
    S = c*dt/dx
    # Interpolate next time step: u(i,t+dt) = (1-S)u(i,t) + S(u(i-1,t))
    t1 = (1-S)*t0 + S*np.roll(t0,1)
    return t0, t1

# Simulation parameters
c = 1.0              # Wave propagation speed
grid_sampling_resolution_per_wavelength = 10        # Grid sampling resolution per wavelength, if > 3, dispersionless
min_wavelength = 0.1
dx = min_wavelength / grid_sampling_resolution_per_wavelength          # Spatial step
xmax = 3         # Maximum spatial point
nx = int(xmax / dx)           # Number of spatial points
print("Number of spatial points:", nx)
print("Spatial step:", dx)
print("Max spatial point:", nx*dx)

# Courant stability factors to test
S_values = [1.0, 0.99, 0.5]

# Time step corresponding to S values
dt_values = [S * dx / c for S in S_values]
print(dt_values)
# Select number of timesteps such that maximum timestep agrees for all different timesteps values
print([x[0]-x[1] for x in zip(dt_values[1:],dt_values[:-1]) ])
factor_10 = np.floor((np.log10(np.min(np.abs([x[0]-x[1] for x in zip(dt_values[1:],dt_values[:-1])])))))
print(factor_10)
dt_values_to_int = [int(np.round(dt / 10**factor_10)) for dt in dt_values]
print(dt_values_to_int)
tmax = np.lcm.reduce(dt_values_to_int) * 10**factor_10

nt_values = [ int(tmax / dt) for dt in dt_values]
if np.max(nt_values) < 5:
    nt_values = np.array(nt_values)*10

print("Number of time steps for each S value:", nt_values)
print("Max time for each:", [dt * nt for dt, nt in zip(dt_values, nt_values)])


def plot_frame(wavefields, S_values, frame=-1, save=False, filename="1DGaussianPulse", foldername=FIGS, show=True):
    plt.figure(figsize=(10, 5))
    for i, S in enumerate(S_values):
        plt.plot(wavefields[i][frame], label=f"S = {S}")

    plt.xlabel("Position")
    plt.ylabel("Amplitude")
    # Show time=tmax at the top left corner
    plt.text(0.02, 0.95, f"t = {tmax:.2f}s", transform=plt.gca().transAxes, fontsize=12)
    # Set the x-axis to be in a range around nmax//2 + tmax*c
    plt.xlim(nx // 2 + tmax*c/dx - 50, nx // 2 + tmax*c /dx+ 50)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    if save:
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        plt.savefig(foldername + filename + '.png')
        print(f"Frame {frame} saved as {foldername + filename + '.png'}")
    if show == True:
        plt.show()

def save_frames(wavefields, S_values, skipframes=1, filename="1DGaussianPulse", foldername=FIGS):
    for j, i in enumerate(range(0, wavefields[0].shape[0], skipframes)):
        plot_frame(wavefields, S_values, frame=i, save=True, filename=filename + f"{j+1}", foldername=foldername, show=False)

# Compute wavefields for all S values
wavefields = [fdtd_1d(c, dt, dx, nx, nt, initial_conditions) for dt, nt in zip(dt_values, nt_values)]

save_frames(wavefields, S_values, skipframes=5, filename="1DRectangularWave", foldername=FIGS + "1DRectangularWave/")

