import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from scipy.interpolate import interp1d

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
    u[0], u[1] = initial_conditions(nx)

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

# Simulation parameters
c = 1.0            # Wave propagation speed
dx = 0.01          # Spatial step
nx = 250           # Number of spatial points

# Define an initial condition for t=0,1
def gaussian_pulse(nx, width=80, a=1.0, b=0.5, c=0.05):
    # Set a Gaussian pulse as initial condition
    # t0 is the initial condition at time t=0
    # t1 should be the initial condition projected one time step into the future
    x = np.linspace(0, 1, width)
    t0 = np.zeros(nx)
    t0[(nx // 2 - width // 2):(nx // 2 + width // 2)] = a* np.exp(- (x - b) ** 2 / (2*c**2))
    t1 = np.zeros(nx)
    t1[(nx // 2 - width // 2 + 1):(nx // 2 + width // 2 + 1)] = a* np.exp(- (x - b) ** 2 / (2*c**2))
    return t0, t1

# Courant stability factors to test
S_values = [1.0, 0.99, 0.5]

# Time step corresponding to S values
dt_values = [S * dx / c for S in S_values]

# Choose a time maximum and choose the number of time steps for each S value so that it perfectly matches the time maximum
nt = 100
nt_values = [int(nt * dt_values[0] / dt) for dt in dt_values]
print("Max time for each:", [dt * nt for dt, nt in zip(dt_values, nt_values)])
print("Number of time steps for each:", nt_values)

# Compute wavefields for all S values
wavefields = [fdtd_1d(c, dt, dx, nx, nt, gaussian_pulse) for dt, nt in zip(dt_values, nt_values)]

def plot_last_timestep(wavefields, S_values, save=False):
    plt.figure(figsize=(10, 5))
    for i, S in enumerate(S_values):
        plt.plot(wavefields[i][-1], label=f"S = {S}")

    plt.xlabel("Position")
    plt.ylabel("Amplitude")
    plt.xlim(150, nx)
    plt.legend()
    if save:
        plt.savefig(FIGS + '1DGaussianPulse.png')
    plt.show()

plot_last_timestep(wavefields, S_values, save=False)