import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from scipy.interpolate import interp1d

ANIMATIONS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/animations/"
FIGS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/figs/"

def fdtd_1d(c, dt, dx, nx, nt, wavelength, initial_conditions):
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
    u[0], u[1] = initial_conditions(nx, dt, cc=wavelength)

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
nx = 1000           # Number of spatial points

# Define an initial condition for t=0,1
def gaussian_pulse(nx, dt, a=1.0, b=100*dx, cc=0.06):
    # Set a Gaussian pulse as initial condition
    # t0 is the initial condition at time t=0
    # t1 should be the initial condition projected one time step into the future
    
    x = np.linspace(0, nx*dx, nx)
    t0 = a* np.exp(- (x - b) ** 2 / (2*cc**2))
    
    # At the next timestep, the initial condition has propagated to the right at speed c, so it should be shifted to the left by c*dt
    t1 = a* np.exp(- (x - b - dt*c) ** 2 / (2*cc**2))
    
    return t0, t1

# Courant stability factors to test
S = 0.5

# Time step corresponding to S values
dt = S * dx / c

wavelengths = [0.05, 0.03, 0.02]

# Choose a time maximum and choose the number of time steps for each S value so that it perfectly matches the time maximum
nt = 500
print("Max time:", dt * nt)
print("Number of time steps:", nt)

# Compute wavefields for all S values
wavefields = [fdtd_1d(c, dt, dx, nx, nt, wavelength, gaussian_pulse) for wavelength in wavelengths]

# Combined animation function
def animate_combined_wavefields(wavefields, wavelengths, save=False):
    # Common time step based on the smallest dt
    fig, ax = plt.subplots(figsize=(10, 5))
    style = ["r-", "g--", "b--"]
    lines = [ax.plot(wavefields[i][0], style[i], label=f"b = {wavelength}")[0] for i, wavelength in enumerate(wavelengths)]
    ax.set_ylim(np.min(wavefields)*1.1, np.max(wavefields)*1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Amplitude")
    ax.legend()

    # Set the title to show the time
    ax.set_title("Time: 0")

    def update(frame):
        for i, line in enumerate(lines):
            line.set_ydata(wavefields[i][frame])
        ax.set_title(f"Time: {(frame) * dt:.2f} s")
        if not save and frame == len(wavefields[0])-1:
            ani.event_source.stop()
            plt.pause(5)
            ani.event_source.start()
        return lines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(len(wavefields[0])), blit=False, repeat=True, interval=100)
    if save:
        ani.save(ANIMATIONS + '1DGaussianPulse.gif', writer='pillow')
        print("Animation saved as 1DGaussianPulse.gif.")
    plt.show()

# Animate all wavefields
print("Animating...")
animate_combined_wavefields(wavefields, wavelengths, save=False)
