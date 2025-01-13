from electrology.simulation import Simulation1D
from electrology.grid import Grid
from electrology.updater import Updater
from electrology.source import GaussianPulse, RickerWavelet
from electrology.boundary import ABC_1order, ABC_2order, Boundary_Update, ABC_1orderLeft
import numpy as np

# Simulation parameters
c = 1  # Speed of light (m/s)
mu_0 = 377  # Permeability of free space (H/m)
epsilon_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)
sigma = 0.0  # Conductivity (S/m)
sigma_star = 0.0  # Magnetic conductivity (S/m)

const = {
    "c": c,
    "mu_0": mu_0,
    "epsilon_0": epsilon_0,
}

L = 1.0  # Length of the simulation domain (m)
nx = 200  # Number of spatial grid points
nt = 150  # Number of time steps
cdtds = 1 # Courant number


def quadratic_grow(nx, max_cond=1):
    # Quadraticically increasing loss from 0 to max_loss
    loss = np.zeros(nx)
    for i in range(nx):
        loss[i] = max_cond*(i/nx)**2
    return loss

# PML layer
pml_length = 100

# Different permeability
mu = np.ones(nx-1) * mu_0

# Lossy
sgm = np.zeros(nx)
sgm[-pml_length:] = quadratic_grow(pml_length)

# Different permitivity
eps = np.ones(nx) * epsilon_0

# Magnetic loss
sgm_m = np.zeros(nx-1)
sgm_m[-pml_length+1:] = quadratic_grow(pml_length-1) * mu_0 / epsilon_0

material = {
    "mu": mu,
    "eps": eps,
    "sgm": sgm,
    "sgm_m": sgm_m
}


g = Grid(nx, nt, L, cdtds, **material)
upd = Updater()
src = GaussianPulse(source_position=50, source_peak_timestep=30, source_width=5)
#src = RickerWavelet(source_position=50, wavelength_dx=20, delay_multiple=50)

#bdr = ABC_2order()

a = Simulation1D(g, upd, src, ABC_1orderLeft(), **const)

a.run()

a.save_frames(5, "Fig1DPML", "PML")

# a.animate(frame_interval=10, reescale_fields=False)