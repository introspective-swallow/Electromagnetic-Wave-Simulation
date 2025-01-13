from electrology.simulation import Simulation1D
from electrology.grid import Grid
from electrology.updater import Updater
from electrology.source import GaussianPulse, RickerWavelet, SmoothSinusoidal1D
from electrology.boundary import ABC_1order, ABC_2order, Boundary_Update, ABC_1orderLeft
import numpy as np

# Simulation parameters
#c = 1  # Speed of light (m/s)
mu_0 = 1.25663706 * 10**-6  # Permeability of free space (H/m)
#epsilon_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)
epsilon_0 = 8.8541878188 *10**-12  # Permittivity of free space (F/m)
c = 1/np.sqrt(mu_0*epsilon_0)
imp_0 = np.sqrt(mu_0/epsilon_0)
print(imp_0)

sigma = 0.0  # Conductivity (S/m)
sigma_star = 0.0  # Magnetic conductivity (S/m)

const = {
    "c": c,
    "mu_0": mu_0,
    "epsilon_0": epsilon_0,
    "imp_0": imp_0
}

L = 1.0  # Length of the simulation domain (m)
nx = 1000  # Number of spatial grid points
nt = 600  # Number of time steps
cdtds = 1 # Courant number

mu = None
eps = None
sgm = None
sgm_m = None

material = {
    "mu": mu,
    "eps": eps,
    "sgm": sgm,
    "sgm_m": sgm_m
}


g = Grid(nx, nt, L, cdtds, **material)
upd = Updater()
src = SmoothSinusoidal1D(source_position=nx//2, source_frequency=1e9, source_width=20)
bdr = ABC_2order()

a = Simulation1D(g, upd, src, bdr, **const)

a.run()
a.animate(reescale_fields=False)
a.save_frames(skip_frames=5, foldername="FigSere", frame_name="Sere")