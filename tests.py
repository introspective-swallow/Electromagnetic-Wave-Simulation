from simulation import Simulation1D
from grid import Grid
from updater import Updater
from source import GaussianPulse, RickerWavelet
from boundary import ABC_1order, ABC_2order, Boundary_Update, ABC_1orderLeft
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
nx = 100  # Number of spatial grid points
nt = 50  # Number of time steps
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
src = GaussianPulse(source_position=50, source_peak_timestep=10, source_width=2)
#src = RickerWavelet(source_position=50, wavelength_dx=20, delay_multiple=50)

bdr = ABC_2order()

a = Simulation1D(g, upd, src, bdr, **const)

a.run()
print(a.source.source)

a.animate(frame_interval=10, reescale_fields=True)
#a.plot_last_frame()
# a.plot_waterfall(field="Ez", title=r"$E_3^-$", filename="Ez_waterfall")
# a.plot_waterfall(field="Hy", title=r"$H_2^-$", filename="Hy_waterfall")
