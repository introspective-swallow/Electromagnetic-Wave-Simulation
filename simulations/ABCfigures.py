from electrology.simulation import Simulation1D
from electrology.grid import Grid
from electrology.updater import Updater
from electrology.source import GaussianPulse, RickerWavelet
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
nx = 100  # Number of spatial grid points
nt = 150  # Number of time steps
cdtds = 0.5 # Courant number

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
src = GaussianPulse(source_position=20, source_peak_timestep=20, source_width=5)
bdr = ABC_1order()
a = Simulation1D(g, upd, src, bdr, **const)

a.run()
a.animate(frame_interval=10, reescale_fields=True)
a.plot_frame_Ez(10, filename="FigABCorder1_f10")
a.plot_frame_Ez(50, filename="FigABCorder1_f50")
a.plot_frame_Ez(60, filename="FigABCorder1_f60")
a.plot_frame_Ez(70, filename="FigABCorder1_f70")

bdr = ABC_2order()
a = Simulation1D(g, upd, src, bdr, **const)
a.run()
a.plot_frame_Ez(10, filename="FigABCorder2_f10")
a.plot_frame_Ez(50, filename="FigABCorder2_f50")
a.plot_frame_Ez(60, filename="FigABCorder2_f60")
a.plot_frame_Ez(70, filename="FigABCorder2_f70")

#a.plot_waterfall(field="Ez", title=r"$E_z$", filename="FigABCorder1", xlim=[0, 30], interval=20)
# a.plot_waterfall(field="Hy", title=r"$H_2^-$", filename="Hy_waterfall")
