from electrology.simulation import Simulation2D
from electrology.grid import Grid2D
from electrology.updater import UpdaterModeTEy
from electrology.source import GaussianPulse2D, GaussianPulse, Sinusoidal2D
from electrology.boundary import Boundary_Update
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

L = 1  # Length of the simulation domain (m)
dx = L/100  # Number of spatial grid points
nt = 100  # Number of time steps
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


g = Grid2D(dx, nt, L, L, cdtds, **material)
upd = UpdaterModeTEy()
src = Sinusoidal2D(source_position=[50, 50], source_frequency=1e9, source_spatial_amplitude=1)
#src = GaussianPulse2D(source_position=[50, 50], source_peak_timestep=10, source_temp_width=2, source_spatial_width=10)
#src = RickerWavelet(source_position=50, wavelength_dx=20, delay_multiple=50)
#src = GaussianPulse(source_position=(50,50), source_peak_timestep=10, source_width=4)
bdr = Boundary_Update()

a = Simulation2D(g, upd, src, bdr, **const)

a.run()

a.plot_frame(filename="FigTEy_mode", field_names=[r"$E_1$", r"$E_3$", r"$H_2$"])
