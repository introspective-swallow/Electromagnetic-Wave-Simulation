import numpy as np

class Updater():
    def __init__(self):
        pass

    def init_updater(self, dx, dt, epsilon_0, mu_0, sigma, sigma_star):
        # Precompute coefficients for FDTD update equations
        # self.ce1 = (dt / epsilon_0) / (1 + dt * sigma / (2 * epsilon_0))
        # self.ce2 = (1 - dt * sigma / (2 * epsilon_0)) / (1 + dt * sigma / (2 * epsilon_0))

        # self.ch1 = (dt / mu_0) / (1 + dt * sigma_star / (2 * mu_0))
        # self.ch2 = (1 - dt * sigma_star / (2 * mu_0)) / (1 + dt * sigma_star / (2 * mu_0))
        pass

    def updateH(self, grid):
        for i in range(grid.nx-1):
            grid.Hy[i] = grid.db[i] * (grid.Ez[i + 1] - grid.Ez[i]) + grid.da[i] * grid.Hy[i]

    def updateE(self, grid):
        for i in range(1, grid.nx-1):
            grid.Ez[i] = grid.cb[i] * (grid.Hy[i] - grid.Hy[i - 1]) + grid.ca[i] * grid.Ez[i]

    def re_escale_fields(self, sim, H=None, E=None):
        if H is None:
            H = np.max(np.abs(np.array(sim.historyH)))
        if E is None:
            E = np.max(np.abs(np.array(sim.historyE)))

        print(f"Original max amplitudes of E: {E} and H: {H} normalized to 1")

        for Hy in range(len(sim.historyH)):
            sim.historyH[Hy] /= H
        for Ez in range(len(sim.historyE)):
            sim.historyE[Ez] /= E


class UpdaterModeTEy():
    def __init__(self):
        pass

    def init_updater(self):
        raise NotImplementedError

    def updateEx(self, grid):
        grid.Ex[:, 1:] = grid.ca[:, 1:] * grid.Ex[:, 1:] + grid.cb[:, 1:] * (grid.Hy[:, :-1] - grid.Hy[:, 1:])

    def updateEz(self, grid):
        grid.Ez[1:, :] = grid.ca[1:, :] * grid.Ez[1:, :] + grid.cb[1:, :] * (grid.Hy[1:, :] - grid.Hy[:-1, :])
    
    def updateHy(self, grid):
        grid.Hy[1:-1, 1:-1] = grid.da[1:-1, 1:-1] * grid.Hy[1:-1, 1:-1] + grid.db[1:-1, 1:-1] * (grid.Ez[2:, 1:-1] - grid.Ez[1:-1, 1:-1] + grid.Ex[1:-1, 1:-1] - grid.Ex[1:-1, 2:])
        
    def re_escale_fields(self, sim, weights=None, replace=False):
        if weights is None:
            Ex = np.max(np.abs(np.array(sim.historyEx)))
            Ez = np.max(np.abs(np.array(sim.historyEz)))
            Hy = np.max(np.abs(np.array(sim.historyHy)))

        print(f"Original max amplitudes of E_x:{Ex}, E_z:{Ez} and H_y:{Hy} normalized to 1")
        
        rescaledEx = []
        rescaledEz = []
        rescaledHy = []

        for t in range(len(sim.historyEx)):
            rescaledEx.append(sim.historyEx[t] / Ex)
        for t in range(len(sim.historyEz)):
            rescaledEz.append(sim.historyEz[t] / Ez)
        for t in range(len(sim.historyHy)):
            rescaledHy.append(sim.historyHy[t] / Hy)

        if replace:
            sim.historyEx = rescaledEx
            sim.historyEz = rescaledEz
            sim.historyHy = rescaledHy

        return rescaledEx, rescaledEz, rescaledHy
