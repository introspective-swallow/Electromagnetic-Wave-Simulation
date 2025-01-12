import numpy as np

class Grid():
    def __init__(self
                 , nx
                 , nt
                 , L
                 , cdtds
                 , eps = None
                 , mu = None
                 , sgm = None
                 , sgm_m = None
                 ):
        self.nx = nx
        self.nt = nt
        self.L = L
        self.cdtds = cdtds
        self.dx = L / nx
        self.eps = eps
        self.mu = mu
        self.sgm = sgm
        self.sgm_m = sgm_m

    def init_simulation(self, sim):
        # Initialize fields
        self.Ez = np.zeros(self.nx)
        self.Hy = np.zeros(self.nx-1)

        # Get the time step
        self.c = sim.c
        self.dt = self.cdtds * self.dx / self.c

        # Initialize coefficients
        if self.eps is None:
            self.eps = np.ones(self.nx) * sim.epsilon_0
        if self.mu is None:
            self.mu = np.ones(self.nx-1) * sim.mu_0
        if self.sgm is None:
            self.sgm = np.ones(self.nx) * sim.sigma
        if self.sgm_m is None:
            self.sgm_m = np.ones(self.nx-1) * sim.sigma_star

        self.cb = (self.dt / self.eps) / (1 + self.dt * self.sgm / (2 * self.eps))
        self.cb /= self.dx # Add dx to the denominator
        self.ca = (1 - self.dt * self.sgm / (2 * self.eps)) / (1 + self.dt * self.sgm / (2 * self.eps))
        self.db = (self.dt / self.mu) / (1 + self.dt * self.sgm_m / (2 * self.mu))
        self.db /= self.dx # Add dx to the denominator
        self.da = (1 - self.dt * self.sgm_m / (2 * self.mu)) / (1 + self.dt * self.sgm_m / (2 * self.mu))

        assert len(self.cb) == self.nx
        assert len(self.ca) == self.nx
        assert len(self.db) == self.nx-1
        assert len(self.da) == self.nx-1

        
class Grid2D():
    def __init__(self
                 , dx
                 , nt
                 , L1
                 , L2
                 , cdtds
                 , eps = None
                 , mu = None
                 , sgm = None
                 , sgm_m = None
                 ):
        
        self.dx = dx
        self.nt = nt
        self.L1 = L1
        self.L2 = L2
        self.cdtds = cdtds
        self.nx1 = int(L1 / dx)
        self.nx2 = int(L2 / dx)
        self.eps = eps
        self.mu = mu
        self.sgm = sgm
        self.sgm_m = sgm_m

    def init_simulation(self, sim):
        # Initialize fields
        self.Hx = np.zeros((self.nx1, self.nx2))
        self.Hy = np.zeros((self.nx1, self.nx2)) 
        self.Hz = np.zeros((self.nx1, self.nx2))
        self.Ex = np.zeros((self.nx1, self.nx2))
        self.Ey = np.zeros((self.nx1, self.nx2))
        self.Ez = np.zeros((self.nx1, self.nx2))

        # Get the time step using the Courant number for the 2D case
        self.c = sim.c
        self.dt = self.cdtds * ( self.dx ) / (2 * self.c)
        print(self.dt)

        # Initialize coefficients
        if self.eps is None:
            self.eps = np.ones((self.nx1, self.nx2)) * sim.epsilon_0
        if self.mu is None:
            self.mu = np.ones((self.nx1, self.nx2)) * sim.mu_0
        if self.sgm is None:
            self.sgm = np.ones((self.nx1, self.nx2)) * sim.sigma
        if self.sgm_m is None:
            self.sgm_m = np.ones((self.nx1, self.nx2)) * sim.sigma_star

        self.cb = (self.dt / self.eps) / (1 + self.dt * self.sgm / (2 * self.eps))
        self.cb /= self.dx
        self.ca = (1 - self.dt * self.sgm / (2 * self.eps)) / (1 + self.dt * self.sgm / (2 * self.eps))
        
        self.db = (self.dt / self.mu) / (1 + self.dt * self.sgm_m / (2 * self.mu))
        self.db /= self.dx
        self.da = (1 - self.dt * self.sgm_m / (2 * self.mu)) / (1 + self.dt * self.sgm_m / (2 * self.mu))

        assert self.ca.shape == (self.nx1, self.nx2)
        assert self.cb.shape == (self.nx1, self.nx2)
        assert self.da.shape == (self.nx1, self.nx2)
        assert self.db.shape == (self.nx1, self.nx2)

    def summary(self):
        print(f"Grid size: {self.nx1} x {self.nx2}")
        print(f"Time steps: {self.nt}")
        print(f"dx: {self.dx}")
        print(f"dt: {self.dt}")
        print(f"cdtds: {self.cdtds}")
        print(f"Courant number: {2 * self.c * self.dt / self.dx}")
        print(f"Material properties:")
        print(f"epsilon: {self.eps}")
        print(f"mu: {self.mu}")
        print(f"sigma: {self.sgm}")
        print(f"sigma_m: {self.sgm_m}")