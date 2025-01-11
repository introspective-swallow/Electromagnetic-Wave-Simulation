import numpy as np

c = 1  # Speed of light (m/s)
mu_0 = 377  # Permeability of free space (H/m)
epsilon_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)

class Boundary_Update():
    def __init__(self):
        pass
    def init_simulation(self, grid):
        pass
    def update_boundary(self, grid):
        pass

class ABC_1orderLeft(Boundary_Update):
    def __init__(self):
        super().__init__()
    def init_simulation(self, grid):
        temp = grid.cdtds/(np.sqrt(mu_0*epsilon_0)*c)
        self.mult_constant = (temp - 1)/(temp + 1)
        self.oldEzLeft = 0

    def update_boundary(self, grid):
        grid.Ez[0] = self.oldEzLeft + self.mult_constant * (grid.Ez[1]-grid.Ez[0])

        self.oldEzLeft = grid.Ez[1]
    
class ABC_1order(Boundary_Update):
    def __init__(self):
        super().__init__()

    def init_simulation(self, grid):
        temp = grid.cdtds/(np.sqrt(mu_0*epsilon_0)*c)
        self.mult_constant = (temp - 1)/(temp + 1)
        self.oldEzLeft = 0
        self.oldEzRight = 0

    def update_boundary(self, grid):
        grid.Ez[0] = self.oldEzLeft + self.mult_constant * (grid.Ez[1]-grid.Ez[0])
        grid.Ez[-1] = self.oldEzRight + self.mult_constant * (grid.Ez[-2]-grid.Ez[-1])

        self.oldEzLeft = grid.Ez[1]
        self.oldEzRight = grid.Ez[-2]

class ABC_2order(Boundary_Update):
    def __init__(self):
        super().__init__()

    def init_simulation(self, grid):
        """
        Initializes the ABC boundary conditions for a 2nd order ABC.
        
        oldFieldMatrix: np.array
            A matrix that stores the previous field values at the boundaries,
            for the first and last three values of the Ez field:
                First row: left boundary at time step n-1
                Second row: right boundary at time step n-1
                Third row: left boundary at time step n-2
                Fourth row: right boundary at time step n-2
            (where current time step is n)
        """
        self.S = grid.cdtds/(np.sqrt(mu_0*epsilon_0)*c)
        self.oldEzLeft = np.zeros((2, 3))
        self.oldEzRight = np.zeros((2, 3))

    def update_boundary(self, grid):

        grid.Ez[0] = -1 / (1/self.S + 2 + self.S) * (
            (1/self.S - 2 + self.S) * (grid.Ez[2] + self.oldEzLeft[1, 0]) +
            2 * (self.S - 1/self.S) * (self.oldEzLeft[0, 0] + self.oldEzLeft[0, 2] - grid.Ez[1] - self.oldEzLeft[1, 1]) -
            4 * (1/self.S + self.S) * self.oldEzLeft[0, 1]
        ) - self.oldEzLeft[1, 2]

        grid.Ez[-1] = -1 / (1/self.S + 2 + self.S) * (
            (1/self.S - 2 + self.S) * (grid.Ez[-3] + self.oldEzRight[1, -1]) +
            2 * (self.S - 1/self.S) * (self.oldEzRight[0, -1] + self.oldEzRight[0, -3] - grid.Ez[-2] - self.oldEzRight[1, -2]) -
            4 * (1/self.S + self.S) * self.oldEzRight[0, -2]
        ) - self.oldEzRight[1, -3]

        # Update old left boundary values
        self.oldEzLeft[1,:] = self.oldEzLeft[0,:]
        self.oldEzLeft[0,:] = grid.Ez[:3]

        # Update old right boundary values
        self.oldEzRight[1,:] = self.oldEzRight[0,:]
        self.oldEzRight[0,:] = grid.Ez[-3:]
        
