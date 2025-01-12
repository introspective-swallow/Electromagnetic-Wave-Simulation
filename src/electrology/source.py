import numpy as np
class Source():
    def __init__(self):
        pass

    def init_simulation(self):
        raise NotImplementedError
    
    def update_source(self, t):
        raise NotImplementedError
    
class PointSource():
    def __init__(self):
        super().__init__()
        self.source_position = None
        self.source_field = None

    def update_source(self, grid, t):
        grid.Ez[self.source_position] += self.source_field[t]

class SmoothSource():
    def __init__(self):
        super().__init__()
        self.source_position = None
        self.source_field = None
        self.gaussian_envelope = None

    def update_source(self, grid, t):
        grid.Ez[:] += self.source_field[t] * self.gaussian_envelope

class GaussianPulse(PointSource):
    def __init__(self, source_position, source_peak_timestep, source_width):
        super().__init__()
        self.nt = None
        self.source_position = source_position
        self.source_peak_timestep = source_peak_timestep
        self.source_width = source_width

    def init_simulation(self, sim):
        tspace = np.arange(sim.grid.nt)
        self.source_field = np.exp(-((tspace - self.source_peak_timestep) ** 2) / (2 * self.source_width ** 2))

class SmoothSinusoidal1D(SmoothSource):
    def __init__(self, source_position, source_frequency, source_width):
        super().__init__()
        self.nt = None
        self.source_position = source_position
        self.source_frequency = source_frequency
        self.source_width = source_width

    def init_simulation(self, sim):
        tspace = np.arange(sim.grid.nt) * sim.grid.dt
        self.source_field = np.sin(2 * np.pi * self.source_frequency * tspace)
        grid_indices = np.arange(sim.grid.nx)
        self.gaussian_envelope = np.exp(-((grid_indices - self.source_position) ** 2) / (2 * self.source_width ** 2))

class RickerWavelet(PointSource):
    def __init__(self, source_position, source_peak_frequency=1, wavelength_dx=None, temporal_delay=1, delay_multiple=None):
        super().__init__()
        self.nt = None
        self.source_position = source_position
        
        # Initialize temporal delay
        if delay_multiple is not None:
            self.temporal_delay = delay_multiple / source_peak_frequency
        elif temporal_delay is not None:
            self.temporal_delay = temporal_delay
        else:
            raise ValueError("Temporal delay not specified")
        self.source_peak_frequency = source_peak_frequency
        self.wavelength_dx = wavelength_dx
        
        
    def init_simulation(self, sim):
        # Initialize peak frequency
        if self.wavelength_dx is not None:
            self.source_peak_frequency = sim.c / (self.wavelength_dx) # Check why not c / (wavelength * dx)
        elif self.source_peak_frequency is not None:
            self.source_peak_frequency = self.source_peak_frequency
        else:
            raise ValueError("Source peak frequency not specified")
        
        # Initialize source
        self.nt = sim.grid.nt
        tspace = np.arange(self.nt)
        self.source_field = (1 - 2 * (np.pi * self.source_peak_frequency * (tspace - self.temporal_delay)) ** 2) * np.exp(- (np.pi * self.source_peak_frequency * (tspace - self.temporal_delay)) ** 2)

class Source2D():
    def __init__(self):
        pass

    def init_simulation(self):
        raise NotImplementedError
    
    def update_source(self, grid, t):
        grid.Ez += self.source_field[t]
    
class GaussianPulse2D(Source2D):
    def __init__(self, source_position, source_peak_timestep, source_temp_width, source_spatial_width):
        super().__init__()
        self.nt = None
        self.source_position = source_position
        self.source_peak_timestep = source_peak_timestep
        self.source_temp_width = source_temp_width
        self.source_spatial_width = source_spatial_width

    def init_simulation(self, sim):
        # Create temporal profile of the source
        tspace = np.arange(sim.grid.nt)
        source_temporal = np.exp(-((tspace - self.source_peak_timestep) ** 2) / (2 * self.source_temp_width ** 2))
        import matplotlib.pyplot as plt
        plt.plot(tspace, source_temporal)
        # Create spatial profile of the source
        x, y = np.meshgrid(np.arange(sim.grid.nx1), np.arange(sim.grid.nx2), indexing="ij")

        spatial_envelope = np.exp(
            -((x - self.source_position[0]) ** 2 + (y - self.source_position[1]) ** 2) / (2 * self.source_spatial_width**2)
        )

        # Create the source field
        self.source_field = np.zeros((sim.grid.nt, sim.grid.nx1, sim.grid.nx2))
        for n in range(sim.grid.nt):
            self.source_field[n] = spatial_envelope * source_temporal[n]

class Sinusoidal2D(Source2D):
    def __init__(self, source_position, source_frequency, source_spatial_amplitude):
        super().__init__()
        self.nt = None
        self.source_position = source_position
        self.source_frequency = source_frequency
        self.source_spatial_amplitude = source_spatial_amplitude

    def init_simulation(self, sim):
        # Create temporal profile of the source
        tspace = np.arange(sim.grid.nt)*sim.grid.dt
        source_temporal = np.sin(2 * np.pi * self.source_frequency * tspace)
        
        # Create spatial profile of the source
        x, y = np.meshgrid(np.arange(sim.grid.nx1), np.arange(sim.grid.nx2), indexing="ij")

        spatial_envelope = np.exp(
            -((x - self.source_position[0]) ** 2 + (y - self.source_position[1]) ** 2) / (2 * self.source_spatial_amplitude**2)
        )

        # Create the source field
        self.source_field = np.zeros((sim.grid.nt, sim.grid.nx1, sim.grid.nx2))
        for n in range(sim.grid.nt):
            self.source_field[n] = spatial_envelope * source_temporal[n]