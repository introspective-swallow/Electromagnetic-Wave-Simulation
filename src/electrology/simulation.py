from .source import Source
from .grid import Grid
from .updater import Updater
from .boundary import Boundary_Update
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

ANIMATIONS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/animations/"
FIGS = "/home/gui/Repos/Electromagnetic-Wave-Simulation/figs/"

class Simulation1D():
    def __init__(self
                 , grid : Grid
                 , updater : Updater
                 , source : Source
                 , boundary : Boundary_Update = None
                 , c : float = 1
                 , mu_0 = None
                 , epsilon_0 = None
                 , imp_0 = None
                 , sigma : float = 0
                 , sigma_star : float = 0
                 ):
        self.grid = grid
        self.updater = updater
        self.source = source
        self.boundary = boundary
        if self.boundary is None:
            self.boundary = Boundary_Update()
        self.historyH = []
        self.historyE = []
        self.c = c
        self.mu_0 = mu_0
        self.epsilon_0 = epsilon_0
        self.imp_0 = imp_0
        self.sigma = sigma
        self.sigma_star = sigma_star

    def run(self):
        # Start simulation timer
        start_time = time.time()
        
        self.grid.init_simulation(self)
        #self.updater.init_updater(self.grid.dx, self.grid.dt, self.epsilon_0, self.mu_0, self.sigma, self.sigma_star)
        self.source.init_simulation(self)
        self.boundary.init_simulation(self.grid)

        for t in tqdm(range(self.grid.nt)):
            self.updater.updateH(self.grid)
            self.updater.updateE(self.grid)
            self.source.update_source(self.grid, t)
            self.boundary.update_boundary(self.grid)
            self.updateHistory()

        # End simulation timer
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    def updateHistory(self):
        self.historyE.append(self.grid.Ez.copy())
        self.historyH.append(self.grid.Hy.copy())

    def plot_frame(self, frame=-1, save=False, filename="", show=True):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axtext = fig.add_axes([0.1, 0.02, 0.86, 1.86])
        axtext.axis("off")
        disptime = axtext.text(0.5,0.5, str(0), ha="center", va="top", fontsize=12)
        disptime.set_text(f"Time: {self.grid.nt*self.grid.dt:.2e} s")

        axes[0].plot(range(self.grid.nx), self.historyE[frame], label='Ez')
        axes[1].plot(range(self.grid.nx-1), self.historyH[frame], label='Hy', color='orange')

        axes[0].set_xlim(0, self.grid.nx)
        axes[0].set_ylabel("Field amplitude")
        axes[0].legend()

        axes[1].set_xlim(0, self.grid.nx-1)
        axes[1].set_xlabel("Grid index")
        axes[1].set_ylabel("Field amplitude")
        axes[1].legend()

        # Add some space between the subplots to fit the time (title of the second subplot)
        plt.subplots_adjust(hspace=0.5)

        # Compute y-axis limits for each field
        Ez_min, Ez_max = np.min(self.historyE), np.max(self.historyE)
        Hy_min, Hy_max = np.min(self.historyH), np.max(self.historyH)

        # Add some padding to the y-axis limits
        Ez_min, Ez_max = 1.1 * Ez_min, 1.1 * Ez_max
        Hy_min, Hy_max = 1.1 * Hy_min, 1.1 * Hy_max

        axes[0].set_ylim(Ez_min, Ez_max)
        axes[1].set_ylim(Hy_min, Hy_max)

        if save or filename != "":
            if filename == "":
                filename = "fdtd_simulation"+str(time.time())
            plt.savefig(FIGS + filename+'.jpg')
            print(f"Animation saved as {filename}.jpg.")
        if show:
            plt.show()

    def plot_frame_Ez(self, frame=-1, save=False, filename="", show=True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(self.grid.nx), self.historyE[frame], label='Ez')
        ax.set_xlim(0, self.grid.nx)
        ax.set_ylabel("Field amplitude")
        ax.set_xlabel("Grid index")
        ax.set_ylim(1.1 * np.min(self.historyE), 1.1 * np.max(self.historyE))
        ax.legend()
        if save or filename != "":
            if filename == "":
                filename = "fdtd_simulation"+str(time.time())
            plt.savefig(FIGS + filename+'.png')
            print(f"Animation saved as {filename}.png.")
        if show:
            plt.show()

    def save_frames(self, skip_frames=1, foldername="Frames", frame_name="frame"):
        if not os.path.exists(FIGS+foldername):
            os.makedirs(FIGS+foldername)
        for i, frame in enumerate(range(0, self.grid.nt, skip_frames)):
            file_name = frame_name+str(i+1)
            self.plot_frame(frame, save=True, filename=foldername+"/"+file_name, show=False)

    def save_frames_Ez(self, skip_frames=1, foldername="Frames", frame_name="frame"):
        if not os.path.exists(FIGS+foldername):
            os.makedirs(FIGS+foldername)
        for i, frame in enumerate(range(0, self.grid.nt, skip_frames)):
            file_name = frame_name+str(i+1)
            self.plot_frame_Ez(frame, save=True, filename=foldername+"/"+file_name, show=False)


    def animate(self, frame_interval=10, reescale_fields=True, save=False, filename="", saveframes=False, foldername=""):
        # Reescale fields
        if reescale_fields:
            self.updater.re_escale_fields(self)

        # Animation setup
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axtext = fig.add_axes([0.1, 0.02, 0.86, 1.86])
        axtext.axis("off")
        disptime = axtext.text(0.5,0.5, str(0), ha="center", va="top", fontsize=12)

        Ez_line, = axes[0].plot([], [], label=r'$E_z$')
        Hy_line, = axes[1].plot([], [], label=r'$H_y$', color='orange')

        axes[0].set_xlim(0, self.grid.nx)
        axes[0].set_ylabel("Field amplitude")
        axes[0].legend()

        axes[1].set_xlim(0, self.grid.nx-1)
        axes[1].set_xlabel("Grid index")
        axes[1].set_ylabel("Field amplitude")
        axes[1].legend()

        # Add some space between the subplots to fit the time (title of the second subplot)
        plt.subplots_adjust(hspace=0.5)


        # Compute y-axis limits for each field
        Ez_min, Ez_max = np.min(self.historyE), np.max(self.historyE)
        Hy_min, Hy_max = np.min(self.historyH), np.max(self.historyH)

        # Add some padding to the y-axis limits
        Ez_min, Ez_max = 1.1 * Ez_min, 1.1 * Ez_max
        Hy_min, Hy_max = 1.1 * Hy_min, 1.1 * Hy_max

        axes[0].set_ylim(Ez_min, Ez_max)
        axes[1].set_ylim(Hy_min, Hy_max)

        # Animation function
        def update(frame):
            Ez_line.set_data(range(self.grid.nx), self.historyE[frame])
            Hy_line.set_data(range(self.grid.nx-1), self.historyH[frame])

            disptime.set_text(f"Time: {frame*self.grid.dt:.2e} s")
            return [Ez_line, Hy_line, disptime]

        ani = animation.FuncAnimation(fig, update, frames=self.grid.nt, interval=frame_interval, blit=True)
        plt.show()

        if save or filename != "":
            if filename == "":
                filename = "fdtd_simulation"+str(time.time())
            ani.save(ANIMATIONS+filename+".gif", writer="pillow", fps=20)
            print(f"Animation saved as {filename}.gif.")
        return fig, ani
    

    def plot_waterfall(self, field="Ez", title =None, save=False, filename="waterfall", interval=1, xlim=None, tlim=None):
        """
        Generate a watesrfall chart for the Ez field history.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if field == "Ez":
            # Create the data matrix for the field
            data = np.array(self.historyE)
        elif field == "Hy":
            data = np.array(self.historyH)

        # Extract time and space dimensions
        time_steps = len(data)
        space_indices = range(data.shape[1])

        # Restrict the plot to a specific time range
        if tlim is not None:
            data = data[tlim[0]:tlim[1]]
            time_steps = len(data)
        # Restrict the plot to a specific space range
        if xlim is not None:
            data = data[:, xlim[0]:xlim[1]]
            space_indices = range(data.shape[1])
        # Apply interval
        data = data[::interval]
        time_steps = len(data)

        # Generate offset values for the waterfall chart
        for t in range(time_steps):
            ax.plot(space_indices, data[t] + t, color="black", linewidth=0.8)
        
        if title == None:
            title = field

        # Add labels and grid
        ax.set_xlabel("Space (spatial index)")
        ax.set_ylabel("Time (frame number)")
        ax.set_title(f"Waterfall Chart of {title} Field")
        ax.grid(visible=True, linestyle="--", linewidth=0.5)

        # Save or show
        if save or filename != "":
            if filename == "":
                filename = "waterfall_"+field

            plt.savefig(FIGS + filename + '.png')
            print(f"Waterfall chart saved as {filename}.png.")
        plt.show()

class Simulation2D():
    def __init__(self
                 , grid : Grid
                 , updater : Updater
                 , source : Source
                 , boundary : None
                 , c : float = 1
                 , mu_0 = None
                 , epsilon_0 = None
                 , imp_0 = None
                 , sigma : float = 0
                 , sigma_star : float = 0
                 ):
        self.grid = grid
        self.updater = updater
        self.source = source
        self.boundary = boundary
        if self.boundary is None:
            self.boundary = Boundary_Update()
        self.historyEx = []
        self.historyEz = []
        self.historyHy = []
        self.c = c
        self.mu_0 = mu_0
        self.epsilon_0 = epsilon_0
        self.imp_0 = imp_0
        self.sigma = sigma
        self.sigma_star = sigma_star

    def run(self):
        # Start simulation timer
        start_time = time.time()
        
        self.grid.init_simulation(self)
        #self.updater.init_updater(self.grid.dx, self.grid.dt, self.epsilon_0, self.mu_0, self.sigma, self.sigma_star)
        self.source.init_simulation(self)
        self.boundary.init_simulation(self.grid)

        for t in tqdm(range(self.grid.nt)):
            self.updater.updateHy(self.grid)
            self.updater.updateEx(self.grid)
            self.updater.updateEz(self.grid)
            self.source.update_source(self.grid, t)
            self.boundary.update_boundary(self.grid)
            self.updateHistory()

        # End simulation timer
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    def updateHistory(self):
        self.historyEx.append(self.grid.Ex.copy())
        self.historyEz.append(self.grid.Ez.copy())
        self.historyHy.append(self.grid.Hy.copy())

    def plot_frame(self, frame=-1, field_names=[], save=False, filename="", show=True):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axtext = fig.add_axes([0.1, 0.02, 0.86, 1.86])
        axtext.axis("off")
        disptime = axtext.text(0.5,0.5, str(0), ha="center", va="top", fontsize=12)
        disptime.set_text(f"Time: {self.grid.nt*self.grid.dt:.2e} s")

        fields = [self.historyEx, self.historyEz, self.historyHy]

        if len(field_names) == 0:
            field_names = [r"$E_x$", r"$E_z$", r"$H_y$"]

        # Use 90th percentile to set color range
        color_ranges = [
            (-np.percentile(field, 99), np.percentile(field, 99)) for field in fields
        ]
        colorbars = []
        imgs = []

        for i, ax in enumerate(axs):
            im = ax.imshow(fields[i][frame]
                      , vmin=color_ranges[i][0]
                      , vmax=color_ranges[i][1]
                      , cmap="RdBu"
                      , interpolation="nearest")
            imgs.append(im)
            ax.set_title(field_names[i])
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            cbar = plt.colorbar(im, ax=ax, orientation="vertical")
            colorbars.append(cbar)

        if save or filename != "":
            if filename == "":
                filename = "2D_simulation"+str(time.time())
            plt.savefig(FIGS+filename+'.png')
            print(f"Frame saved as {filename}.png.")
        if show:
            plt.show()

    def save_frames(self, skip_frames=1, foldername="Frames", frame_name="frame"):
        if not os.path.exists(FIGS+foldername):
            os.makedirs(FIGS+foldername)
        for i, frame in enumerate(range(0, self.grid.nt, skip_frames)):
            file_name = frame_name+str(i+1)
            self.plot_frame(frame, save=True, filename=foldername+"/"+file_name, show=False)

    def animate(self, frame_interval=10, reescale_fields=False, save=False, filename="", time_colormap=False):
        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # Add a text annotation for simulation time at the center bottom of the figure
        axtext = fig.add_axes([0.1, 0.02, 0.8, 0.07])
        axtext.axis("off")
        time = axtext.text(0.5,0.5, str(0), ha="center", va="bottom", fontsize=12)


        if reescale_fields:
            fields = self.updater.re_escale_fields(self, replace=False)
        else:
            fields = [self.historyEx, self.historyEz, self.historyHy]
        
        titles = [r"$E_x$", r"$E_z$", r"$H_y$"]


    
        # Use 90th percentile to set color range
        color_ranges = [
            (np.percentile(field, 1), np.percentile(field, 99)) for field in fields
        ]
    
        
        images = []
        colorbars = []

        # Initialize plots
        for i, ax in enumerate(axs):
            img = ax.imshow(
                fields[i][0],
                vmin=color_ranges[i][0],
                vmax=color_ranges[i][1],
                cmap="RdBu",
                interpolation="nearest",
            )
            images.append(img)
            ax.set_title(titles[i])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = plt.colorbar(img, ax=ax, orientation="vertical")
            colorbars.append(cbar)

        # Update function for animation
        def update(n):
            for i, img in enumerate(images):
                img.set_data(fields[i][n])
                #img.set_clim(vmin=color_ranges[i][0], vmax=color_ranges[i][1])
            time.set_text(f"Time: {n*self.grid.dt*1e9:.2f} ns")
            return [*images, time]

        # Animation with progress bar
        print("Generating animation...")
        ani = animation.FuncAnimation(fig, update, frames=range(self.grid.nt), interval=frame_interval, blit=True)
        if save or filename != "":
            if filename == "":
                filename = "2D_simulation"+str(time.time())
            ani.save(ANIMATIONS+filename+".gif", writer="pillow", fps=20)
            print(f"Animation saved as {filename}.gif.")

        plt.show()
