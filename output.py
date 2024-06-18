import matplotlib.pyplot as plt
import numpy as np


class Output:
    """
    Stores the results in the attribute `results`.
    Includes methods for plotting.
    """

    def __init__(self, days: int, num_locations: int) -> None:
        # `3` indicates number of compartments
        self.results = np.zeros((days, num_locations, 3))
        self.num_locations = num_locations

    def plot_infected(self) -> None:
        # plot infected compartments for all locations together in one plot
        plt.plot(self.results[:, :, 1])
        # add legend to the plot
        plt.legend(range(self.num_locations))
        # add title to the plot
        plt.title("Infected compartments for all locations")

    def plot_susceptible(self) -> None:
        # plot susceptible compartments for all locations together in one plot
        plt.plot(self.results[:, :, 0])
        # add legend to the plot
        plt.legend(range(self.num_locations))
        # add title to the plot
        plt.title("Susceptible compartments for all locations")
