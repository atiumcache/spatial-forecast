from matplotlib import pyplot as plt

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Simulation:
    num_locations: int
    days: int
    population: np.ndarray = field(init=False)
    movement: np.ndarray = field(init=False)
    mov_ratio: np.ndarray = field(init=False)
    initial_cond: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)
    real_beta: np.ndarray = field(init=False)
    results: np.ndarray = field(init=False)

    def __post_init__(self):
        np.random.seed(2)
        self.population = self.gen_population(self.num_locations)
        self.movement = self.gen_movement(self.population, chain=1)
        self.mov_ratio = self.gen_mov_ratio(self.movement, self.population)
        self.initial_cond = self.gen_initial_cond(self.population)
        self.initial_cond[0, 1] = 5
        self.real_beta = self.gen_step_beta(self.num_locations, self.days)
        self.results = np.zeros((self.days, self.num_locations, 3))
        self.results[0, :, :] = self.initial_cond

    @staticmethod
    def gen_population(n: int) -> np.ndarray:
        """
        Generates a numpy array of n integers representing the population of each location.
        The population of each location is a random number between 5,000,000 and 10,000,000.

        Args:
            n: Number of locations.

        Returns:
            A numpy array of populations.
        """
        return np.random.randint(5_000_000, 10_000_000, n)

    @staticmethod
    def gen_movement(
        population: np.ndarray, min_move=0.03, max_move=0.08, mov=1, chain=1
    ):
        """
        Generates an n by n numpy array of integers representing the movement of people from each location.
        Movement is a random number between 3% and 8% of the total population of the location.
        Movement happens from column to row.

        Args:
            population: Array of populations for each location.
            min_move: Minimum movement ratio.
            max_move: Maximum movement ratio.
            mov: Flag for movement mode.
            chain: Flag for chain movement mode.

        Returns:
            A numpy array representing the movement matrix.
        """
        movement = np.zeros((len(population), len(population)))
        if mov == 1:
            for i in range(len(population)):
                for j in range(len(population)):
                    movement[i][j] = np.random.randint(
                        min_move * population[j], max_move * population[j]
                    )
            np.fill_diagonal(movement, 0)
        if chain == 1:
            movement = np.zeros((len(population), len(population)))
            for i in range(1, len(population)):
                movement[i][i - 1] = np.random.randint(
                    min_move * population[i - 1], max_move * population[i - 1]
                )

        return movement

    @staticmethod
    def gen_mov_ratio(movement, population):
        """
        Takes the movement matrix and divides each column by the population to generate an n by n numpy array of floats.

        Args:
            movement: Movement matrix.
            population: Population array.

        Returns:
            A numpy array representing the movement ratio matrix.
        """
        return movement @ np.linalg.inv(np.diag(population))

    @staticmethod
    def gen_initial_cond(population) -> np.array:
        """
        Generates the initial condition of the SIR model for each location.

        Args:
            population: Array of populations for each location.

        Returns:
            A numpy array of initial conditions with shape (n, 3).
        """
        n = len(population)
        I = 0 * np.random.randint(0, 5, n)
        S = population - I
        R = np.zeros(n)
        return np.array([S, I, R]).T

    @staticmethod
    def switch_beta_value(
        current_value: float, high_value: float, low_value: float
    ) -> float:
        """
        Switches the beta value from high to low or vice versa.

        Args:
            current_value: Current beta value.
            high_value: High beta value.
            low_value: Low beta value.

        Returns:
            The switched beta value.
        """
        return low_value if current_value == high_value else high_value

    def gen_step_beta(self, n: int, t: int, period: int = 31) -> np.ndarray:
        """
        Generates a step-function beta that switches value at each period.

        Args:
            n: Number of locations.
            t: Number of time steps.
            period: Time step at which beta switches value.

        Returns:
            A numpy array of beta values with shape (n, t).
        """
        beta_mean = np.random.uniform(0.5, 0.7, n)
        beta = np.zeros((n, t))

        for i in range(n):
            high_value = beta_mean[i] + 0.08
            low_value = beta_mean[i] - 0.08
            current_value = high_value

            for day in range(t):
                if day % period == 0 and day != 0:
                    current_value = self.switch_beta_value(
                        current_value, high_value, low_value
                    )
                beta[i, day] = current_value

        return beta

    def plot_mov_ratio(self):
        """Plot the movement ratio matrix using imshow."""
        plt.imshow(self.mov_ratio, cmap="bwr", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def plot_real_beta(self):
        """Plot the beta values for each location."""
        for i in range(self.num_locations):
            plt.plot(self.real_beta[i], label="Location " + str(i))
        plt.legend()
        plt.show()

    def run_sir_model(self, sir_tau_leap):
        """
        Run the SIR model using the provided SIR_tau_leap function.

        Args:
            sir_tau_leap: A function to run the SIR model.
        """
        for t in range(1, self.days):
            self.results[t, :, :] = sir_tau_leap(
                self.population,
                self.movement,
                self.mov_ratio,
                self.results[t - 1, :, :],
                self.real_beta[:, t],
            )[:, :, -1]

    def plot_infected(self) -> None:
        """PLot the infected compartment of the simulated data."""
        # plot infected compartments for all locations together in one plot
        plt.plot(self.results[:, :, 1])
        plt.legend(range(self.num_locations))
        plt.title("Infected compartments for all locations")

    def plot_susceptible(self) -> None:
        """Plot the susceptible compartment of the simulated data."""
        # plot susceptible compartments for all locations together in one plot
        plt.plot(self.results[:, :, 0])
        plt.legend(range(self.num_locations))
        plt.title("Susceptible compartments for all locations")
