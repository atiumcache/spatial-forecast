import numpy as np


def gen_population(n: int) -> np.ndarray:
    """
    Takes the number of locations n and generate a numpy array
    of n integers representing the population of each location.
    The population of each location is a random number between 5000 and 10000.
    The function returns the numpy array of population.

    Args:
        n: Number of locations/nodes.
    """
    return np.random.randint(5_000_000, 10_000_000, n)


def gen_movement(population: np.ndarray, min_move=0.03, max_move=0.08, mov=1, chain=1):
    """
    Takes the population of n locations
    and generates an n by n numpy array of integers representing the movement of people from each location.
    Movement should be a random number between 3 and 8 percent of the total population of the location.
    Note that the movement happens from column to row.
    """
    movement = np.zeros((len(population), len(population)))
    if mov == 1:
        for i in range(len(population)):
            for j in range(len(population)):
                movement[i][j] = np.random.randint(
                    min_move * population[j], max_move * population[j]
                )
        np.fill_diagonal(movement, 0)
        # if Chain==1, only allow movement from i to i+1, everything else is 0
    if chain == 1:
        movement = np.zeros((len(population), len(population)))
        for i in range(1, len(population)):
            movement[i][i - 1] = np.random.randint(
                min_move * population[i - 1], max_move * population[i - 1]
            )

    return movement


def gen_mov_ratio(movement, population):
    """
    Takes the movement matrix and divides each column by the population
    to generate an n by n np.array of floats.
    """
    return movement @ np.linalg.inv(np.diag(population))


def gen_initial_cond(population) -> np.array:
    """
    Takes the population of n locations and generate a numpy array
    of n times 3 integers representing the initial condition of SIR model for each location.
    The initial condition for I is a random number between 0 and 1.
    or 0.01 to 0.03 of the total population of the location.
    """
    n = len(population)
    # I = np.random.randint(0.01*population, 0.03*population)
    I = 0 * np.random.randint(0, 5, n)
    S = population - I
    R = np.zeros(n)
    # return transpose of the array
    return np.array([S, I, R]).T
