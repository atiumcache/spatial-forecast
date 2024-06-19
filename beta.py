import numpy as np


def gen_beta(n: int, t: int) -> np.array:
    """
    Generates a numpy array of n by t floats representing the transmission rate in each pair of locations.
    The transmission rate has a seasonality pattern that is a normal distribution
    center at a random number between 0.1 and 0.3, variance 0.01, and cosine seasonality function that has period 365 days.
    """
    beta_mean = np.random.uniform(0.6, 0.8, n)
    # for each mean, generate a random normal distribution with variance 0.01
    # combine the seasonality pattern with the random normal distribution
    beta = np.array(
        [beta_mean[i] + 0.1 * np.cos(4 * np.pi * np.arange(t) / 365) for i in range(n)]
    )
    # inject some noise to beta
    # comment the following line to remove noise if needed
    # beta += np.random.normal(0, 0.01, (n,t))
    return beta


def switch_beta_value(
    current_value: float, high_value: float, low_value: float
) -> float:
    """
    Helper function for gen_step_beta()
    Switches the beta value from high to low or vice versa, at the period
    intervals defined in gen_step_beta().
    """
    return low_value if current_value == high_value else high_value


def gen_step_beta(n: int, t: int, period: int = 31) -> np.ndarray:
    """
    Generate a step-function beta.
    Switches value at each period.

    Args:
        n: number of locations
        t: number of time steps
        period: time step at which beta switches value

    Returns:
        Array of beta values; n x t dimensions.
    """
    # beta varies by location
    beta_mean = np.random.uniform(0.5, 0.7, n)

    beta = np.zeros((n, t))

    for i in range(n):
        # high value as the infection is ramping up
        high_value = beta_mean[i] + 0.08
        # low value as the infection dies down
        low_value = beta_mean[i] - 0.08

        current_value = high_value
        for day in range(t):
            if day % period == 0 and day != 0:
                current_value = switch_beta_value(current_value, high_value, low_value)
            beta[i, day] = current_value

    return beta
