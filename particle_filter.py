from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from tau_leap import SIR_tau_leap


@dataclass
class ParticleFilterParams:
    """
    This class should be deprecated and all usage
    should be replaced with the Simulation Object.
    """

    num_particles: int
    num_locations: int
    population: np.ndarray
    movement: np.ndarray
    mov_ratio: np.ndarray
    results: np.ndarray
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = self.results[:, :, 1]


@dataclass
class ParticleFilterState:
    """
    This object models an array of particles, including the
    current beta values and weights.

    It also stores the posterior distributions and
    implements methods for keeping track of weights and resampling indices
    at each time step for analysis.
    """

    particles: np.ndarray
    betas: np.ndarray
    weights: np.ndarray
    m_post: np.ndarray
    beta_post: np.ndarray
    all_weights: list = field(default_factory=list)
    all_resampling_indices: list = field(default_factory=list)

    def save_weights(self) -> None:
        self.all_weights.append(self.weights.copy())

    def get_all_weights_as_array(self) -> np.ndarray:
        # Convert the list of arrays to a 2D numpy array
        return np.vstack(self.all_weights)

    def save_resampling_indices(self, indices: np.ndarray) -> None:
        self.all_resampling_indices.append(indices)

    def get_all_resampling_indices_as_array(self) -> np.ndarray:
        return np.vstack(self.all_resampling_indices)


def run_log_particle_filter(params: ParticleFilterParams) -> ParticleFilterState:
    """
    Runs the log domain particle filter
    according to the parameters passed in.
    """
    np.random.seed(13)
    state = initialize_particles(params)

    for t in range(params.results.shape[0]):
        print(f"Iteration: {t + 1} \r")

        if t != 0:
            update_particles(params, state, t)

        compute_log_weights(params, state, t)
        state.save_weights()
        log_resample_algo(state, t)
        perturb_betas(state, params.num_locations, t)

        posterior_update(state, t)

    return state


def run_linear_particle_filter(params: ParticleFilterParams) -> ParticleFilterState:
    """
    Runs the linear domain particle filter
    according to the parameters passed in.
    """
    np.random.seed(13)
    state = initialize_particles(params)

    for t in range(params.results.shape[0]):
        print(f"Iteration: {t + 1} \r")

        if t != 0:
            update_particles(params, state, t)

        compute_linear_weights(params, state, t)
        state.save_weights()
        linear_resample_algo(state, t)
        perturb_betas(state, params.num_locations, t)

        posterior_update(state, t)

    return state


def initialize_particles(params: ParticleFilterParams) -> ParticleFilterState:
    """
    Initializes the PF state class.
    Returns a PFState object that contains particles and associated data.
    """
    np.random.seed(47)  # to ensure particle init is always the same

    particles = np.zeros(
        (params.num_particles, params.num_locations, 3, params.results.shape[0])
    )
    betas = []

    for i in range(params.num_particles):
        beta = np.zeros((params.num_locations, params.results[:, :, 1].shape[0]))
        for loc in range(params.num_locations):
            I = np.random.randint(0, 5)
            S = params.population[loc] - I
            R = 0
            particles[i, loc, :, 0] = S, I, R

            beta[loc, 0] = np.random.uniform(0.6, 0.8)
        betas.append(beta)

    particles = np.array(particles)
    betas = np.array(betas)
    weights = np.ones(params.num_particles)
    m_post = np.zeros((params.results.shape[0], params.num_locations, 3))
    beta_post = np.zeros((params.results.shape[0], params.num_locations))

    return ParticleFilterState(particles, betas, weights, m_post, beta_post)


def linear_resample_algo(state: ParticleFilterState, t) -> None:
    num_particles = state.weights.size
    resampling_indices = np.zeros(num_particles, dtype=int)
    cdf = np.cumsum(state.weights)
    u = np.random.uniform(0, 1 / num_particles)

    i = 0
    for j in range(num_particles):
        r = u + j / num_particles
        while r > cdf[i]:
            i += 1
        resampling_indices[j] = i

    # save indices to compare to log domain
    state.save_resampling_indices(resampling_indices)

    resample_particles_and_params(resampling_indices, state, t)


def log_resample_algo(state: ParticleFilterState, t: int) -> None:
    num_particles = state.weights.size
    resampling_indices = np.zeros(num_particles, dtype=int)
    log_cdf = jacobian(state.weights)
    u = np.random.uniform(0, 1 / num_particles)

    i = 0
    for j in range(num_particles):
        r = np.log(u + 1 / num_particles * j)
        while r > log_cdf[i]:
            i += 1
        resampling_indices[j] = i

    # save indices to compare to lin domain
    state.save_resampling_indices(resampling_indices)

    resample_particles_and_params(resampling_indices, state, t)


def resample_particles_and_params(
    resampling_indices: np.ndarray, state: ParticleFilterState, t: int
) -> None:
    """
    Helper function for lin and log resampling algos.
    Performs the actual resampling after the indices are chosen.
    """
    state.particles = state.particles[resampling_indices]
    state.betas = state.betas[resampling_indices]


def perturb_betas(state: ParticleFilterState, n: int, t: int):
    """
    Perturbs the beta values with a multi-variate normal distribution.

    Args:
        state: ParticleFilterState class
        n: number of particles
        t: time steps
    """
    for i in range(state.particles.shape[0]):
        state.betas[i, :, t] = np.exp(
            np.random.multivariate_normal(
                np.log(state.betas[i, :, t]), 0.001 * np.eye(n)
            )
        )





def posterior_update(state, t):
    state.m_post[t, :, :] = np.average(
        state.particles[:, :, :, t], axis=0, weights=state.weights
    )
    state.beta_post[t, :] = np.average(
        state.betas[:, :, t], axis=0, weights=state.weights
    )





def compute_log_weights(
    params: ParticleFilterParams, state: ParticleFilterState, t: int
) -> None:
    """Compute particle weights in log domain."""
    state.weights = np.zeros(params.num_particles)

    for index in range(params.num_particles):
        for loc in range(params.num_locations):
            state.weights[index] += norm.logpdf(
                params.data[t, loc], state.particles[index, loc, 1, t], 100_000
            )

    state.weights = log_norm(state.weights)


def compute_linear_weights(
    params: ParticleFilterParams, state: ParticleFilterState, t: int
) -> None:
    """Compute particle weights in linear domain."""

    for i in range(params.num_particles):
        for loc in range(params.num_locations):
            state.weights[i] *= norm.pdf(
                params.data[t, loc], state.particles[i, loc, 1, t], 100_000
            )
    state.weights /= np.sum(state.weights)


def update_particles(
    params: ParticleFilterParams, state: ParticleFilterState, t: int
) -> None:
    """
    Update the state of all particles at time t using parallel processing.

    Args:
        params (Any): Parameters for the SIR model, including population and movement.
        state (Any): Current state of the particles and betas.
        t (int): Current time step.
    """
    for i in range(params.num_particles):
        state.particles[i, :, :, t] = SIR_tau_leap(
            params.population,
            params.movement,
            params.mov_ratio,
            state.particles[i, :, :, t - 1],
            state.betas[i, :, t - 1],
        )[:, :, -1]
        state.betas[i, :, t] = state.betas[i, :, t - 1]


def jacobian(δ: np.ndarray) -> np.ndarray:
    """
    The jacobian logarithm, used in log likelihood normalization and
    resampling processes.

    Args:
        δ: An array of values to sum

    Returns:
        The vector of partial sums of δ.
    """
    n = len(δ)
    Δ = np.zeros(n)
    Δ[0] = δ[0]
    for i in range(1, n):
        Δ[i] = max(δ[i], Δ[i - 1]) + np.log(1 + np.exp(-1 * np.abs(δ[i] - Δ[i - 1])))
    return Δ


def log_norm(log_weights):
    """
    Normalizes the probability space using the jacobian logarithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
