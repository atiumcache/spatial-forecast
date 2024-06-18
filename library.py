import numpy as np
from scipy.stats import norm

from particle_filter import ParticleFilterParams, ParticleFilterState
from tau_leap import SIR_tau_leap


def gen_population(n):
    """
    Takes the number of locations n and generate a numpy array
    of n integers representing the population of each location.
    The population of each location is a random number between 5000 and 10000.
    The function returns the numpy array of population.
    """
    return np.random.randint(5_000_000, 10_000_000, n)


def gen_movement(population, min=0.03, max=0.08, Mov=1, Chain=1):
    """
    Takes the population of n locations
    and generates an n by n numpy array of integers representing the movement of people from each location.
    Movement should be a random number between 3 and 8 percent of the total population of the location.
    Note that the movement happens from column to row.
    """
    movement = np.zeros((len(population), len(population)))
    if Mov == 1:
        for i in range(len(population)):
            for j in range(len(population)):
                movement[i][j] = np.random.randint(min * population[j], max * population[j])
        np.fill_diagonal(movement, 0)
        #if Chain==1, only allow movement from i to i+1, everything else is 0
    if Chain == 1:
        movement = np.zeros((len(population), len(population)))
        for i in range(1, len(population)):
            movement[i][i - 1] = np.random.randint(min * population[i - 1], max * population[i - 1])

    return movement


def gen_mov_ratio(movement, population):
    """
    Takes the movement matrix and divides each column by the population
    to generate an n by n np.array of floats.
    """
    return movement @ np.linalg.inv(np.diag(population))


def initialize_particles(params: ParticleFilterParams) -> ParticleFilterState:
    particles = np.zeros((params.num_particles, params.n, 3, params.results.shape[0]))
    betas = np.zeros((params.num_particles, params.n, params.results.shape[0]))

    for i in range(params.num_particles):
        for loc in range(params.n):
            I = np.random.randint(0, 5)
            S = params.population[loc] - I
            R = 0
            particles[i, loc, :, 0] = S, I, R
            betas[i, loc, 0] = np.random.uniform(0.6, 0.8)

    weights = np.ones(params.num_particles)
    m_post = np.zeros((params.results.shape[0], params.n, 3))
    beta_post = np.zeros((params.results.shape[0], params.n))

    return ParticleFilterState(particles, betas, weights, m_post, beta_post)


def resample_particles(state: ParticleFilterState):
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

    state.particles = state.particles[resampling_indices]
    state.betas = state.betas[resampling_indices]


def perturb_betas(state: ParticleFilterState, n: int):
    for i in range(state.particles.shape[0]):
        state.betas[i, :, -1] = np.exp(np.random.multivariate_normal(np.log(state.betas[i, :, -1]), 0.001 * np.eye(n)))


def run_particle_filter(params: ParticleFilterParams) -> ParticleFilterState:
    state = initialize_particles(params)

    for t in range(params.results.shape[0]):
        print(f"Iteration: {t}")

        if t != 0:
            for i in range(params.num_particles):
                state.particles[i, :, :, t] = SIR_tau_leap(params.population, params.movement,
                                                           state.particles[i, :, :, t - 1], state.betas[i, :, t - 1])[:,
                                              :, -1]
                state.betas[i, :, t] = state.betas[i, :, t - 1]

        for i in range(params.num_particles):
            for loc in range(params.num_locations):
                state.weights[i] *= norm.pdf(params.data[t, loc], state.particles[i, loc, 1, t], 100_000)

        state.weights /= np.sum(state.weights)
        resample_particles(state)
        perturb_betas(state, params.num_locations)

        state.m_post[t, :, :] = np.average(state.particles[:, :, :, t], axis=0, weights=state.weights)
        state.beta_post[t, :] = np.average(state.betas[:, :, t], axis=0, weights=state.weights)

    return state


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


