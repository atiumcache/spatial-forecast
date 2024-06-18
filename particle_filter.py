from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParticleFilterParams:
    num_particles: int
    num_locations: int
    population: np.ndarray
    movement: np.ndarray
    results: np.ndarray
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = self.results[:, :, 1]


@dataclass
class ParticleFilterState:
    particles: np.ndarray
    betas: np.ndarray
    weights: np.ndarray
    m_post: np.ndarray
    beta_post: np.ndarray
