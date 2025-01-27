"""Particle filtering posterior."""
import abc
from dataclasses import dataclass
# from functools import cached_property
from cached_property import cached_property
import numpy as np

from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ParticleFilterPosterior(FiltSmoothPosterior):
    """Particle filter posterior distribution."""

    def __init__(self, particle_state_list, locations):
        self.particle_state_list = particle_state_list
        self.locations = locations

    def __call__(self, t):
        raise NotImplementedError

    @cached_property
    def supports(self):
        return np.array([state.support for state in self.particle_state_list])

    @cached_property
    def event_probabilities(self):
        return np.array(
            [state.event_probabilities for state in self.particle_state_list]
        )

    @cached_property
    def mean(self):
        return np.einsum("ijk,ij->ik", self.supports, self.event_probabilities)

    def __len__(self):
        return len(self.particle_state_list)

    def __getitem__(self):
        state = self.particle_state_list[idx]
        return state.particles, state.weights
