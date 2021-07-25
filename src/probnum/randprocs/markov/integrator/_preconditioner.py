"""Coordinate changes in state space models."""

import abc

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.special  # for vectorised factorial

from probnum import config, linops


class Preconditioner(abc.ABC):
    """Coordinate change transformations as preconditioners in state space models.

    For some models, this makes the filtering and smoothing steps more numerically
    stable.
    """

    @abc.abstractmethod
    def __call__(self, step) -> np.ndarray:
        # if more than step is needed, add them into the signature in the future
        raise NotImplementedError

    @cached_property
    def inverse(self) -> "Preconditioner":
        raise NotImplementedError


class NordsieckLikeCoordinates(Preconditioner):
    """Nordsieck-like coordinates.

    Similar to Nordsieck coordinates (which store the Taylor coefficients instead of the
    derivatives), but better for ODE filtering and smoothing. Used in integrator-transitions, e.g. in
    :class:`IntegratedWienerTransition`.
    """

    def __init__(self, powers, scales, dimension):
        # Clean way of assembling these coordinates cheaply,
        # because the powers and scales of the inverse
        # are better read off than inverted
        self.powers = powers
        self.scales = scales
        self.dimension = dimension

    @classmethod
    def from_order(cls, order, dimension):
        # used to conveniently initialise in the beginning
        powers = np.arange(order, -1, -1)
        scales = scipy.special.factorial(powers)
        return cls(
            powers=powers + 0.5,
            scales=scales,
            dimension=dimension,
        )

    def __call__(self, step):
        scaling_vector = np.abs(step) ** self.powers / self.scales
        if config.lazy_linalg:
            return linops.Kronecker(
                A=linops.Identity(self.dimension),
                B=linops.Scaling(factors=scaling_vector),
            )
        return np.kron(np.eye(self.dimension), np.diag(scaling_vector))

    @cached_property
    def inverse(self) -> "NordsieckLikeCoordinates":
        return NordsieckLikeCoordinates(
            powers=-self.powers,
            scales=1.0 / self.scales,
            dimension=self.dimension,
        )