"""Gaussian filtering and smoothing."""

from ._kalman import Kalman
from ._kalman_posterior import FilteringPosterior, KalmanPosterior, SmoothingPosterior
from ._kalman_state import KalmanState

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kalman",
    "KalmanPosterior",
    "FilteringPosterior",
    "SmoothingPosterior",
    "KalmanState",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
Kalman.__module__ = "probnum.filtsmooth.gaussian"
KalmanPosterior.__module__ = "probnum.filtsmooth.gaussian"
FilteringPosterior.__module__ = "probnum.filtsmooth.gaussian"
SmoothingPosterior.__module__ = "probnum.filtsmooth.gaussian"
KalmanState.__module__ = "probnum.filtsmooth.gaussian"
