"""Markov processes and probabilistic state-space model routines.

This package implements continuous-discrete and discrete-discrete state space models,
which are the basis for Bayesian filtering and smoothing, but also for probabilistic ODE
solvers.
"""

from . import continuous, discrete, integrator, utils
from ._markov_process import MarkovProcess
from ._transition import Transition
from ._transition_utils import (
    jointly_transform_base_measure_realization_list_backward,
    jointly_transform_base_measure_realization_list_forward,
    smooth_list,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "MarkovProcess",
    "Transition",
    "smooth_list",
    "jointly_transform_base_measure_realization_list_forward",
    "jointly_transform_base_measure_realization_list_backward",
]

# Set correct module paths. Corrects links and module paths in documentation.
MarkovProcess.__module__ = "probnum.randprocs.markov"
Transition.__module__ = "probnum.randprocs.markov"
smooth_list.__module__ = "probnum.randprocs.markov"
jointly_transform_base_measure_realization_list_forward.__module__ = (
    "probnum.randprocs.markov"
)
jointly_transform_base_measure_realization_list_backward.__module__ = (
    "probnum.randprocs.markov"
)
