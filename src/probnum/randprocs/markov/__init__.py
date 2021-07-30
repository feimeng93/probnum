"""Markov processes and probabilistic state-space model routines.

This package implements continuous-discrete and discrete-discrete state space models,
which are the basis for Bayesian filtering and smoothing, but also for probabilistic ODE
solvers.
"""

from . import continuous, discrete, integrator, utils
from ._approx_transition import ApproximateTransition
from ._markov_process import MarkovProcess
from ._transition import Transition

# Public classes and functions. Order is reflected in documentation.
__all__ = ["MarkovProcess", "Transition", "ApproximateTransition"]

# Set correct module paths. Corrects links and module paths in documentation.
MarkovProcess.__module__ = "probnum.randprocs.markov"
Transition.__module__ = "probnum.randprocs.markov"
ApproximateTransition.__module__ = "probnum.randprocs.markov"
