"""Approximate, discrete Markov transitions."""

from ._moment_matching import MomentMatchingTransition, UnscentedTransformTransition
from ._taylor_series import LocallyLinearizingTransition

__all__ = [
    "LocallyLinearizingTransition",
    "MomentMatchingTransition",
    "UnscentedTransformTransition",
]

LocallyLinearizingTransition.__module__ = "probnum.randprocs.markov.discrete.approx"
MomentMatchingTransition.__module__ = "probnum.randprocs.markov.discrete.approx"
UnscentedTransformTransition.__module__ = "probnum.randprocs.markov.discrete.approx"
