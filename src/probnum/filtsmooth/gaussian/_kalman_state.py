"""Kalman filter/smoother state."""

import dataclasses

from probnum import randprocs, randvars
from probnum.typing import FloatArgType


@dataclasses.dataclass
class KalmanState:
    """State of a Kalman filter/smoother."""

    rv: randvars.RandomVariable
    t: FloatArgType
    transition: randprocs.markov.Transition
