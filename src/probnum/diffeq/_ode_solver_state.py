"""ODE solver states."""

import dataclasses
from typing import Optional, Union

import numpy as np

from probnum import randprocs, randvars
from probnum.typing import FloatArgType


@dataclasses.dataclass(frozen=True)
class ODESolverState:
    """ODE solver states."""

    t: float
    rv: randvars.RandomVariable
    error_estimate: np.ndarray
    reference_state: np.ndarray
