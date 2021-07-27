"""ODE filter states."""

import dataclasses

from probnum import randprocs
from probnum.diffeq import _ode_solver_state


@dataclasses.dataclass(frozen=True)
class ODEFilterState(_ode_solver_state.ODESolverState):
    """ODE filter state."""

    transition: randprocs.markov.Transition
