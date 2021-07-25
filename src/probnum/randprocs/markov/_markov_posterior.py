"""Markov process posteriors."""

from typing import Iterable, Optional, Type, TypeVar

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.randprocs.markov import _markov_process, _transition
from probnum.typing import FloatArgType, ShapeArgType

_InputType = TypeVar("InputType")
_OutputType = TypeVar("OutputType")
import dataclasses


class MarkovProcessPosterior(_markov_process.MarkovProcess):
    @dataclasses.dataclass
    class State:
        """State of a Markov process posterior."""

        rv: randvars.RandomVariable
        t: FloatArgType
        transition: _transition.Transition

    def __init__(
        self,
        prior_markov_process,
        locations: Optional[Iterable[FloatArgType]] = None,
        states: Optional[Iterable[randvars.RandomVariable]] = None,
    ) -> None:
        super().__init__(
            initarg=prior_markov_process.initarg,
            initrv=prior_markov_process.initrv,
            transition=prior_markov_process.transition,
        )
        self._locations = list(locations) if locations is not None else []
        self._states = list(states) if states is not None else []
        self._frozen = False

    def __call__(self, args: _InputType) -> randvars.RandomVariable[_OutputType]:
        raise NotImplementedError

    def mean(self, args: _InputType) -> _OutputType:
        raise NotImplementedError

    def cov(self, args0: _InputType, args1: Optional[_InputType] = None) -> _OutputType:
        raise NotImplementedError

    def marginal(self, args: _InputType) -> _randomvariablelist._RandomVariableList:
        raise NotImplementedError

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: _InputType,
        size: ShapeArgType = (),
    ) -> _OutputType:
        raise NotImplementedError

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
