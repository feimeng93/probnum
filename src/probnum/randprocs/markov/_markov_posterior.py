"""Markov process posteriors."""

from typing import Iterable, Optional, Type, TypeVar

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.randprocs import _random_process
from probnum.typing import FloatArgType, ShapeArgType

_InputType = TypeVar("InputType")
_OutputType = TypeVar("OutputType")


class MarkovProcessPosterior(_random_process.RandomProcess):
    def __init__(
        self,
        output_dim,
        dtype,
        locations: Optional[Iterable[FloatArgType]] = None,
        states: Optional[Iterable[randvars.RandomVariable]] = None,
    ) -> None:
        super().__init__(input_dim=1, output_dim=output_dim, dtype=dtype)
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
