"""Approximate transitions."""

import abc

from probnum import randvars
from probnum.randprocs.markov import _transition

__all__ = ["ApproximateTransition"]


class ApproximateTransition(randprocs.markov.Transition, abc.ABC):
    """Approximate transitions.

    Interface for Taylor-series approximations (extended Kalman filtering), moment
    matching (including e.g. unscented Kalman filtering), and more.
    """

    def __init__(self, non_linear_model: _transition.Transition):
        super().__init__(
            input_dim=non_linear_model.input_dim, output_dim=non_linear_model.output_dim
        )
        self._non_linear_model = non_linear_model

    @property
    def non_linear_model(self):
        return self._non_linear_model

    @abc.abstractmethod
    def linearize(self, at: randvars.RandomVariable) -> _transition.Transition:
        raise NotImplementedError

    @abc.abstractmethod
    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        lin_state = rv if _linearise_at is not None else _linearise_at
        linearized = self.linearize(at=lin_state)
        return linearized.forward_rv(
            rv=rv, t=t, dt=dt, compute_gain=compute_gain, _diffusion=_diffusion
        )

    @abc.abstractmethod
    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        lin_state = (
            randvars.Constant(realization)
            if _linearise_at is not None
            else _linearise_at
        )
        linearized = self.linearize(at=lin_state)
        return linearized.forward_realization(
            realization=realization,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
        )

    @abc.abstractmethod
    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        lin_state = rv if _linearise_at is not None else _linearise_at
        linearized = self.linearize(at=lin_state)
        return linearized.backward_realization(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
        )

    @abc.abstractmethod
    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        lin_state = rv if _linearise_at is not None else _linearise_at
        linearized = self.linearize(at=lin_state)
        return linearized.backward_realization(
            realization_obtained=realization_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
        )
