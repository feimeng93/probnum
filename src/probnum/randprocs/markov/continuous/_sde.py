"""SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from probnum import randvars
from probnum.randprocs.markov import _transition, discrete
from probnum.randprocs.markov.continuous import _utils
from probnum.typing import FloatArgType, IntArgType
from probnum.utils.linalg import tril_to_positive_tril


class SDE(_transition.Transition):
    r"""Stochastic differential equation.

    .. math:: d x(t) = g(t, x(t)) d t + l(t, x(t)) d w(t),

    driven by a Wiener process :math:`w` with isotropic diffusion :math:`\Gamma(t) = \gamma(t) I_d`.
    """

    def __init__(
        self,
        state_dimension: IntArgType,
        wiener_process_dimension: IntArgType,
        drift_function: Callable[[FloatArgType, np.ndarray], np.ndarray],
        dispersion_function: Callable[[FloatArgType, np.ndarray], np.ndarray],
        drift_jacobian: Optional[Callable[[FloatArgType, np.ndarray], np.ndarray]],
        squared_scalar_diffusion_function: Optional[
            Callable[[FloatArgType], FloatArgType]
        ] = None,
        _duplicate=None,
    ):
        _duplicate = _duplicate or self.__duplicate
        super().__init__(
            input_dim=state_dimension, output_dim=state_dimension, _duplicate=_duplicate
        )

        # Mandatory arguments
        self.state_dimension = state_dimension
        self.wiener_process_dimension = wiener_process_dimension
        self.drift_function = drift_function
        self.dispersion_function = dispersion_function

        # Optional arguments
        def unit_diffusion(t):
            return 1.0

        self.squared_scalar_diffusion_function = (
            squared_scalar_diffusion_function or unit_diffusion
        )
        self.drift_jacobian = drift_jacobian

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        **kwargs,
    ):
        return self._forward_realization_via_forward_rv(
            realization,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            **kwargs,
        )

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        **kwargs,
    ):
        raise NotImplementedError("Not available.")

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        **kwargs,
    ):
        return self._backward_realization_via_backward_rv(
            realization_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            **kwargs,
        )

    def backward_rv(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        **kwargs,
    ):
        raise NotImplementedError("Not available.")

    def __duplicate(self, **changes):
        def replace_key(key):
            try:
                return changes[key]
            except KeyError:
                return getattr(self, key)

        state_dimension = replace_key("state_dimension")
        wiener_process_dimension = replace_key("wiener_process_dimension")
        drift_function = replace_key("drift_function")
        dispersion_function = replace_key("dispersion_function")
        drift_jacobian = replace_key("drift_jacobian")
        squared_scalar_diffusion_function = replace_key(
            "squared_scalar_diffusion_function"
        )
        return SDE(
            state_dimension=state_dimension,
            wiener_process_dimension=wiener_process_dimension,
            drift_function=drift_function,
            dispersion_function=dispersion_function,
            drift_jacobian=drift_jacobian,
            squared_scalar_diffusion_function=squared_scalar_diffusion_function,
        )
