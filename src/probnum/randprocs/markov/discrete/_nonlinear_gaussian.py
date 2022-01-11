"""Discrete transitions."""

from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from probnum import randvars
from probnum.randprocs.markov import _transition
from probnum.randprocs.markov.discrete import _condition_state
from probnum.typing import FloatLike, IntLike


class _DiscreteAsContinuousTransition(_transition.Transition):
    def __init__(self, discrete_transition):
        super().__init__(
            input_dim=discrete_transition.input_dim,
            output_dim=discrete_transition.output_dim,
        )
        self._discrete_transition = discrete_transition

    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        return self._discrete_transition.forward_rv(
            rv=rv,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self._discrete_transition.forward_realization(
            realization=realization,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

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
        return self._discrete_transition.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

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
        return self._discrete_transition.backward_realization(
            realization_obtained=realization_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    # Special arguments.

    # Nonlinear Gaussians:

    def state_trans_fun(self, t, x):
        return self._discrete_transition.state_trans_fun(x)

    def jacob_state_trans_fun(self, t, x):
        return self._discrete_transition.jacob_state_trans_fun(x)

    def proc_noise_cov_mat_fun(self, t):
        return self._discrete_transition.proc_noise_cov_mat_fun()

    def proc_noise_cov_cholesky_fun(self, t):
        return self._discrete_transition.proc_noise_cov_cholesky_fun()

    # Linear Gaussians:

    def state_trans_mat_fun(self, t):
        return self._discrete_transition.state_trans_mat_fun()

    def shift_vec_fun(self, t):
        return self._discrete_transition.shift_vec_fun()

    # Other wrappers:
    def _forward_rv_classic(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        return self._discrete_transition._forward_rv_classic(
            rv=rv,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def _forward_rv_sqrt(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        return self._discrete_transition._forward_rv_sqrt(
            rv=rv,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def _backward_rv_classic(self, *args, t, **kwargs):
        return self._discrete_transition._backward_rv_classic(*args, **kwargs)

    def _backward_rv_sqrt(self, *args, t, **kwargs):
        return self._discrete_transition._backward_rv_sqrt(*args, **kwargs)

    def _backward_rv_joseph(self, *args, t, **kwargs):
        return self._discrete_transition._backward_rv_joseph(*args, **kwargs)


class NonlinearGaussian:
    r"""Discrete transitions with additive Gaussian noise.

    .. math:: x_{i+1} \sim \mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g: \mathbb{R}^m \rightarrow \mathbb{R}^n` and process noise covariance matrix :math:`S`.

    Parameters
    ----------
    input_dim
        Dimension of the support of :math:`g` (in terms of :math:`x`), i.e. the input dimension.
    output_dim
        Dimension of the image of :math:`g`, i.e. the output dimension.
    state_trans_fun :
        State transition function :math:`g=g(t, x)`. Signature: ``state_trans_fun(t, x)``.
    proc_noise_cov_mat_fun :
        Process noise covariance matrix function :math:`S=S(t)`. Signature: ``proc_noise_cov_mat_fun(t)``.
    jacob_state_trans_fun :
        Jacobian of the state transition function :math:`g` (with respect to :math:`x`), :math:`Jg=Jg(t, x)`.
        Signature: ``jacob_state_trans_fun(t, x)``.
    proc_noise_cov_cholesky_fun :
        Cholesky factor of the process noise covariance matrix function :math:`\sqrt{S}=\sqrt{S}(t)`. Signature: ``proc_noise_cov_cholesky_fun(t)``.


    See Also
    --------
    :class:`DiscreteModel`
    :class:`NonlinearGaussianLinearModel`
    """

    def __init__(
        self,
        input_dim: IntLike,
        output_dim: IntLike,
        state_trans_fun: Callable[[FloatLike, np.ndarray], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatLike], np.ndarray],
        jacob_state_trans_fun: Optional[
            Callable[[FloatLike, np.ndarray], np.ndarray]
        ] = None,
        proc_noise_cov_cholesky_fun: Optional[Callable[[FloatLike], np.ndarray]] = None,
    ):
        self.state_trans_fun = state_trans_fun
        self.proc_noise_cov_mat_fun = proc_noise_cov_mat_fun

        # "Private", bc. if None, overwritten by the property with the same name
        self._proc_noise_cov_cholesky_fun = proc_noise_cov_cholesky_fun

        def dummy_if_no_jacobian(x):
            raise NotImplementedError

        self.jacob_state_trans_fun = (
            jacob_state_trans_fun
            if jacob_state_trans_fun is not None
            else dummy_if_no_jacobian
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def as_continuous_transition(self):
        return _DiscreteAsContinuousTransition(self)

    def _backward_realization_via_backward_rv(self, realization, *args, **kwargs):
        real_as_rv = randvars.Constant(support=realization)
        return self.backward_rv(real_as_rv, *args, **kwargs)

    def _forward_realization_via_forward_rv(self, realization, *args, **kwargs):
        real_as_rv = randvars.Constant(support=realization)
        return self.forward_rv(real_as_rv, *args, **kwargs)

    def forward_realization(
        self, realization, compute_gain=False, _diffusion=1.0, **kwargs
    ):

        newmean = self.state_trans_fun(realization)
        newcov = _diffusion * self.proc_noise_cov_mat_fun()

        return randvars.Normal(newmean, newcov), {}

    def forward_rv(self, rv, compute_gain=False, _diffusion=1.0, **kwargs):
        raise NotImplementedError("Not available")

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        _diffusion=1.0,
        **kwargs,
    ):
        raise NotImplementedError("Not available")

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        _diffusion=1.0,
        **kwargs,
    ):

        # Should we use the _backward_rv_classic here?
        # It is only intractable bc. forward_rv is intractable
        # and assuming its forward formula would yield a valid
        # gain, the backward formula would be valid.
        # This is the case for the UKF, for instance.
        raise NotImplementedError("Not available")

    # Implementations that are the same for all sorts of
    # discrete Gaussian transitions, in particular shared
    # by LinearNonlinearGaussian and e.g. DiscreteUKFComponent.

    def _backward_rv_classic(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        _diffusion=None,
        _linearise_at=None,
    ):

        if rv_forwarded is None or gain is None:
            rv_forwarded, info_forwarded = self.forward_rv(
                rv,
                compute_gain=True,
                _diffusion=_diffusion,
                _linearise_at=_linearise_at,
            )
            gain = info_forwarded["gain"]
        info = {"rv_forwarded": rv_forwarded}
        return (
            _condition_state.condition_state_on_rv(rv_obtained, rv_forwarded, rv, gain),
            info,
        )

    @lru_cache(maxsize=None)
    def proc_noise_cov_cholesky_fun(self):
        if self._proc_noise_cov_cholesky_fun is not None:
            return self._proc_noise_cov_cholesky_fun()
        covmat = self.proc_noise_cov_mat_fun()
        return np.linalg.cholesky(covmat)

    @classmethod
    def from_callable(
        cls,
        input_dim: IntLike,
        output_dim: IntLike,
        state_trans_fun: Callable[[FloatLike, np.ndarray], np.ndarray],
        jacob_state_trans_fun: Callable[[FloatLike, np.ndarray], np.ndarray],
    ):
        """Turn a callable into a deterministic transition."""

        def diff():
            return np.zeros((output_dim, output_dim))

        def diff_cholesky():
            return np.zeros((output_dim, output_dim))

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_fun=state_trans_fun,
            jacob_state_trans_fun=jacob_state_trans_fun,
            proc_noise_cov_mat_fun=diff,
            proc_noise_cov_cholesky_fun=diff_cholesky,
        )
