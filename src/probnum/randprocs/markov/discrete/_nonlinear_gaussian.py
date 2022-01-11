"""Discrete transitions."""

from functools import cached_property, lru_cache
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

    # LTI Gaussians

    @cached_property
    def proc_noise_cov_cholesky(self):
        return self._discrete_transition.proc_noise_cov_cholesky

    @property
    def proc_noise_cov_mat(self):
        return self._discrete_transition.proc_noise_cov_mat

    @property
    def shift_vec(self):
        return self._discrete_transition.shift_vec

    @property
    def state_trans_mat(self):
        return self._discrete_transition.state_trans_mat

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

    # Smoothing and sampling implementations
    # Todo: move somewhere else. For example into functions that get lists of discrete transitions.

    def smooth_list(
        self, rv_list, locations, _diffusion_list, _previous_posterior=None
    ):
        """Apply smoothing to a list of random variables, according to the present
        transition.

        Parameters
        ----------
        rv_list : randvars._RandomVariableList
            List of random variables to be smoothed.
        locations :
            Locations :math:`t` of the random variables in the time-domain. Used for continuous-time transitions.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Specify a previous posterior to improve linearisation in approximate backward passes.
            Used in iterated smoothing based on posterior linearisation.

        Returns
        -------
        randvars._RandomVariableList
            List of smoothed random variables.
        """

        final_rv = rv_list[-1]
        curr_rv = final_rv
        out_rvs = [curr_rv]
        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = rv_list[idx - 1]

            _linearise_smooth_step_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(locations[idx - 1])
            )
            squared_diffusion = _diffusion_list[idx - 1]

            # Actual smoothing step
            curr_rv, _ = self.backward_rv(
                curr_rv,
                unsmoothed_rv,
                _diffusion=squared_diffusion,
                _linearise_at=_linearise_smooth_step_at,
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return randvars._RandomVariableList(out_rvs)

    def jointly_transform_base_measure_realization_list_backward(
        self,
        base_measure_realizations: np.ndarray,
        t: FloatLike,
        rv_list: randvars._RandomVariableList,
        _diffusion_list: np.ndarray,
        _previous_posterior=None,
    ) -> np.ndarray:
        """Transform samples from a base measure into joint backward samples from a list
        of random variables.

        Parameters
        ----------
        base_measure_realizations :
            Base measure realizations (usually samples from a standard Normal distribution).
            These are transformed into joint realizations of the random variable list.
        rv_list :
            List of random variables to be jointly sampled from.
        t :
            Locations of the random variables in the list. Assumed to be sorted.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Previous posterior. Used for iterative posterior linearisation.

        Returns
        -------
        np.ndarray
            Jointly transformed realizations.
        """
        curr_rv = rv_list[-1]

        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            -1
        ].reshape((-1,))
        out_samples = [curr_sample]

        for idx in reversed(range(1, len(t))):
            unsmoothed_rv = rv_list[idx - 1]
            _linearise_smooth_step_at = (
                None if _previous_posterior is None else _previous_posterior(t[idx - 1])
            )

            # Condition on the 'future' realization and sample
            squared_diffusion = _diffusion_list[idx - 1]
            dt = t[idx] - t[idx - 1]
            curr_rv, _ = self.backward_realization(
                curr_sample,
                unsmoothed_rv,
                _linearise_at=_linearise_smooth_step_at,
                _diffusion=squared_diffusion,
            )
            curr_sample = (
                curr_rv.mean
                + curr_rv.cov_cholesky
                @ base_measure_realizations[idx - 1].reshape(
                    -1,
                )
            )
            out_samples.append(curr_sample)

        out_samples.reverse()
        return out_samples

    def jointly_transform_base_measure_realization_list_forward(
        self,
        base_measure_realizations: np.ndarray,
        t: FloatLike,
        initrv: randvars.RandomVariable,
        _diffusion_list: np.ndarray,
        _previous_posterior=None,
    ) -> np.ndarray:
        """Transform samples from a base measure into joint backward samples from a list
        of random variables.

        Parameters
        ----------
        base_measure_realizations :
            Base measure realizations (usually samples from a standard Normal distribution).
            These are transformed into joint realizations of the random variable list.
        initrv :
            Initial random variable.
        t :
            Locations of the random variables in the list. Assumed to be sorted.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Previous posterior. Used for iterative posterior linearisation.

        Returns
        -------
        np.ndarray
            Jointly transformed realizations.
        """
        curr_rv = initrv

        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            0
        ].reshape((-1,))
        out_samples = [curr_sample]

        for idx in range(1, len(t)):

            _linearise_prediction_step_at = (
                None if _previous_posterior is None else _previous_posterior(t[idx - 1])
            )

            squared_diffusion = _diffusion_list[idx - 1]
            dt = t[idx] - t[idx - 1]
            curr_rv, _ = self.forward_realization(
                curr_sample,
                _linearise_at=_linearise_prediction_step_at,
                _diffusion=squared_diffusion,
            )
            curr_sample = (
                curr_rv.mean
                + curr_rv.cov_cholesky
                @ base_measure_realizations[idx - 1].reshape((-1,))
            )
            out_samples.append(curr_sample)
        return out_samples
