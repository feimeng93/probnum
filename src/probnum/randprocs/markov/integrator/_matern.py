"""Matern processes."""
import warnings

import numpy as np
import scipy.special

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from probnum import randvars
from probnum.randprocs.markov import _markov_process, continuous
from probnum.randprocs.markov.integrator import _integrator, _utils


class MaternProcess(_markov_process.MarkovProcess):
    r"""Matern process.

    Convenience access to (:math:`d` dimensional) Matern(:math:`\nu`) processes.

    Parameters
    ----------
    lengthscale
        Lengthscale of the Matern process.
    initarg
        Initial time point.
    num_derivatives
        Number of modelled derivatives of the integrated process (''order'', ''number of integrations'').
        Optional. Default is :math:`\nu=1`.
    wiener_process_dimension
        Dimension of the underlying Wiener process.
        Optional. Default is :math:`d=1`.
        The dimension of the integrated Wiener process itself is :math:`d(\nu + 1)`.
    initrv
        Law of the integrated Wiener process at the initial time point.
        Optional. Default is a :math:`d(\nu + 1)` dimensional standard-normal distribution.
    diffuse
        Whether to instantiate a diffuse prior. A diffuse prior has large initial variances.
        Optional. Default is `False`.
        If `True`, and if an initial random variable is not passed, an initial random variable is created,
        where the initial covariance is of the form :math:`\kappa I_{d(\nu + 1)}`
        with :math:`\kappa=10^6`.
        Diffuse priors are used when initial distributions are not known.
        They are common for filtering-based probabilistic ODE solvers.
    forward_implementation
        Implementation of the forward-propagation in the underlying transitions.
        Optional. Default is `classic`. `sqrt` implementation is more computationally expensive, but also more stable.
    backward_implementation
        Implementation of the backward-conditioning in the underlying transitions.
        Optional. Default is `classic`. `sqrt` implementation is more computationally expensive, but also more stable.

    Raises
    ------
    Warning
        If `initrv` is not None and `diffuse` is True.

    Examples
    --------
    >>> matern1 = MaternProcess(lengthscale=1., initarg=0.)
    >>> print(matern1)
    <MaternProcess with input_dim=1, output_dim=2, dtype=float64>

    >>> matern2 = MaternProcess(lengthscale=1.,initarg=0., num_derivatives=2)
    >>> print(matern2)
    <MaternProcess with input_dim=1, output_dim=3, dtype=float64>

    >>> matern3 = MaternProcess(lengthscale=1.,initarg=0., wiener_process_dimension=10)
    >>> print(matern3)
    <MaternProcess with input_dim=1, output_dim=20, dtype=float64>

    >>> matern4 = MaternProcess(lengthscale=1.,initarg=0., num_derivatives=4, wiener_process_dimension=1)
    >>> print(matern4)
    <MaternProcess with input_dim=1, output_dim=5, dtype=float64>
    """

    def __init__(
        self,
        lengthscale,
        initarg,
        num_derivatives=1,
        wiener_process_dimension=1,
        initrv=None,
        diffuse=False,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        matern_transition = MaternTransition(
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            lengthscale=lengthscale,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
        if initrv is not None and diffuse:
            warnings.warn(
                "Parameter `diffuse` has no effect, because an `initrv` has been provided."
            )
        if initrv is None:
            if diffuse:
                scale_cholesky = 1e3
            else:
                scale_cholesky = 1.0
            zeros = np.zeros(matern_transition.state_dimension)
            cov_cholesky = scale_cholesky * np.eye(matern_transition.state_dimension)
            initrv = randvars.Normal(
                mean=zeros, cov=cov_cholesky ** 2, cov_cholesky=cov_cholesky
            )

        super().__init__(transition=matern_transition, initrv=initrv, initarg=initarg)


class MaternTransition(_integrator.IntegratorTransition, continuous.LTISDE):
    """Matern process in :math:`d` dimensions."""

    def __init__(
        self,
        num_derivatives: int,
        wiener_process_dimension: int,
        lengthscale: float,
        squared_scalar_diffusion: float = 1.0,
        forward_implementation="classic",
        backward_implementation="classic",
    ):

        self.lengthscale = lengthscale

        _integrator.IntegratorTransition.__init__(
            self,
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
        )
        continuous.LTISDE.__init__(
            self,
            drift_matrix=self._driftmat,
            force_vector=self._forcevec,
            dispersion_matrix=self._dispmat,
            squared_scalar_diffusion=squared_scalar_diffusion,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

    @cached_property
    def _driftmat(self):
        driftmat = np.diag(np.ones(self.num_derivatives), 1)
        nu = self.num_derivatives + 0.5
        D, lam = self.num_derivatives + 1, np.sqrt(2 * nu) / self.lengthscale
        driftmat[-1, :] = np.array(
            [-scipy.special.binom(D, i) * lam ** (D - i) for i in range(D)]
        )
        return np.kron(np.eye(self.wiener_process_dimension), driftmat)

    @cached_property
    def _forcevec(self):
        force_1d = np.zeros(self.num_derivatives + 1)
        return np.kron(np.ones(self.wiener_process_dimension), force_1d)

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.num_derivatives + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion
        return np.kron(np.eye(self.wiener_process_dimension), dispmat_1d).T

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Fetch things into preconditioned space
        rv = _utils.apply_precon(self.precon.inverse(dt), rv)

        # Apply preconditioning to system matrices
        self.drift_matrix = (
            self.precon.inverse(dt) @ self.drift_matrix @ self.precon(dt)
        )
        self.force_vector = self.precon.inverse(dt) @ self.force_vector
        self.dispersion_matrix = self.precon.inverse(dt) @ self.dispersion_matrix

        # Discretise and propagate
        discretized_model = self.discretize(dt=dt)
        rv, info = discretized_model.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        # Undo preconditioning and return
        rv = _utils.apply_precon(self.precon(dt), rv)
        info["crosscov"] = self.precon(dt) @ info["crosscov"] @ self.precon(dt).T
        if "gain" in info:
            info["gain"] = self.precon(dt) @ info["gain"] @ self.precon.inverse(dt).T

        self.drift_matrix = (
            self.precon(dt) @ self.drift_matrix @ self.precon.inverse(dt)
        )
        self.force_vector = self.precon(dt) @ self.force_vector
        self.dispersion_matrix = self.precon(dt) @ self.dispersion_matrix

        return rv, info

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Fetch things into preconditioned space
        rv_obtained = _utils.apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _utils.apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _utils.apply_precon(self.precon.inverse(dt), rv_forwarded)
            if rv_forwarded is not None
            else None
        )
        gain = (
            self.precon.inverse(dt) @ gain @ self.precon.inverse(dt).T
            if gain is not None
            else None
        )

        # Apply preconditioning to system matrices
        self.drift_matrix = (
            self.precon.inverse(dt) @ self.drift_matrix @ self.precon(dt)
        )
        self.force_vector = self.precon.inverse(dt) @ self.force_vector
        self.dispersion_matrix = self.precon.inverse(dt) @ self.dispersion_matrix

        # Discretise and propagate
        discretized_model = self.discretize(dt=dt)
        rv, info = discretized_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

        # Undo preconditioning and return
        rv = _utils.apply_precon(self.precon(dt), rv)
        self.drift_matrix = (
            self.precon(dt) @ self.drift_matrix @ self.precon.inverse(dt)
        )
        self.force_vector = self.precon(dt) @ self.force_vector
        self.dispersion_matrix = self.precon(dt) @ self.dispersion_matrix
        return rv, info

    def _duplicate(self, **changes):
        def replace_key(key):
            """If the key is part of the desired changes, change appropriately.

            Otherwise, take the current value.
            """
            try:
                return changes[key]
            except KeyError:
                return getattr(self, key)

        num_derivatives = replace_key("num_derivatives")
        wiener_process_dimension = replace_key("wiener_process_dimension")
        lengthscale = replace_key("lengthscale")
        squared_scalar_diffusion = replace_key("squared_scalar_diffusion")
        forward_implementation = replace_key("forward_implementation")
        backward_implementation = replace_key("backward_implementation")

        return MaternTransition(
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            lengthscale=lengthscale,
            squared_scalar_diffusion=squared_scalar_diffusion,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
