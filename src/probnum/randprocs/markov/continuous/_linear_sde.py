"""Linear SDEs."""

import functools
from typing import Callable, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from probnum import randvars
from probnum.randprocs.markov import _transition, discrete
from probnum.randprocs.markov.continuous import _sde, _utils
from probnum.typing import FloatArgType, IntArgType
from probnum.utils.linalg import tril_to_positive_tril


class LinearSDE(_sde.SDE):
    """Linear stochastic differential equation (SDE),

    .. math:: d x(t) = [G(t) x(t) + v(t)] d t + L(t) x(t) d w(t).

    For Gaussian initial conditions, this solution is a Gaussian process.

    Parameters
    ----------
    drift_matrix_function :
        This is G = G(t). The evaluations of this function are called
        the driftmatrix of the SDE.
        Returns np.ndarray with shape=(n, n)
    force_vector_function :
        This is v = v(t). Evaluations of this function are called
        the force(vector) of the SDE.
        Returns np.ndarray with shape=(n,)
    dispersion_matrix_function :
        This is L = L(t). Evaluations of this function are called
        the dispersion(matrix) of the SDE.
        Returns np.ndarray with shape=(n, s)
    mde_atol
        Absolute tolerance passed to the solver of the moment differential equations (MDEs). Optional. Default is 1e-6.
    mde_rtol
        Relative tolerance passed to the solver of the moment differential equations (MDEs). Optional. Default is 1e-6.
    mde_solver
        Method that is chosen in `scipy.integrate.solve_ivp`. Any string that is compatible with ``solve_ivp(..., method=mde_solve,...)`` works here.
        Usual candidates are ``[RK45, LSODA, Radau, BDF, RK23, DOP853]``. Optional. Default is LSODA.
    """

    def __init__(
        self,
        state_dimension: IntArgType,
        wiener_process_dimension: IntArgType,
        drift_matrix_function: Callable[[FloatArgType], np.ndarray],
        force_vector_function: Callable[[FloatArgType], np.ndarray],
        dispersion_matrix_function: Callable[[FloatArgType], np.ndarray],
        squared_scalar_diffusion_function: Optional[
            Callable[[FloatArgType], FloatArgType]
        ] = None,
        mde_atol: Optional[FloatArgType] = 1e-6,
        mde_rtol: Optional[FloatArgType] = 1e-6,
        mde_solver: Optional[str] = "RK45",
        forward_implementation: Optional[str] = "classic",
    ):

        # Transform functions to be SDE-compatible and initialize super().
        def drift_function(t, x):
            return drift_matrix_function(t) @ x + force_vector_function(t)

        def drift_jacobian(t, x):
            return drift_matrix_function(t)

        def dispersion_function(t, x):
            return dispersion_matrix_function(t)

        super().__init__(
            state_dimension=state_dimension,
            wiener_process_dimension=wiener_process_dimension,
            drift_function=drift_function,
            drift_jacobian=drift_jacobian,
            dispersion_function=dispersion_function,
            squared_scalar_diffusion_function=squared_scalar_diffusion_function,
        )

        # Choose implementation for forward transitions
        choose_mde_forward_implementation = {
            "classic": self._solve_mde_forward_classic,
            "sqrt": self._solve_mde_forward_sqrt,
        }
        self._mde_forward_implementation = choose_mde_forward_implementation[
            forward_implementation
        ]

        # Once different smoothing algorithms are available,
        # replicate the scheme from DiscreteGaussian here, in which
        # the initialisation decides between, e.g., classic and sqrt implementations.

        # Store remaining functions and attributes
        self.drift_matrix_function = drift_matrix_function
        self.force_vector_function = force_vector_function
        self.dispersion_matrix_function = dispersion_matrix_function
        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver
        self.forward_implementation = forward_implementation

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        _compute_gain=False,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        return self._mde_forward_implementation(rv, t, dt)

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Ignore rv_forwarded
        return self._solve_mde_backward(
            rv_obtained=rv_obtained,
            rv=rv,
            t=t,
            dt=dt,
        )

    # Forward and backward implementation(s)

    def _solve_mde_forward_classic(self, rv, t, dt):
        """Solve forward moment differential equations (MDEs)."""
        dim = rv.mean.shape[0]
        mde, y0 = self._setup_vectorized_mde_forward_classic(rv)

        sol, new_mean, new_cov = self._solve_mde_forward(mde, y0, t, dt, dim)

        # Useful for backward transitions
        # Aka continuous time smoothing.
        sol_mean = lambda t: sol.sol(t)[:dim]
        sol_cov = lambda t: sol.sol(t)[dim:].reshape((dim, dim))

        return randvars.Normal(mean=new_mean, cov=new_cov), {
            "sol": sol,
            "sol_mean": sol_mean,
            "sol_cov": sol_cov,
        }

    def _solve_mde_forward_sqrt(self, rv, t, dt):
        """Solve forward moment differential equations (MDEs) using a square-root
        implementation."""
        dim = rv.mean.shape[0]
        mde, y0 = self._setup_vectorized_mde_forward_sqrt(
            rv,
        )

        sol, new_mean, new_cov_cholesky = self._solve_mde_forward(mde, y0, t, dt, dim)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        # Useful for backward transitions
        # Aka continuous time smoothing.
        sol_mean = lambda t: sol.sol(t)[:dim]
        sol_cov_cholesky = lambda t: sol.sol(t)[dim:].reshape((dim, dim))
        sol_cov = (
            lambda t: sol.sol(t)[dim:].reshape((dim, dim))
            @ sol.sol(t)[dim:].reshape((dim, dim)).T
        )

        return randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        ), {
            "sol": sol,
            "sol_mean": sol_mean,
            "sol_cov_cholesky": sol_cov_cholesky,
            "sol_cov": sol_cov,
        }

    def _solve_mde_forward(self, mde, y0, t, dt, dim):
        """Solve forward moment differential equations (MDEs)."""
        # Dense output for lambda-expression
        sol = scipy.integrate.solve_ivp(
            mde,
            (t, t + dt),
            y0,
            method=self.mde_solver,
            atol=self.mde_atol,
            rtol=self.mde_rtol,
            dense_output=True,
        )
        y_end = sol.y[:, -1]
        new_mean = y_end[:dim]
        # If forward_sqrt is used, new_cov_or_cov_cholesky is a Cholesky factor of the covariance
        # If forward_classic is used, new_cov_or_cov_cholesky is the covariance
        new_cov_or_cov_cholesky = y_end[dim:].reshape((dim, dim))

        return sol, new_mean, new_cov_or_cov_cholesky

    def _solve_mde_backward(self, rv_obtained, rv, t, dt):
        """Solve backward moment differential equations (MDEs)."""
        _, mde_forward_info = self._mde_forward_implementation(rv, t, dt)

        mde_forward_sol_mean = mde_forward_info["sol_mean"]
        mde_forward_sol_cov = mde_forward_info["sol_cov"]

        mde, y0 = self._setup_vectorized_mde_backward(
            rv_obtained,
        )
        # Use forward solution for mean and covariance in scipy's ivp
        # Dense output for lambda-expression
        sol = scipy.integrate.solve_ivp(
            mde,
            (t + dt, t),
            y0,
            method=self.mde_solver,
            atol=self.mde_atol,
            rtol=self.mde_rtol,
            args=(mde_forward_sol_mean, mde_forward_sol_cov),
            dense_output=True,
        )
        dim = rv.mean.shape[0]
        y_end = sol.y[:, -1]
        new_mean = y_end[:dim]
        new_cov = y_end[dim:].reshape((dim, dim))

        # Useful for backward transitions
        # Aka continuous time smoothing.
        sol_mean = lambda t: sol.sol(t)[:dim]
        sol_cov = lambda t: sol.sol(t)[dim:].reshape((dim, dim))

        return randvars.Normal(mean=new_mean, cov=new_cov), {
            "sol": sol,
            "sol_mean": sol_mean,
            "sol_cov": sol_cov,
        }

    def _setup_vectorized_mde_forward_classic(self, initrv):
        """Set up forward moment differential equations (MDEs).

        Compute an ODE vector field that represents the MDEs and is compatible with
        scipy.solve_ivp.
        """
        dim = len(initrv.mean)

        def f(t, y):
            # Undo vectorization
            mean, cov_flat = y[:dim], y[dim:]
            cov = cov_flat.reshape((dim, dim))

            # Apply iteration
            G = self.drift_matrix_function(t)
            u = self.force_vector_function(t)
            L = self.dispersion_matrix_function(t)
            new_mean = G @ mean + u
            new_cov = (
                G @ cov
                + cov @ G.T
                + self.squared_scalar_diffusion_function(t) * L @ L.T
            )

            # Vectorize outcome
            new_cov_flat = new_cov.flatten()
            y_new = np.hstack((new_mean, new_cov_flat))
            return y_new

        initcov_flat = initrv.cov.flatten()
        y0 = np.hstack((initrv.mean, initcov_flat))

        return f, y0

    def _setup_vectorized_mde_forward_sqrt(self, initrv):
        r"""Set up forward moment differential equations (MDEs) using a square-root
        implementation. (https://ieeexplore.ieee.org/document/4045974)

        The covariance :math:`P(t)` obeys the Riccati equation

        .. math::
            \dot P(t) = G(t)P(t) + P(t)G^\top(t) + L(t)L^\top(t).

        Let :math:`S(t)` be a square-root of :math:`P(t)`, :math:`P(t)` positive definite, then

        .. math::
            P(t) = S(t)S^\top(t)

        and we get the Riccati-Equation

        .. math::
            \dot P(t) = G(t)S(t)S^\top(t) + 1/2 \cdot L(t)L^\top(t)S^{-\top}S^\top
                        + S(t)S^\top(t)G^\top(t) + 1/2 \cdot S(t)S^{-1}(t)L(t)L^\top(t).

        One solution can be found by the square-root :math:`\dot S(t)`

        .. math::
            \dot S(t) = G(t)S(t) + (A + 1/2 \cdot L(t)L^\top(t))S^{-\top}

        where :math:`A` is an arbitrary symmetric matrix.
        :math:`A` can be chosen to make S lower-triangular which can be achieved by

        .. math::
            M(t) = S^{-1}(t)\dot S(t) + \dot S(t)^\top S^{-\top}

        and

        .. math::
            M(t) = \bar G(t) + \bar G^\top(t) + \bar L(t) \bar L^\top(t)

        and

        .. math::
            \bar G(t) = S^{-1}(t)G(t)S(t),
            \bar L(t) = S^{-1}L(t)

        and

        .. math::
            \dot S(t) = S(t)[M(t)]_{\mathrm{lt}}

        where :math:`\mathrm{lt}` denotes the lower-triangular operator defined by

        .. math::
            [M(t)]{_{\mathrm{lt}}}_{ij} =
                \begin{cases}
                    0 & i < j\\
                    1/2 m(t)_{ij} & i=j\\
                    m(t)_{ij} & i > j
                \end{cases}.

        Compute an ODE vector field that represents the MDEs and is
        compatible with scipy.solve_ivp.
        """
        dim = len(initrv.mean)

        def f(t, y):
            # Undo vectorization
            mean, cov_cholesky_flat = y[:dim], y[dim:]
            cov_cholesky = cov_cholesky_flat.reshape((dim, dim))

            # Apply iteration
            G = self.drift_matrix_function(t)
            u = self.force_vector_function(t)
            L = self.dispersion_matrix_function(t)

            new_mean = G @ mean + u
            G_bar = scipy.linalg.solve_triangular(
                cov_cholesky, G @ cov_cholesky, lower=True
            )
            L_bar = np.sqrt(
                self.squared_scalar_diffusion_function(t)
            ) * scipy.linalg.solve_triangular(cov_cholesky, L, lower=True)
            M = G_bar + G_bar.T + L_bar @ L_bar.T

            new_cov_cholesky = tril_to_positive_tril(
                cov_cholesky @ (np.tril(M, -1) + 1 / 2 * np.diag(np.diag(M)))
            )

            # Vectorize outcome
            new_cov_cholesky_flat = new_cov_cholesky.flatten()
            y_new = np.hstack((new_mean, new_cov_cholesky_flat))
            return y_new

        initcov_cholesky_flat = initrv.cov_cholesky.flatten()
        y0 = np.hstack((initrv.mean, initcov_cholesky_flat))

        return f, y0

    def _setup_vectorized_mde_backward(self, finalrv_obtained):
        """Set up backward moment differential equations (MDEs).

        Compute an ODE vector field that represents the MDEs and is compatible with
        scipy.solve_ivp.
        """
        dim = len(finalrv_obtained.mean)

        def f(t, y, mde_forward_sol_mean, mde_forward_sol_cov):
            # Undo vectorization
            mean, cov_flat = y[:dim], y[dim:]
            cov = cov_flat.reshape((dim, dim))

            # Apply iteration
            G = self.drift_matrix_function(t)
            u = self.force_vector_function(t)
            L = self.dispersion_matrix_function(t)

            mde_forward_sol_cov_mat = mde_forward_sol_cov(t)
            mde_forward_sol_mean_vec = mde_forward_sol_mean(t)

            LL = self.squared_scalar_diffusion_function(t) * L @ L.T
            LL_inv_cov = np.linalg.solve(mde_forward_sol_cov_mat, LL.T).T

            new_mean = G @ mean + LL_inv_cov @ (mean - mde_forward_sol_mean_vec) + u
            new_cov = (G + LL_inv_cov) @ cov + cov @ (G + LL_inv_cov).T - LL

            new_cov_flat = new_cov.flatten()
            y_new = np.hstack((new_mean, new_cov_flat))
            return y_new

        finalcov_flat = finalrv_obtained.cov.flatten()
        y0 = np.hstack((finalrv_obtained.mean, finalcov_flat))

        return f, y0

    def _duplicate(self, **changes):
        """Create a new object of the same type, replacing fields with values from
        changes."""

        def replace_key(key):
            """If the key is part of the desired changes, change appropriately.

            Otherwise, take the current value.
            """
            try:
                return changes[key]
            except KeyError:
                return getattr(self, key)

        state_dimension = replace_key("state_dimension")
        wiener_process_dimension = replace_key("wiener_process_dimension")

        drift_matrix_function = replace_key("drift_matrix_function")
        dispersion_matrix_function = replace_key("dispersion_matrix_function")
        force_vector_function = replace_key("force_vector_function")
        squared_scalar_diffusion_function = replace_key(
            "squared_scalar_diffusion_function"
        )

        mde_atol = replace_key("mde_atol")
        mde_rtol = replace_key("mde_rtol")
        mde_solver = replace_key("mde_solver")
        forward_implementation = replace_key("forward_implementation")

        return LinearSDE(
            state_dimension=state_dimension,
            wiener_process_dimension=wiener_process_dimension,
            drift_matrix_function=drift_matrix_function,
            dispersion_matrix_function=dispersion_matrix_function,
            force_vector_function=force_vector_function,
            squared_scalar_diffusion_function=squared_scalar_diffusion_function,
            mde_atol=mde_atol,
            mde_rtol=mde_rtol,
            mde_solver=mde_solver,
            forward_implementation=forward_implementation,
        )