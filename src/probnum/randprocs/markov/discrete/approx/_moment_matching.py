"""Moment matching linearization."""
import abc
import dataclasses

import numpy as np

from probnum import randvars
from probnum.randprocs.markov import _approx_transition
from probnum.randprocs.markov.discrete import _nonlinear_gaussian


# Not the same as _MomentMatchedTransition!
class MomentMatchingTransition(_approx_transition.ApproximateTransition, abc.ABC):
    """Approximate a transition by moment-matching through the non-linearities."""

    def __init__(
        self,
        non_linear_model,
    ) -> None:

        super().__init__(non_linear_model=non_linear_model)

    def linearize(self, at) -> "_MomentMatchedTransition":
        quadrule = self.assemble_quadrature_rule(at=at)
        return _MomentMatchedTransition(quadrule, self.non_linear_model)

    @abc.abstractmethod
    def assemble_quadrature_rule(self, at):
        raise NotImplementedError


class UnscentedTransformTransition(MomentMatchingTransition):
    """Moment-match with the unscented-transform."""

    def __init__(self, non_linear_model, spread=1e-4, priorpar=2.0, special_scale=0.0):
        super().__init__(non_linear_model=non_linear_model)

        self._scaling_parameter = (
            spread ** 2 * (non_linear_model.input_dim + special_scale)
            - non_linear_model.input_dim
        )

        self._mean_weights, self._cov_weights = self._unscented_weights(
            spread=spread,
            priorpar=priorpar,
            dimension=non_linear_model.input_dim,
            scale=self._scaling_parameter,
        )
        self._unit_quadrature_nodes = self._unit_nodes()

    def assemble_quadrature_rule(self, at):
        nodes = at.mean[None, :] + self._unit_quadrature_nodes @ at.cov_cholesky
        return _MomentMatchingQuadratureRule(
            nodes=nodes, mean_weights=self._mean_weights, cov_weights=self._cov_weights
        )

    @staticmethod
    def _unscented_weights(spread, priorpar, dimension, scale):
        mweights = UnscentedTransformTransition._mean_weights(dimension, scale)
        cweights = UnscentedTransformTransition._cov_weights(
            dimension, spread, priorpar, scale
        )
        return mweights, cweights

    @staticmethod
    def _mean_weights(dimension, lam):
        mw0 = np.ones(1) * lam / (dimension + lam)
        mw = np.ones(2 * dimension) / (2.0 * (dimension + lam))
        return np.hstack((mw0, mw))

    @staticmethod
    def _cov_weights(dimension, alp, bet, lam):
        cw0 = np.ones(1) * lam / (dimension + lam) + (1 - alp ** 2 + bet)
        cw = np.ones(2 * dimension) / (2.0 * (dimension + lam))
        return np.hstack((cw0, cw))

    def _unit_nodes(self):
        N = self.non_linear_model.input_dim
        sigma_pts = np.zeros((2 * N + 1, N))
        sigma_pts[1 : N + 1] = np.diag(
            np.sqrt(N * np.ones(N) + self._scaling_parameter)
        )
        sigma_pts[N + 1 : 2 * N + 1] = np.diag(
            -np.sqrt(N * np.ones(N) + self._scaling_parameter)
        )
        return sigma_pts


# Auxiliary objects


@dataclasses.dataclass
class _MomentMatchingQuadratureRule:
    """Quadrature rules for Gaussian moment matching."""

    nodes: np.ndarray
    mean_weights: np.ndarray
    cov_weights: np.ndarray


class _MomentMatchedTransition(_nonlinear_gaussian.NonlinearGaussian):
    """A non-linear, discrete, Gaussian transition that knows forward_rv (thus enables
    inference)."""

    def __init__(self, quadrature_rule, non_linear_model):
        super().__init__(
            input_dim=non_linear_model.input_dim,
            output_dim=non_linear_model.output_dim,
            state_trans_fun=non_linear_model.state_trans_fun,
            proc_noise_cov_mat_fun=non_linear_model.proc_noise_cov_mat_fun,
            jacob_state_trans_fun=non_linear_model.jacob_state_trans_fun,
            proc_noise_cov_cholesky_fun=non_linear_model.proc_noise_cov_cholesky_fun,
        )
        self.quadrature_rule = quadrature_rule
        self.non_linear_model = non_linear_model

    def forward_rv(self, rv, t, compute_gain=False, _diffusion=1.0, _linearise_at=None):
        # Already linearized.
        # The only way the `rv` enters is in the computation of the cross-covariance
        g = lambda x: self.non_linear_model.state_trans_fun(t, x)
        S = _diffusion * self.non_linear_model.proc_noise_cov_mat_fun(t)
        new_rv, info = self._propagate_moments(g, S, rv.mean)
        if compute_gain:
            info["gain"] = info["crosscov"] @ np.linalg.inv(new_rv.cov)
        return new_rv, info

    def _propagate_moments(self, transition_function, transition_cov_matrix, init_mean):
        mw, cw = self.quadrature_rule.mean_weights, self.quadrature_rule.cov_weights
        X = self.quadrature_rule.nodes
        gx = np.stack([transition_function(x) for x in X])
        new_mean = mw @ gx
        new_cov = cw @ np.outer(gx - new_mean, gx - new_mean) + transition_cov_matrix
        print(np.outer(X - init_mean[None, :], gx - new_mean).shape)
        print(cw.shape)
        new_crosscov = cw @ np.outer(X - init_mean[None, :], gx - new_mean)
        return randvars.Normal(new_mean, new_cov), {"crosscov": new_crosscov}
