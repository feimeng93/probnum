import numpy as np
import pytest

from probnum import linops, randprocs, randvars
from tests.test_randprocs.test_markov import test_transition
from tests.test_randprocs.test_markov.test_continuous import test_sde


class TestLinearSDE(test_sde.TestSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1, spdmat2):

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2
        self.transition = randprocs.markov.continuous.LinearSDE(
            state_dimension=test_ndim,
            wiener_process_dimension=test_ndim,
            drift_matrix_function=self.G,
            force_vector_function=self.v,
            dispersion_matrix_function=self.L,
        )

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: spdmat2

    def test_drift_matrix_function(self):
        expected = self.G(0.0)
        received = self.transition.drift_matrix_function(0.0)
        np.testing.assert_allclose(expected, received)

    def test_force_vector_function(self):
        expected = self.v(0.0)
        received = self.transition.force_vector_function(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forward_rv(self, some_normal_rv1):
        out, _ = self.transition.forward_rv(some_normal_rv1, t=0.0, dt=0.1)
        assert isinstance(out, randvars.Normal)

    def test_forward_realization(self, some_normal_rv1):
        out, info = self.transition.forward_realization(
            some_normal_rv1.mean, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(
            some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.mean, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_forward_realization_value_error_caught(self, some_normal_rv1):
        """the forward realization only works if a time-increment dt is provided."""
        with pytest.raises(ValueError):
            self.transition.forward_realization(some_normal_rv1.mean, t=0.0)

    def test_backward_realization_value_error_caught(
        self, some_normal_rv1, some_normal_rv2
    ):
        """the backward realization only works if a time-increment dt is provided."""
        with pytest.raises(ValueError):
            out, _ = self.transition.backward_realization(
                some_normal_rv1.mean,
                some_normal_rv2,
                t=0.0,
            )

    #
    # def test_duplicate_with_changed_coordinates(self, some_normal_rv1, some_normal_rv2):
    #
    #     # Inexact transition, so we need a higher-res version
    #     high_res_transition = self.transition.duplicate(mde_atol=1e-10, mde_rtol=1e-10)
    #
    #     P = linops.Scaling(np.arange(3, 3 + len(some_normal_rv1.mean)))
    #     changed_transition = high_res_transition.duplicate_with_changed_coordinates(
    #         linop=P
    #     )
    #
    #     # Forward test
    #     x1, _ = high_res_transition.forward_realization(
    #         some_normal_rv1.mean, t=0.1, dt=0.1
    #     )
    #     x2, _ = changed_transition.forward_realization(
    #         P.inv() @ some_normal_rv1.mean, t=0.1, dt=0.1
    #     )
    #     np.testing.assert_allclose(P @ x2.mean, x1.mean)
    #     np.testing.assert_allclose(P @ x2.cov @ P.T, x1.cov)


@pytest.fixture
def G_const():
    return np.array([[0.0, 1.0], [0.0, 0.0]])


@pytest.fixture
def v_const():
    return np.array([1.0, 1.0])


@pytest.fixture
def L_const():
    return np.array([[0.0], [1.0]])


@pytest.fixture
def ltisde_as_linearsde(G_const, v_const, L_const):
    G = lambda t: G_const
    v = lambda t: v_const
    L = lambda t: L_const

    return randprocs.markov.continuous.LinearSDE(
        state_dimension=G_const.shape[0],
        wiener_process_dimension=L_const.shape[1],
        drift_matrix_function=G,
        force_vector_function=v,
        dispersion_matrix_function=L,
        mde_atol=1e-12,
        mde_rtol=1e-12,
        forward_implementation="classic",
    )


@pytest.fixture
def ltisde_as_linearsde_sqrt_forward_implementation(G_const, v_const, L_const):
    G = lambda t: G_const
    v = lambda t: v_const
    L = lambda t: L_const

    return randprocs.markov.continuous.LinearSDE(
        state_dimension=G_const.shape[0],
        wiener_process_dimension=L_const.shape[1],
        drift_matrix_function=G,
        force_vector_function=v,
        dispersion_matrix_function=L,
        mde_atol=1e-12,
        mde_rtol=1e-12,
        forward_implementation="sqrt",
    )


@pytest.fixture
def ltisde(G_const, v_const, L_const):
    return randprocs.markov.continuous.LTISDE(
        drift_matrix=G_const, force_vector=v_const, dispersion_matrix=L_const
    )


def test_solve_mde_forward_values(ltisde_as_linearsde, ltisde, v_const):
    out_linear, _ = ltisde_as_linearsde.forward_realization(v_const, t=0.0, dt=0.1)
    out_lti, _ = ltisde.forward_realization(v_const, t=0.0, dt=0.1)

    np.testing.assert_allclose(out_linear.mean, out_lti.mean)
    np.testing.assert_allclose(out_linear.cov, out_lti.cov)


def test_solve_mde_forward_sqrt_values(
    ltisde_as_linearsde,
    ltisde_as_linearsde_sqrt_forward_implementation,
    v_const,
):
    """mde forward values in sqrt-implementation and classic implementation should be
    equal."""
    out_linear, _ = ltisde_as_linearsde.forward_realization(v_const, t=0.0, dt=0.1)

    out_linear_2, _ = ltisde_as_linearsde.forward_rv(out_linear, t=0.1, dt=0.1)
    out_linear_2_sqrt, _ = ltisde_as_linearsde_sqrt_forward_implementation.forward_rv(
        out_linear, t=0.1, dt=0.1
    )

    np.testing.assert_allclose(out_linear_2_sqrt.mean, out_linear_2.mean)
    np.testing.assert_allclose(out_linear_2_sqrt.cov, out_linear_2.cov)


def test_solve_mde_backward_values(ltisde_as_linearsde, ltisde, v_const):
    out_linear_forward, _ = ltisde_as_linearsde.forward_realization(
        v_const, t=0.0, dt=0.1
    )
    out_lti_forward, _ = ltisde.forward_realization(v_const, t=0.0, dt=0.1)
    out_linear_forward_next, _ = ltisde_as_linearsde.forward_rv(
        out_linear_forward, t=0.1, dt=0.1
    )
    out_lti_forward_next, _ = ltisde.forward_rv(out_lti_forward, t=0.1, dt=0.1)

    out_linear, _ = ltisde_as_linearsde.backward_realization(
        realization_obtained=out_linear_forward_next.mean,
        rv=out_linear_forward,
        t=0.1,
        dt=0.1,
    )
    out_lti, _ = ltisde.backward_realization(
        realization_obtained=out_lti_forward_next.mean,
        rv=out_lti_forward,
        t=0.1,
        dt=0.1,
    )

    np.testing.assert_allclose(out_linear.mean, out_lti.mean)
    np.testing.assert_allclose(out_linear.cov, out_lti.cov, atol=1e-9)
