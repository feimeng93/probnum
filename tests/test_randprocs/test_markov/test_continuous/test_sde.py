import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov import test_transition


class TestSDE(test_transition.InterfaceTestTransition):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.l = lambda t, x: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = randprocs.markov.continuous.SDE(
            state_dimension=test_ndim,
            wiener_process_dimension=test_ndim,
            drift_function=self.g,
            dispersion_function=self.l,
            drift_jacobian=self.dg,
        )

    # Test access to system matrices

    def test_drift(self, some_normal_rv1):
        expected = self.g(0.0, some_normal_rv1.mean)
        received = self.transition.drift_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_dispersion(self, some_normal_rv1):
        expected = self.l(0.0, some_normal_rv1.mean)
        received = self.transition.dispersion_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_jacobfun(self, some_normal_rv1):
        expected = self.dg(0.0, some_normal_rv1.mean)
        received = self.transition.drift_jacobian(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0, dt=0.1)

    def test_forward_realization(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_realization(some_normal_rv1.mean, 0.0, dt=0.1)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2, 0.0, dt=0.1)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.mean, some_normal_rv2, 0.0, dt=0.1
            )

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim

    def test_state_dimension(self, test_ndim):
        assert self.transition.state_dimension == test_ndim

    def test_wiener_process_dimension(self, test_ndim):
        assert self.transition.wiener_process_dimension == test_ndim


class TestLinearSDE(TestSDE):

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

    def test_driftmatfun(self):
        expected = self.G(0.0)
        received = self.transition.drift_matrix_function(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forcevecfun(self):
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


class TestLTISDE(TestLinearSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.G_const = spdmat1
        self.v_const = np.arange(test_ndim)
        self.L_const = spdmat2

        self.transition = randprocs.markov.continuous.LTISDE(
            drift_matrix=self.G_const,
            force_vector=self.v_const,
            dispersion_matrix=self.L_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: spdmat2

    def test_discretize(self):
        out = self.transition.discretize(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.DiscreteLTIGaussian)

    def test_discretize_no_force(self):
        """LTISDE.discretize() works if there is zero force (there is an "if" in the
        fct)."""
        new_trans = self.transition.duplicate(
            force_vector=0.0 * self.transition.force_vector
        )

        assert (
            np.linalg.norm(new_trans.force_vector_function(0.0)) == 0.0
        )  # side quest/test
        out = new_trans.discretize(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.DiscreteLTIGaussian)

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
