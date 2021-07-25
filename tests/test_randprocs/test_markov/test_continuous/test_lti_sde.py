import numpy as np
import pytest

from probnum import linops, randprocs, randvars
from tests.test_randprocs.test_markov import test_transition
from tests.test_randprocs.test_markov.test_continuous import test_linear_sde


class TestLTISDE(test_linear_sde.TestLinearSDE):

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

    #
    # def test_duplicate_with_changed_coordinates(self, some_normal_rv1, some_normal_rv2):
    #
    #     P = linops.aslinop(
    #         linops.Scaling(np.arange(10, 10 + len(some_normal_rv1.mean))).todense()
    #     )
    #     # P = linops.Scaling(np.ones(len(some_normal_rv1.mean)))
    #
    #     # Sanity check
    #     eye_dxd = np.eye(P.shape[0])
    #     eye_approx = P.inv() @ P
    #     np.testing.assert_allclose(eye_approx.todense(), eye_dxd)
    #     changed_transition = self.transition.duplicate_with_changed_coordinates(linop=P)
    #
    #     disc1 = self.transition.discretize(dt=0.1)
    #     disc2 = changed_transition.discretize(dt=0.1)
    #     print(disc1.state_trans_mat, P @ disc2.state_trans_mat @ P.inv())
    #     print(disc1.proc_noise_cov_mat, P @ disc2.proc_noise_cov_mat @ P)
    #
    #     #
    #     print("something is wrong with this shift vector here?!?!")
    #     assert False
    #     print(disc1.shift_vec, P @ disc2.shift_vec)
    #     print()
    #     # Forward test
    #     m, C = some_normal_rv1.mean, some_normal_rv1.cov
    #     changed_rv = randvars.Normal(P @ m, P @ C @ P.T)
    #
    #     expected, _ = self.transition.forward_realization(
    #         some_normal_rv1.mean, t=0.1, dt=0.1
    #     )
    #     x, _ = changed_transition.forward_realization(changed_rv.mean, t=0.1, dt=0.1)
    #     received = P.inv() @ x
    #
    #     print(expected.mean, received.mean)
    #
    #     np.testing.assert_allclose(received.mean, expected.mean)
    #     np.testing.assert_allclose(received.cov, expected.cov)
