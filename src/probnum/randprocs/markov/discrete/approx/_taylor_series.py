"""Discrete Taylor-series approximations."""


from randprocs.markov import _approx_transition

__all__ = ["LocallyLinearizingTransition"]


class LocallyLinearizingTransition(_approx_transition.ApproximateTransition):
    def __init__(
        self,
        non_linear_model,
        forward_implementation="classic",
        backward_implementation="classic",
    ) -> None:
        super().__init__(non_linear_model=non_linear_model)

        self._forward_implementation_string = forward_implementation
        self._backward_implementation_string = backward_implementation

    def linearize(self, at: randvars.RandomVariable) -> _transition.Transition:
        """Linearize with a first order Taylor expansion around the mean."""

        g = self.non_linear_model.state_trans_fun
        dg = self.non_linear_model.jacob_state_trans_fun

        x0 = at.mean

        def forcevecfun(t):
            return g(t, x0) - dg(t, x0) @ x0

        def dynamicsmatfun(t):
            return dg(t, x0)

        return randprocs.markov.discrete.LinearGaussian(
            input_dim=self.non_linear_model.input_dim,
            output_dim=self.non_linear_model.output_dim,
            state_trans_mat_fun=dynamicsmatfun,
            shift_vec_fun=forcevecfun,
            proc_noise_cov_mat_fun=self.non_linear_model.proc_noise_cov_mat_fun,
            proc_noise_cov_cholesky_fun=self.non_linear_model.proc_noise_cov_cholesky_fun,
            forward_implementation=self._forward_implementation_string,
            backward_implementation=self._backward_implementation_string,
        )
