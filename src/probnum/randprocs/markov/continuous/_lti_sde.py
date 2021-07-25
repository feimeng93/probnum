"""Linear, time-invariante SDEs."""

from probnum.randprocs.markov.continuous import _linear_sde


class LTISDE(_linear_sde.LinearSDE):
    """Linear time-invariant continuous Markov models of the form.

    .. math:: d x(t) = [G x(t) + v] d t + L d w(t).

    In the language of dynamic models,
    x(t) : state process
    G : drift matrix
    v : force term/vector
    L : dispersion matrix.
    w(t) : Wiener process with unit diffusion.

    Parameters
    ----------
    drift_matrix :
        This is F. It is the drift matrix of the SDE.
    force_vector :
        This is U. It is the force vector of the SDE.
    dispersion_matrix :
        This is L. It is the dispersion matrix of the SDE.
    """

    def __init__(
        self,
        drift_matrix: np.ndarray,
        force_vector: np.ndarray,
        dispersion_matrix: np.ndarray,
        squared_scalar_diffusion: Optional[FloatArgType] = 1.0,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # Assert all shapes match
        _check_initial_state_dimensions(drift_matrix, force_vector, dispersion_matrix)

        # Convert input into super() compatible format and initialize super()
        state_dimension = drift_matrix.shape[0]
        wiener_process_dimension = dispersion_matrix.shape[1]

        def drift_matrix_function(t):
            return drift_matrix

        def force_vector_function(t):
            return force_vector

        def dispersion_matrix_function(t):
            return dispersion_matrix

        def squared_scalar_diffusion_function(t):
            return squared_scalar_diffusion

        super().__init__(
            state_dimension=state_dimension,
            wiener_process_dimension=wiener_process_dimension,
            drift_matrix_function=drift_matrix_function,
            dispersion_matrix_function=dispersion_matrix_function,
            force_vector_function=force_vector_function,
            squared_scalar_diffusion_function=squared_scalar_diffusion_function,
        )

        # Initialize remaining attributes
        self.drift_matrix = drift_matrix
        self.force_vector = force_vector
        self.dispersion_matrix = dispersion_matrix
        self.squared_scalar_diffusion = squared_scalar_diffusion
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )
        discretized_model = self.discretize(dt=dt)
        return discretized_model.forward_rv(rv, t, compute_gain=compute_gain)

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
        discretized_model = self.discretize(dt=dt)
        return discretized_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
        )

    @functools.lru_cache(maxsize=None)
    def discretize(self, dt):
        """Return a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """

        if np.linalg.norm(self.force_vector) > 0:
            zeros = np.zeros((self.state_dimension, self.state_dimension))
            eye = np.eye(self.state_dimension)
            driftmat = np.block([[self.drift_matrix, eye], [zeros, zeros]])
            dispmat = np.concatenate(
                (self.dispersion_matrix, np.zeros(self.dispersion_matrix.shape))
            )
            ah_stack, qh_stack, _ = _utils.matrix_fraction_decomposition(
                driftmat, dispmat, dt
            )
            proj = np.eye(self.state_dimension, 2 * self.state_dimension)
            proj_rev = np.flip(proj, axis=1)
            ah = proj @ ah_stack @ proj.T
            sh = proj @ ah_stack @ proj_rev.T @ self.force_vector
            qh = proj @ qh_stack @ proj.T
        else:
            ah, qh, _ = _utils.matrix_fraction_decomposition(
                self.drift_matrix, self.dispersion_matrix, dt
            )
            sh = np.zeros(len(ah))
        return discrete.DiscreteLTIGaussian(
            ah,
            sh,
            qh,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )

    def duplicate(self, **changes):
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

        drift_matrix = replace_key("drift_matrix")
        dispersion_matrix = replace_key("dispersion_matrix")
        force_vector = replace_key("force_vector")
        squared_scalar_diffusion = replace_key("squared_scalar_diffusion")
        forward_implementation = replace_key("forward_implementation")
        backward_implementation = replace_key("backward_implementation")

        return LTISDE(
            drift_matrix=drift_matrix,
            dispersion_matrix=dispersion_matrix,
            force_vector=force_vector,
            squared_scalar_diffusion=squared_scalar_diffusion,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


def _check_initial_state_dimensions(drift_matrix, force_vector, dispersion_matrix):
    """Checks that the matrices all align and are of proper shape.

    Parameters
    ----------
    drift_matrix : np.ndarray, shape=(n, n)
    force_vector : np.ndarray, shape=(n,)
    dispersion_matrix : np.ndarray, shape=(n, s)
    """
    if drift_matrix.ndim != 2 or drift_matrix.shape[0] != drift_matrix.shape[1]:
        raise ValueError("drift_matrix not of shape (n, n)")
    if force_vector.ndim != 1:
        raise ValueError("force_vector not of shape (n,)")
    if force_vector.shape[0] != drift_matrix.shape[1]:
        raise ValueError(
            "force_vector not of shape (n,) or drift_matrix not of shape (n, n)"
        )
    if dispersion_matrix.ndim != 2:
        raise ValueError("dispersion_matrix not of shape (n, s)")
