"""Abstract ODESolver class.

Interface for Runge-Kutta, ODEFilter.
"""

import typing
from abc import ABC, abstractmethod

import probnum.random_variables as pnrv

from .odesolution import ODESolution


class ODESolver(ABC):
    """Interface for ODESolver."""

    def __init__(self, ivp, order):
        self.ivp = ivp
        self.order = order  # e.g.: RK45 has order=5, IBM(q) has order=q
        self.num_steps = 0

    def solve(
        self, steprule, output="dense"
    ) -> typing.Union[ODESolution, pnrv.RandomVariable]:
        """Solve an IVP.

        Parameters
        ----------
        steprule : :class:`StepRule`
            Step-size selection rule, e.g. constant steps or adaptive steps.
        output
            Dense output (`output="dense"`)  or just the final random variable (`output="only_final"`) ?
        """
        t, current_rv = self.initialise()
        if output == "dense":
            times, rvs = [t], [current_rv]
        stepsize = steprule.firststep

        while t < self.ivp.tmax:

            t_new = t + stepsize
            proposed_rv, errorest = self.step(t, t_new, current_rv)
            internal_norm = steprule.errorest_to_norm(
                errorest=errorest,
                proposed_state=proposed_rv.mean,
                current_state=current_rv.mean,
            )
            if steprule.is_accepted(internal_norm):
                self.num_steps += 1
                self.method_callback(
                    time=t_new, current_guess=proposed_rv, current_error=errorest
                )
                t = t_new
                current_rv = proposed_rv
                if output == "dense":
                    times.append(t)
                    rvs.append(current_rv)
                # 'else: output = "only_final"'

            suggested_stepsize = steprule.suggest(
                stepsize, internal_norm, localconvrate=self.order + 1
            )
            stepsize = min(suggested_stepsize, self.ivp.tmax - t)

        if output == "dense":
            odesol = self.rvlist_to_odesol(times=times, rvs=rvs)
            odesol = self.postprocess(odesol)
            return odesol
        # 'else: output = "only_final"':
        return current_rv

    @abstractmethod
    def initialise(self):
        """Returns t0 and y0 (for the solver, which might be different to ivp.y0)"""
        raise NotImplementedError

    @abstractmethod
    def step(self, start, stop, current, **kwargs):
        """Every ODE solver needs a step() method that returns a new random variable and
        an error estimate."""
        raise NotImplementedError

    @abstractmethod
    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""
        raise NotImplementedError

    def postprocess(self, odesol):
        """Process the ODESolution object before returning."""
        return odesol

    def method_callback(self, time, current_guess, current_error):
        """Optional callback.

        Can be overwritten. Do this as soon as it is clear that the
        current guess is accepted, but before storing it. No return. For
        example: tune hyperparameters (sigma).
        """
        pass
