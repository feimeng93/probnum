"""Particle filters."""

import itertools
from collections import abc
from typing import Iterable, Optional, Union

import numpy as np

from probnum import problems, randprocs, randvars, statespace
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.type import FloatArgType, IntArgType

from ._importance_distributions import ImportanceDistribution
from ._particle_filter_posterior import ParticleFilterPosterior

# Terribly long variable names, but internal only, so no worries.
ParticleFilterMeasurementModelArgType = Union[
    statespace.DiscreteGaussian, Iterable[statespace.DiscreteGaussian]
]
ParticleFilterLinearisedMeasurementModelArgType = Union[
    statespace.DiscreteGaussian, Iterable[statespace.DiscreteGaussian]
]


def effective_number_of_events(categ_rv: randvars.Categorical) -> float:
    """Approximate effective number of events in the support of a categorical random
    variable.

    In a particle filter, this is used as the effective number of
    particles which may indicate the need for resampling.
    """
    return 1.0 / np.sum(categ_rv.probabilities ** 2)


class ParticleFilter(BayesFiltSmooth):
    r"""Particle filter (PF). Also known as sequential Monte Carlo method.

    A PF estimates the posterior distribution of a Markov process given noisy, non-linear observations,
    with a set of particles.

    The random state of the particle filter is inferred from the random state of the initial random variable.

    Parameters
    ----------
    dynamics_model :
        Prior dynamics. Since the PF is essentially a discrete-time algorithm,
        the prior must be a discrete model (or at least one with an equivalent discretisation).
        This transition must support `forward_realization`.
    initrv :
        Initial random variable. Can be any `RandomVariable` object that implements `sample()`.
    num_particles :
        Number of particles to use.
    with_resampling :
        Whether after each step the effective number of particles shall be checked, and, if too low,
        the state should be resampled. Optional. Default is `True`.
    resampling_percentage_threshold :
        Percentage threshold for resampling. That is, it is the value :math:`p` such that
        resampling is performed if :math:`N_{\text{eff}} < p \, N_\text{particles}` holds.
        Optional. Default is 0.1. If this value is non-positive, resampling is never performed.
        If it is larger than 1, resampling is performed after each step.
    """

    def __init__(
        self,
        prior_process: randprocs.MarkovProcess,
        num_particles: IntArgType,
        with_resampling: bool = True,
        resampling_percentage_threshold: FloatArgType = 0.1,
    ) -> None:
        super().__init__(
            prior_process=prior_process,
        )
        self.num_particles = num_particles

        self.with_resampling = with_resampling
        self.resampling_percentage_threshold = resampling_percentage_threshold
        self.min_effective_num_of_particles = (
            resampling_percentage_threshold * num_particles
        )

    def filter(
        self,
        regression_problem: problems.RegressionProblem,
        measurement_model: ParticleFilterMeasurementModelArgType,
        importance_distribution: ImportanceDistribution,
        linearized_measurement_model: Optional[
            ParticleFilterLinearisedMeasurementModelArgType
        ] = None,
    ):
        """Apply particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
        measurement_model :
            Measurement model. Must be a discrete model that supports `forward_realization`.
        linearized_measurement_model :
            Linearized measurement model that is used as an importance density. In principle,
            any discrete-time model that supports `backward_realization` is applicable.
            In practice, it will almost always be one out of `DiscreteEKFComponent`, `DiscreteUKFComponent`,
            or `IteratedDiscreteComponent`. Linear components are also possible, but would most often imply
            that a particle filter is not required, because the filtering problem can be used much faster
            with a Kalman filter. The exception to this rule is if the initial random variable is not Gaussian.
            Optional. Default is None, which implies the bootstrap PF: the importance samples are generated by sampling from the prior.

        Returns
        -------
        posterior
            Posterior distribution of the filtered output
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """
        filtered_rvs = []
        info_dicts = []

        for rv, info in self.filter_generator(
            regression_problem,
            measurement_model,
            importance_distribution,
            linearized_measurement_model,
        ):
            filtered_rvs.append(rv)
            info_dicts.append(info)

        posterior = ParticleFilterPosterior(
            states=filtered_rvs,
            locations=regression_problem.locations,
        )

        return posterior, info_dicts

    def filter_generator(
        self,
        regression_problem: problems.RegressionProblem,
        measurement_model: ParticleFilterMeasurementModelArgType,
        importance_distribution: ImportanceDistribution,
        linearized_measurement_model: Optional[
            ParticleFilterLinearisedMeasurementModelArgType
        ] = None,
    ):
        """Apply Particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
        measurement_model :
            Measurement model. Must be a discrete model that supports `forward_realization`.
        linearized_measurement_model :
            Linearized measurement model that is used as an importance density. In principle,
            any discrete-time model that supports `backward_realization` is applicable.
            In practice, it will almost always be one out of `DiscreteEKFComponent`, `DiscreteUKFComponent`,
            or `IteratedDiscreteComponent`. Linear components are also possible, but would most often imply
            that a particle filter is not required, because the filtering problem can be used much faster
            with a Kalman filter. The exception to this rule is if the initial random variable is not Gaussian.
            Optional. Default is None, which implies the bootstrap PF: the importance samples are generated by sampling from the prior.

        Yields
        ------
        curr_rv
            Filtering random variable at each grid point.
        info_dict
            Dictionary containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """

        dataset, times = regression_problem.observations, regression_problem.locations

        # It is not clear to me at the moment how to handle this.
        if not np.all(np.diff(times) > 0):
            raise ValueError(
                "Particle filtering cannot handle repeating time points currently."
                "Or, rather: if you know how, teach me."
            )

        if not isinstance(measurement_model, abc.Iterable):
            measurement_model = itertools.repeat(measurement_model, len(times))
        if not isinstance(linearized_measurement_model, abc.Iterable):
            linearized_measurement_model = itertools.repeat(
                linearized_measurement_model, len(times)
            )

        initarg = times[0]
        t_old = times[0]  # will be replaced by initarg soon.

        particles = np.nan * np.ones(
            (self.num_particles,) + self.prior_process.initrv.shape
        )
        weights = np.ones(self.num_particles) / self.num_particles

        for t, data, measmod, lin_measmod in zip(
            times, dataset, measurement_model, linearized_measurement_model
        ):

            dt = t - t_old
            new_particles = particles.copy()
            new_weights = weights.copy()

            # Initialization: no .apply, but .process_initrv_wiht_data...
            if t == initarg:
                particle_generator = self.importance_rv_generator_initial(
                    importance_distribution,
                    particles,
                    weights,
                    data,
                    t_old,
                    dt,
                    t,
                    lin_measmod,
                )
            else:
                particle_generator = self.importance_rv_generator_later(
                    importance_distribution,
                    particles,
                    weights,
                    data,
                    t_old,
                    dt,
                    t,
                    lin_measmod,
                )

            for idx, (importance_rv, dynamics_rv, p, w) in enumerate(
                particle_generator
            ):

                new_particle = importance_rv.sample()
                meas_rv, _ = measmod.forward_realization(new_particle, t=t)
                loglikelihood = meas_rv.logpdf(data)
                log_correction_factor = importance_distribution.log_correction_factor(
                    proposal_state=new_particle,
                    importance_rv=importance_rv,
                    dynamics_rv=dynamics_rv,
                    old_weight=w,
                )
                new_weight = np.exp(loglikelihood + log_correction_factor)

                new_particles[idx] = new_particle
                new_weights[idx] = new_weight

            weights = new_weights / np.sum(new_weights)
            particles = new_particles
            new_rv = randvars.Categorical(
                support=particles, probabilities=weights, random_state=self.random_state
            )

            if self.with_resampling:
                if (
                    effective_number_of_events(new_rv)
                    < self.min_effective_num_of_particles
                ):
                    new_rv = new_rv.resample()
            yield new_rv, {}

    def importance_rv_generator_initial(
        self,
        importance_distribution,
        particles,
        weights,
        data,
        t_old,
        dt,
        t,
        lin_measmod,
    ):

        processed = importance_distribution.process_initrv_with_data(
            self.prior_process.initrv, data, t, lin_measmod
        )
        importance_rv, dynamics_rv, _ = processed
        for (p, w) in zip(particles, weights):
            yield importance_rv, dynamics_rv, p, w

    def importance_rv_generator_later(
        self,
        importance_distribution,
        particles,
        weights,
        data,
        t_old,
        dt,
        t,
        lin_measmod,
    ):

        for (p, w) in zip(particles, weights):
            importance_rv, dynamics_rv, _ = importance_distribution.apply(
                p, data, t_old, dt, lin_measurement_model=lin_measmod
            )
            yield importance_rv, dynamics_rv, p, w

    @property
    def random_state(self):
        """Random state of the particle filter.

        Inferred from the random state of the initial random variable.
        """
        return self.prior_process.initrv.random_state
