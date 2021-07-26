"""Transition utilities."""


import abc
import copy

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.typing import FloatArgType, IntArgType


def smooth_list(rv_list, locations, transition_list, _previous_posterior=None):
    """Apply smoothing to a list of random variables, according to the present
    transition.

    Returns
    -------
    _randomvariablelist._RandomVariableList
        List of smoothed random variables.
    """

    final_rv = rv_list[-1]

    curr_rv = final_rv
    out_rvs = [curr_rv]
    for idx in reversed(range(1, len(locations))):
        unsmoothed_rv = rv_list[idx - 1]
        transition = transition_list[idx - 1]

        _linearise_smooth_step_at = (
            None
            if _previous_posterior is None
            else _previous_posterior(locations[idx - 1])
        )

        # Actual smoothing step
        curr_rv, _ = transition.backward_rv(
            curr_rv,
            unsmoothed_rv,
            t=locations[idx - 1],
            dt=locations[idx] - locations[idx - 1],
            _linearise_at=_linearise_smooth_step_at,
        )
        out_rvs.append(curr_rv)
    out_rvs.reverse()
    return _randomvariablelist._RandomVariableList(out_rvs)


def jointly_transform_base_measure_realization_list_backward(
    base_measure_realizations: np.ndarray,
    t: FloatArgType,
    rv_list: _randomvariablelist._RandomVariableList,
    transition_list,
    _previous_posterior=None,
) -> np.ndarray:
    curr_rv = rv_list[-1]

    curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
        -1
    ].reshape((-1,))
    out_samples = [curr_sample]

    for idx in reversed(range(1, len(t))):
        unsmoothed_rv = rv_list[idx - 1]
        transition = transition_list[idx - 1]
        _linearise_smooth_step_at = (
            None if _previous_posterior is None else _previous_posterior(t[idx - 1])
        )

        # Condition on the 'future' realization and sample
        dt = t[idx] - t[idx - 1]
        curr_rv, _ = transition.backward_realization(
            curr_sample,
            unsmoothed_rv,
            t=t[idx - 1],
            dt=dt,
            _linearise_at=_linearise_smooth_step_at,
        )
        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            idx - 1
        ].reshape(
            -1,
        )
        out_samples.append(curr_sample)

    out_samples.reverse()
    return out_samples


def jointly_transform_base_measure_realization_list_forward(
    base_measure_realizations: np.ndarray,
    t: FloatArgType,
    initrv: randvars.RandomVariable,
    transition_list,
    _previous_posterior=None,
) -> np.ndarray:

    curr_rv = initrv

    curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
        0
    ].reshape((-1,))
    out_samples = [curr_sample]

    for idx in range(1, len(t)):

        _linearise_prediction_step_at = (
            None if _previous_posterior is None else _previous_posterior(t[idx - 1])
        )

        transition = transition_list[idx - 1]
        dt = t[idx] - t[idx - 1]
        curr_rv, _ = transition.forward_realization(
            curr_sample,
            t=t[idx - 1],
            dt=dt,
            _linearise_at=_linearise_prediction_step_at,
        )
        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            idx - 1
        ].reshape((-1,))
        out_samples.append(curr_sample)
    return out_samples
