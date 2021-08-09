"""Tests for Taylor-series transitions."""

import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import filtsmooth as filtsmooth_zoo


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1)


@pytest.fixture
def pendulum(rng):
    return filtsmooth_zoo.pendulum(rng)


@pytest.fixture
def non_linear_model(pendulum):
    problem, *_ = pendulum
    return problem.measurement_models[0]


@pytest.fixture
def linearizing_model(non_linear_model):
    return randprocs.markov.discrete.approx.LocallyLinearizingTransition(
        non_linear_model
    )


@pytest.fixture
def initvals(pendulum):
    _, info = pendulum
    return (info["prior_process"].initrv, info["prior_process"].initarg)


def test_forward_rv(non_linear_model, linearizing_model, initvals):
    rv, t = initvals
    print(non_linear_model, linearizing_model, rv)

    with pytest.raises(NotImplementedError):
        non_linear_model.forward_rv(rv, t)

    out, _ = linearizing_model.forward_rv(rv, t)

    assert isinstance(out, randvars.RandomVariable)


def test_backward_rv(non_linear_model, linearizing_model, initvals):
    rv, t = initvals
    print(non_linear_model, linearizing_model, rv)

    with pytest.raises(NotImplementedError):
        non_linear_model.backward_rv(rv[0], rv)

    out, _ = linearizing_model.backward_rv(rv[0], rv)

    assert isinstance(out, randvars.RandomVariable)


def test_approximate(linearizing_model, initvals):
    rv, _ = initvals
    linear_model = linearizing_model.linearize(at=rv)
    assert isinstance(linear_model, randprocs.markov.discrete.LinearGaussian)
