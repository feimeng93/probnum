""""""
import numpy as np

from probnum import randprocs


def test_sth():
    markov_prior = randprocs.markov.integrator.IntegratedWienerProcess(initarg=0.0)
    randprocs.markov.MarkovProcessPosterior(markov_prior)
