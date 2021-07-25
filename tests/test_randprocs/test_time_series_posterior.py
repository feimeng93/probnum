""""""
import numpy as np

from probnum import randprocs


def test_sth():
    randprocs.TimeSeriesPosterior(output_dim=1, dtype=np.float64)
