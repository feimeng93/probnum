# """Time-series algorithms."""
#
# import abc
#
# from probnum import randprocs
# from probnum.filtsmooth import _time_series_state, _time_series_posterior
#
#
# class TimeSeriesAlgorithm(abc.ABC):
#     """Time-series algorithms."""
#     @abc.abstractmethod
#     def solve(self, problem) -> _time_series_posterior.TimeSeriesPosterior:
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def solution_generator(self, problem):
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def initialize(self, problem) -> _time_series_state.TimeSeriesState:
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def perform_step(self, state: _time_series_state.TimeSeriesState)-> _time_series_state.TimeSeriesState:
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def finalize(self) -> _time_series_posterior.TimeSeriesPosterior:
#         raise NotImplementedError
