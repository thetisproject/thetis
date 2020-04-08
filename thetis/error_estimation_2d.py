from __future__ import absolute_import
from .utility import *
from .equation import ErrorEstimatorTerm, ErrorEstimator
from .tracer_eq_2d import TracerTerm

__all__ = [
    'TracerErrorEstimatorTerm',
    'TracerErrorEstimator2D',
]


class TracerErrorEstimatorTerm(ErrorEstimatorTerm, TracerTerm):
    # TODO: doc
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        """
        TracerTerm.__init__(self, function_space)
        # TODO: I do not want jacobian method
        ErrorEstimatorTerm.__init__(self, function_space)


class HorizontalAdvectionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalDiffusionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class SourceErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerErrorEstimator2D(ErrorEstimator):
    # TODO: doc
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerErrorEstimator2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(HorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(SourceErrorEstimatorTerm(*args), 'source')
