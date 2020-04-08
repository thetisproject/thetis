from __future__ import absolute_import
from .utility import *
from .equation import ErrorEstimatorTerm, ErrorEstimator
from .tracer_eq_2d import TracerTerm
from .shallowwater_eq import ShallowWaterTerm

__all__ = [
    'TracerErrorEstimator2D',
    'ShallowWaterErrorEstimator',
]

g_grav = physical_constants['g_grav']


class ShallowWaterErrorEstimatorTerm(ErrorEstimatorTerm, ShallowWaterTerm):
    # TODO: doc
    def __init__(self, function_space, bathymetry=None, options=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        :kwarg options: :class:`ModelOptions2d` parameter object
        """
        ShallowWaterTerm.__init__(self, function_space, bathymetry, options)
        ErrorEstimatorTerm.__init__(self, function_space.mesh())


class TracerErrorEstimatorTerm(ErrorEstimatorTerm, TracerTerm):
    # TODO: doc
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        """
        TracerTerm.__init__(self, function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        ErrorEstimatorTerm.__init__(self, function_space.mesh())


class ExternalPressureGradientErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HUDivErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalAdvectionErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalViscosityErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class CoriolisErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class QuadraticDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TurbineDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerHorizontalAdvectionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerHorizontalDiffusionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerSourceErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class ShallowWaterEstimator(ErrorEstimator):
    # TODO: doc
    def __init__(self, function_space, bathymetry, options):
        super(ShallowWaterEstimator, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

        # Momentum terms
        args = (function_space.sub(0), bathymetry, options)
        self.add_term(ExternalPressureGradientErrorEstimatorTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityErrorEstimatorTerm(*args), 'explicit')
        self.add_term(CoriolisErrorEstimatorTerm(*args), 'explicit')
        # self.add_term(WindStressErrorEstimatorTerm(*args), 'source')  # TODO
        # self.add_term(AtmosphericPressureErrorEstimatorTerm(*args), 'source')  # TODO
        self.add_term(QuadraticDragErrorEstimatorTerm(*args), 'explicit')
        # self.add_term(LinearDragErrorEstimatorTerm(*args), 'explicit')  # TODO
        # self.add_term(BottomDrag3DErrorEstimatorTerm(*args), 'source')  # TODO
        self.add_term(TurbineDragErrorEstimatorTerm(*args), 'implicit')
        # self.add_term(MomentumSourceErrorEstimatorTerm(*args), 'source')  # TODO

        # Continuity terms
        args = (function_space.sub(1), bathymetry, options)
        self.add_term(HUDivErrorEstimatorTerm(*args), 'implicit')
        # self.add_term(ContinuitySourceErrorEstimatorTerm(*args), 'source')  # TODO


class TracerErrorEstimator2D(ErrorEstimator):
    # TODO: doc
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerErrorEstimator2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(TracerHorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceErrorEstimatorTerm(*args), 'source')
