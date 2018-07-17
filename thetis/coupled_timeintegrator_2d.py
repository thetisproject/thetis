"""
Time integrators for solving coupled shallow water equations with one tracer.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from abc import ABCMeta, abstractproperty


class CoupledTimeIntegrator2D(timeintegrator.TimeIntegratorBase):
    """
    Base class of time integrator for coupled shallow water and tracer equations
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

    @abstractproperty
    def tracer_integrator(self):
        """time integrator for the tracer equation"""
        pass

    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.fields = solver.fields
        self.timesteppers = AttrDict()
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        print_output('  Shallow Water time integrator: {:}'.format(self.swe_integrator.__name__))
        print_output('  Tracer time integrator: {:}'.format(self.tracer_integrator.__name__))
        self._initialized = False

        self._create_integrators()

    def _create_swe_integrator(self):
        """
        Create time integrator for 2D system
        """
        solver = self.solver
        fields = {
            'linear_drag_coefficient': self.options.linear_drag_coefficient,
            'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
            'manning_drag_coefficient': self.options.manning_drag_coefficient,
            'viscosity_h': self.options.horizontal_viscosity,
            'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
            'coriolis': self.options.coriolis_frequency,
            'wind_stress': self.options.wind_stress,
            'atmospheric_pressure': self.options.atmospheric_pressure,
            'momentum_source': self.options.momentum_source_2d,
            'volume_source': self.options.volume_source_2d, }

        if issubclass(self.swe_integrator, timeintegrator.CrankNicolson):
            self.timesteppers.swe2d = self.swe_integrator(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters,
                semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                theta=self.options.timestepper_options.implicitness_theta)
        else:
            self.timesteppers.swe2d = self.swe_integrator(solver.eq_sw, self.fields.solution_2d,
                                                          fields, solver.dt,
                                                          bnd_conditions=solver.bnd_functions['shallow_water'],
                                                          solver_parameters=self.options.timestepper_options.solver_parameters)

    def _create_tracer_integrator(self):
        """
        Create time integrator for tracer equation
        """
        solver = self.solver

        if self.solver.options.solve_tracer:
            uv, elev = self.fields.solution_2d.split()
            fields = {'elev_2d': elev,
                      'uv_2d': uv,
                      'diffusivity_h': self.options.horizontal_diffusivity,
                      'source': self.options.tracer_source_2d,
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            if issubclass(self.tracer_integrator, timeintegrator.CrankNicolson):
                self.timesteppers.tracer = self.tracer_integrator(
                    solver.eq_tracer, solver.fields.tracer_2d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['tracer'],
                    solver_parameters=self.options.timestepper_options.solver_parameters_tracer,
                    semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                    theta=self.options.timestepper_options.implicitness_theta)
            else:
                self.timesteppers.tracer = self.tracer_integrator(solver.eq_tracer, solver.fields.tracer_2d, fields, solver.dt,
                                                                  bnd_conditions=solver.bnd_functions['tracer'],
                                                                  solver_parameters=self.options.timestepper_options.solver_parameters_tracer,)

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        self._create_swe_integrator()
        self._create_tracer_integrator()
        self.cfl_coeff_2d = min(self.timesteppers.swe2d.cfl_coeff, self.timesteppers.tracer.cfl_coeff)

    def set_dt(self, dt):
        """
        Set time step for the coupled time integrator

        :arg float dt: Time step.
        """
        for stepper in sorted(self.timesteppers):
            self.timesteppers[stepper].set_dt(dt)

    def initialize(self, solution2d):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """

        # solution2d is only provided to make a timeintegrator of just the swe
        # compatible with the 2d coupled timeintegrator
        assert solution2d == self.fields.solution_2d

        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.options.solve_tracer:
            self.timesteppers.tracer.initialize(self.fields.tracer_2d)

        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self.options.tracer_only:
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
        self.timesteppers.tracer.advance(t, update_forcings=update_forcings)
        if self.options.use_limiter_for_tracers:
            self.solver.tracer_limiter.apply(self.fields.tracer_2d)


class CoupledCrankNicolson2D(CoupledTimeIntegrator2D):
    swe_integrator = timeintegrator.CrankNicolson
    tracer_integrator = timeintegrator.CrankNicolson


class CoupledCrankEuler2D(CoupledTimeIntegrator2D):
    swe_integrator = timeintegrator.CrankNicolson
    tracer_integrator = timeintegrator.ForwardEuler
