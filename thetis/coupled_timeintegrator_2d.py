"""
Time integrators for solving coupled shallow water equations with one tracer.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from abc import ABCMeta


class CoupledTimeIntegrator2D(timeintegrator.TimeIntegratorBase):
    """
    Base class of time integrator for coupled shallow water and tracer equations
    """
    __metaclass__ = ABCMeta

    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

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

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        self.timesteppers.swe2d = self.solver.get_swe_timestepper(self.swe_integrator)
        if self.solver.options.solve_tracer:
            self.timesteppers.tracer = self.solver.get_tracer_timestepper(self.tracer_integrator)
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


class CoupledMatchingTimeIntegrator2D(CoupledTimeIntegrator2D):
    def __init__(self, solver, integrator):
        self.swe_integrator = integrator
        self.tracer_integrator = integrator
        super(CoupledMatchingTimeIntegrator2D, self).__init__(solver)


class CoupledCrankEuler2D(CoupledTimeIntegrator2D):
    swe_integrator = timeintegrator.CrankNicolson
    tracer_integrator = timeintegrator.ForwardEuler
