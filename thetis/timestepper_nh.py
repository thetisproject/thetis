"""
Coupled time integrators for solving equations with the non-hydrostatic pressure.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from abc import ABCMeta


class NonhydrostaticTimeStepper(timeintegrator.TimeIntegratorBase):
    """
    Base class of time integrator for non-hydrostatic equations
    """
    __metaclass__ = ABCMeta

    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

    def fs_integrator(self):
        """time integrator for the free surface equation"""
        pass

    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.options_nh = self.options.nh_model_options
        self.fields = solver.fields
        self.timesteppers = AttrDict()
        # print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        # print_output('... using shallow water time integrator: {:}'.format(self.swe_integrator.__name__))
        # print_output('... using free surface time integrator: {:}'.format(self.fs_integrator.__name__))
        self._initialized = False

        self._create_integrators()

    def _create_integrators(self):
        """
        Creates associated time integrators with the correct arguments
        """
        solver = self.solver
        fields_fs = {
            'uv': solver.fields.solution_2d.sub(0),
            'volume_source': self.options.volume_source_2d,
        }
        self.timesteppers.fs2d = self.fs_integrator(
            solver.eq_free_surface, solver.fields.solution_2d.sub(1), fields_fs, solver.dt,
            bnd_conditions=solver.bnd_functions['shallow_water'],
            semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
            theta=self.options.timestepper_options.implicitness_theta)

        if self.options_nh.use_2d_solver:
            self.timesteppers.swe2d = solver.get_swe_timestepper(self.swe_integrator)
        else:
            fields_mom = {
                'eta': self.fields.elev_domain_2d.view_3d,  # FIXME rename elev
                'int_pg': self.fields.get('int_pg_3d'),
                'viscosity_v': expl_v_visc,
                'viscosity_h': solver.tot_h_visc.get_sum(),
                'source': self.options.momentum_source_3d,
                'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                'coriolis': self.fields.get('coriolis_3d'),
                'linear_drag_coefficient': self.options.linear_drag_coefficient,
                'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
                'wind_stress': self.fields.get('wind_stress_3d'),
                'bottom_roughness': self.options.bottom_roughness,
            } # TODO add implicit vertical diffusion option
            self.timesteppers.mom3d = self.mom_integrator(
                solver.eq_momentum, solver.fields.uv_3d, fields_mom, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)


class TimeStepper2d(NonhydrostaticTimeStepper):
    """
    Non-hydrostatic timestepper based on 2D sover
    """
    def __init__(self, solver, integrator):
        self.fs_integrator = integrator
        self.swe_integrator = integrator
        super(TimeStepper2d, self).__init__(solver)
        self.elev_old = Function(self.fields.solution_2d.sub(1))
        assert self.options_nh.use_2d_solver

    def initialize(self, solution2d):
        """
        Assign initial conditions for 2D timestepping
        """

        assert solution2d == self.fields.solution_2d

        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        self.timesteppers.fs2d.initialize(self.fields.elev_2d)

        # TODO modify accordingly to be acompatible with nh model?
        if self.options.solve_tracer:
            self.timesteppers.tracer.initialize(self.fields.tracer_2d)
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.initialize(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.initialize(self.fields.bathymetry_2d)

        self._initialized = True

    def advance(self, t, update_forcings=None):
        """
        Advances the 2D equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations
        """
        self.elev_old.assign(self.fields.elev_2d)

        if not self.options.tracer_only:
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)

        # solve non-hydrostatic pressure q
        self.solver.solver_q.solve()
        # update velocities uv_2d, w_2d
        self.solver.solver_u.solve()
        self.solver.solver_w.solve()
        # update free surface elevation
        if self.options_nh.update_free_surface:
            self.fields.elev_2d.assign(self.elev_old)
            self.timesteppers.fs2d.advance(t, update_forcings=update_forcings)

        # TODO modify accordingly to be acompatible with nh model?
        if self.options.solve_tracer:
            self.timesteppers.tracer.advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.tracer_2d)
        if self.solver.sediment_model is not None:
            self.solver.sediment_model.update()
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.advance(t, update_forcings=update_forcings)


class TimeStepper3d(NonhydrostaticTimeStepper):
    """
    Non-hydrostatic timestepper based on 3D sover
    """
    def __init__(self, solver, integrator2d, integrator3d):
        self.fs_integrator = integrator2d#timeintegrator.CrankNicolson
        self.mom_integrator = integrator3d#timeintegrator.SSPRK22ALE
        # TODO add integrator for tracer
        super(TimeStepper3d, self).__init__(solver)
        assert (not self.options_nh.use_2d_solver)

    def initialize(self, solution2d):
        """
        Assign initial conditions for 3D timestepping
        """

        pass # TODO

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the 3D equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """

        pass # TODO
