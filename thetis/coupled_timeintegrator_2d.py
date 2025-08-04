"""
Time integrators for solving coupled shallow water equations with one tracer or sediment.
"""
from .utility import *
from . import timeintegrator
from .log import *
from abc import ABC


class CoupledTimeIntegrator2D(timeintegrator.TimeIntegratorBase, ABC):
    """
    Base class of time integrator for coupled shallow water and tracer/sediment equations and exner equation
    """
    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

    def tracer_integrator(self):
        """time integrator for the tracer equation"""
        pass

    def exner_integrator(self):
        """time integrator for the exner equation"""
        pass

    @PETSc.Log.EventDecorator("thetis.CoupledTimeIntegrator2D.__init__")
    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.fields = solver.fields
        self.timesteppers = AttrDict()
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        if not self.options.tracer_only:
            print_output('  Shallow Water time integrator: {:}'.format(self.swe_integrator.__name__))
        if self.solver.solve_tracer:
            print_output('  Tracer time integrator: {:}'.format(self.tracer_integrator.__name__))
        if self.options.sediment_model_options.solve_suspended_sediment:
            print_output('  Sediment time integrator: {:}'.format(self.sediment_integrator.__name__))
        if self.options.sediment_model_options.solve_exner:
            print_output('  Exner time integrator: {:}'.format(self.exner_integrator.__name__))
        self._initialized = False

        self._create_integrators()

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        if not self.options.tracer_only:
            self.timesteppers.swe2d = self.solver.get_swe_timestepper(self.swe_integrator)
        for system in self.options.tracer_fields:
            self.timesteppers[system] = self.solver.get_tracer_timestepper(self.tracer_integrator, system)
        if self.solver.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment = self.solver.get_sediment_timestepper(self.sediment_integrator)
        if self.solver.options.sediment_model_options.solve_exner:
            self.timesteppers.exner = self.solver.get_exner_timestepper(self.exner_integrator)

    def set_dt(self, dt):
        """
        Set time step for the coupled time integrator

        :arg float dt: Time step.
        """
        for stepper in sorted(self.timesteppers):
            self.timesteppers[stepper].set_dt(dt)

    @PETSc.Log.EventDecorator("thetis.CoupledTimeIntegrator2D.initialize")
    def initialize(self, solution2d):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """

        # solution2d is only provided to make a timeintegrator of just the swe
        # compatible with the 2d coupled timeintegrator
        assert solution2d == self.fields.solution_2d

        if not self.options.tracer_only:
            self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        for system in self.options.tracer_fields:
            self.timesteppers[system].initialize(self.fields[system])
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.initialize(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.initialize(self.fields.bathymetry_2d)

        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.CoupledTimeIntegrator2D.advance")
    def advance(self, t, update_forcings=None):
        if self.options.tracer_picard_iterations > 1:
            self.advance_picard(t, update_forcings=update_forcings)
        else:
            if not self.options.tracer_only:
                self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
            for system in self.options.tracer_fields:
                self.timesteppers[system].advance(t, update_forcings=update_forcings)
                if self.options.use_limiter_for_tracers:
                    if ',' in system:
                        raise NotImplementedError("Slope limiters not supported for mixed systems of tracers")
                    self.solver.tracer_limiter.apply(self.fields[system])
            if self.solver.sediment_model is not None:
                self.solver.sediment_model.update()
            if self.options.sediment_model_options.solve_suspended_sediment:
                self.timesteppers.sediment.advance(t, update_forcings=update_forcings)
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.sediment_2d)
            if self.options.sediment_model_options.solve_exner:
                self.timesteppers.exner.advance(t, update_forcings=update_forcings)

    @PETSc.Log.EventDecorator("thetis.CoupledTimeIntegrator2D.advance_picard")
    def advance_picard(self, t, update_forcings=None):
        if not self.options.tracer_only:
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
        p = self.options.tracer_picard_iterations
        for i in range(p):
            kwargs = {'update_lagged': i == 0, 'update_fields': i == p-1}
            for system in self.options.tracer_fields:
                self.timesteppers[system].advance_picard(t, update_forcings=update_forcings, **kwargs)
                if self.options.use_limiter_for_tracers:
                    if ',' in system:
                        raise NotImplementedError("Slope limiters not supported for mixed systems of tracers")
                    self.solver.tracer_limiter.apply(self.fields[system])
        if self.solver.sediment_model is not None:
            self.solver.sediment_model.update()
        if self.options.sediment_model_options.solve_suspended_sediment:
            self.timesteppers.sediment.advance(t, update_forcings=update_forcings)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.sediment_2d)
        if self.options.sediment_model_options.solve_exner:
            self.timesteppers.exner.advance(t, update_forcings=update_forcings)


class GeneralCoupledTimeIntegrator2D(CoupledTimeIntegrator2D):
    """
    A :class:`CoupledTimeIntegrator2D` which supports
    a general set of time integrators for the different
    components.
    """
    def __init__(self, solver, integrators):
        """
        :arg solver: the :class:`FlowSolver2d` object
        :arg integrators: dictionary of time integrators
            to be used for each equation
        """
        if not solver.options.tracer_only:
            self.swe_integrator = integrators['shallow_water']
        if solver.solve_tracer:
            self.tracer_integrator = integrators['tracer']
        if solver.options.sediment_model_options.solve_suspended_sediment:
            self.sediment_integrator = integrators['sediment']
        if solver.options.sediment_model_options.solve_exner:
            self.exner_integrator = integrators['exner']
        super(GeneralCoupledTimeIntegrator2D, self).__init__(solver)


class NonHydrostaticTimeIntegrator2D(CoupledTimeIntegrator2D):
    """
    2D non-hydrostatic time integrator based on Shallow Water time integrator

    This time integration method uses SWE time integrator to advance
    the hydrostatic equations, depth-integrated Poisson solver to be
    solved for NH pressure, and a free surface integrator to advance
    the free surface correction. Advancing in serial or in a whole
    time stepping depends on the selection of time integrators.
    """
    def __init__(self, solver, swe_integrator, fs_integrator):
        self.swe_integrator = swe_integrator
        super().__init__(solver)
        self.poisson_solver = solver.poisson_solver
        print_output('  Non-hydrostatic pressure solver: {:}'.format(self.poisson_solver.__class__.__name__))
        print_output('  Free Surface time integrator: {:}'.format(fs_integrator.__name__))
        self.nh_options = solver.options.nh_model_options
        if self.nh_options.update_free_surface:
            self.timesteppers.fs2d = self.solver.get_fs_timestepper(fs_integrator)
            self.elev_old = Function(self.fields.elev_2d)
        self.serial_advancing = not hasattr(self.timesteppers.swe2d, 'n_stages') \
            or self.options.swe_timestepper_type == 'SSPIMEX'
        self.multi_stages_fs = hasattr(self.timesteppers.fs2d, 'n_stages') \
            and self.nh_options.free_surface_timestepper_type != 'BackwardEuler'
        if self.multi_stages_fs:
            msg = 'The multi-stage type of Shallow Water and ' \
                  'Free Surface time integrators should be the same.'
            assert self.options.swe_timestepper_type == self.nh_options.free_surface_timestepper_type, msg

    @PETSc.Log.EventDecorator("thetis.NonHydrostaticTimeIntegrator2D.initialize")
    def initialize(self, solution2d):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        assert solution2d == self.fields.solution_2d

        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.nh_options.update_free_surface:
            self.timesteppers.fs2d.initialize(self.fields.elev_2d)
            self.elev_old.assign(self.fields.elev_2d)

    @PETSc.Log.EventDecorator("thetis.NonHydrostaticTimeIntegrator2D.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if self.serial_advancing:
            # --- advance in serial ---
            self.timesteppers.swe2d.advance(t, update_forcings=update_forcings)
            # solve non-hydrostatic pressure q and update velocities
            self.poisson_solver.solve()
            # update free surface elevation
            if self.nh_options.update_free_surface:
                self.fields.elev_2d.assign(self.elev_old)
                self.timesteppers.fs2d.advance(t, update_forcings=update_forcings)
                self.elev_old.assign(self.fields.elev_2d)
            # update old solution
            if self.options.swe_timestepper_type == 'SSPIMEX':
                self.timesteppers.swe2d.erk.solution_old.assign(self.fields.solution_2d)
                self.timesteppers.swe2d.dirk.solution_old.assign(self.fields.solution_2d)
        else:
            # --- advance in a whole stepping ---
            for i in range(self.timesteppers.swe2d.n_stages):
                last_stage = i == self.timesteppers.swe2d.n_stages - 1
                self.timesteppers.swe2d.solve_stage(i, t, update_forcings)
                # solve non-hydrostatic pressure q and update velocities
                self.poisson_solver.solve(solve_w=last_stage)
                # update free surface elevation
                if self.nh_options.update_free_surface:
                    if self.multi_stages_fs:
                        self.fields.elev_2d.assign(self.elev_old)
                        self.timesteppers.fs2d.solve_stage(i, t, update_forcings)
                        self.elev_old.assign(self.fields.elev_2d)
                    elif last_stage:
                        self.fields.elev_2d.assign(self.elev_old)
                        self.timesteppers.fs2d.advance(t, update_forcings=update_forcings)
                        self.elev_old.assign(self.fields.elev_2d)
