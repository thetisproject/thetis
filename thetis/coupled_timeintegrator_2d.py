"""
Time integrators for solving coupled 2D-3D system of equations.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from . import rungekutta
from . import implicitexplicit
from abc import ABCMeta, abstractproperty


class CoupledTimeIntegrator(timeintegrator.TimeIntegratorBase):
    """
    Base class of mode-split time integrators that use 2D, 3D and implicit 3D
    time integrators.
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
        print_output('  2D time integrator: {:}'.format(self.integrator_2d.__name__))
        self._initialized = False

        self._create_integrators()
        self.n_stages = self.timesteppers.swe2d.n_stages

    def _create_swe_integrator(self):
        """
        Create time integrator for 2D system
        """
        solver = self.solver
        momentum_source_2d = solver.fields.split_residual_2d
        if self.options.momentum_source_2d is not None:
            momentum_source_2d = solver.fields.split_residual_2d + self.options.momentum_source_2d
        fields = {
            'coriolis': self.options.coriolis_frequency,
            'momentum_source': momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
            'atmospheric_pressure': self.options.atmospheric_pressure,
        }

        if issubclass(self.swe_integrator, (rungekutta.ERKSemiImplicitGeneric)):
            self.timesteppers.swe2d = self.swe_integrator(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe,
                semi_implicit=True,
                theta=self.options.timestepper_options.implicitness_theta_2d)
        else:
            self.timesteppers.swe2d = self.swe_integrator(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe)

    def _create_tracer_integrator(self):
        """
        Create time integrator for salinity equation
        """
        solver = self.solver

        if self.solver.options.solve_tracer:
            fields = {'elev_2d': self.fields.elev_2d,
                      'uv_2d': self.fields.uv_2d,
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'source': self.options.tracer_source_2d,
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.tracer = self.tracer_integrator(
                solver.eq_tracer, solver.fields.tracer_2d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['tracer'],
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer)

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

    def initialize(self):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.options.solve_tracer:
            self.timesteppers.tracer.initialize(self.fields.tracer_2d)
        self._initialized = True


class CoupledSSPRKSemiImplicit(CoupledTimeIntegrator):
    """
    Solves coupled equations with SSPRK33 time integrator using the same time
    step for the 2D and 3D modes.

    In the 2D mode the surface gravity waves are solved semi-implicitly. This
    allows longer time steps but diffuses free surface waves.

    This time integrator uses a static 3D mesh. It is not compliant with the
    ALE moving mesh.
    """
    integrator_2d = rungekutta.SSPRK33SemiImplicit
    integrator_3d = rungekutta.SSPRK33
    integrator_vert_3d = rungekutta.BackwardEuler

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_stage(k, t)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_stage(k, t)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_stage(k, t)
                if self.options.use_limiter_for_velocity:
                    self.solver.uv_limiter.apply(self.fields.uv_3d)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.solve_stage(k, t, update_forcings)
            last_step = (k == 2)
            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=last_step,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class CoupledERKALE(CoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation

    A fully explicit mode-split time integrator where both the 2D and 3D modes
    use the same time step. The time step is typically chosen to match the 2D
    surface gravity wave speed. Only vertical diffusion is treated implicitly.
    """
    integrator_2d = rungekutta.ERKLPUM2
    integrator_3d = rungekutta.ERKLPUM2ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledERKALE, self).__init__(solver)

        self.elev_cg_old_2d = []
        for i in range(self.n_stages + 1):
            f = Function(self.solver.fields.elev_cg_2d)
            self.elev_cg_old_2d.append(f)

        import numpy.linalg as linalg
        ti = self.timesteppers.swe2d
        assert not ti.is_implicit
        a = ti.butcher[1:, :]
        self.a_inv = linalg.inv(a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :arg i_stage: state of the Runge-Kutta iteration
        """
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """
        Compute mesh velocity from 2D solver runge-kutta scheme

        Mesh velocity is solved from the Runge-Kutta coefficients of the
        implicit 2D solver.

        :arg i_stage: state of the Runge-Kutta iteration
        """
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velocity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref/h + 1.0)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            # FIXME mesh velocity is too high ~2x with EKRLPUM2
            last_step = (k == self.n_stages - 1)
            self._compute_mesh_velocity_pre(k)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.update_solution(k)
                self.timesteppers.swe2d.solve_tendency(k, t, update_forcings)
                if last_step:
                    self.timesteppers.swe2d.get_final_solution()
            self.compute_mesh_velocity(k)

            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_tendency(k, t, update_forcings3d)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_tendency(k, t, update_forcings3d)

            self._update_moving_mesh()

            if last_step:
                with timed_stage('salt_eq'):
                    if self.options.solve_salinity:
                        self.timesteppers.salt_expl.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temperature:
                        self.timesteppers.temp_expl.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if 'psi_expl' in self.timesteppers:
                        self.timesteppers.psi_expl.get_final_solution()
                    if 'tke_expl' in self.timesteppers:
                        self.timesteppers.tke_expl.get_final_solution()
                with timed_stage('momentum_eq'):
                    self.timesteppers.mom_expl.get_final_solution()
                    if self.options.use_limiter_for_velocity:
                        self.solver.uv_limiter.apply(self.fields.uv_3d)
            else:
                with timed_stage('salt_eq'):
                    if self.options.solve_salinity:
                        self.timesteppers.salt_expl.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temperature:
                        self.timesteppers.temp_expl.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if 'psi_expl' in self.timesteppers:
                        self.timesteppers.psi_expl.update_solution(k)
                    if 'tke_expl' in self.timesteppers:
                        self.timesteppers.tke_expl.update_solution(k)
                with timed_stage('momentum_eq'):
                    self.timesteppers.mom_expl.update_solution(k)
                    if self.options.use_limiter_for_velocity:
                        self.solver.uv_limiter.apply(self.fields.uv_3d)

            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=True,
                                          do_ale_update=False,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class CoupledIMEXALE(CoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation

    Advances the 2D-3D system with IMEX scheme: the free surface gravity waves
    are solved with the implicit scheme while all other terms are solved with the
    explicit scheme. Vertical diffusion is however solved with a separate
    implicit scheme (backward Euler) for efficiency.
    """
    integrator_2d = implicitexplicit.IMEXLPUM2
    integrator_3d = rungekutta.ERKLPUM2ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledIMEXALE, self).__init__(solver)

        self.elev_cg_old_2d = []
        for i in range(self.n_stages + 1):
            f = Function(self.solver.fields.elev_cg_2d)
            self.elev_cg_old_2d.append(f)

        import numpy.linalg as linalg
        ti = self.timesteppers.swe2d.dirk
        assert ti.is_implicit
        self.a_inv = linalg.inv(ti.a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :arg i_stage: state of the Runge-Kutta iteration
        """
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """
        Compute mesh velocity from 2D solver runge-kutta scheme

        Mesh velocity is solved from the Runge-Kutta coefficients of the
        implicit 2D solver.

        :arg i_stage: state of the Runge-Kutta iteration
        """
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velocity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref/h + 1.0)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            # IMEX update
            # - EX: set solution to u_n + dt*sum(a*k_erk)
            # - IM: solve implicit tendency (this is implicit solve)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            # - EX: evaluate explicit tendency

            # - EX: set solution to u_n + dt*sum(a*k_erk)
            if k > 0:
                self.timesteppers.swe2d.erk.update_solution(k)
                self.timesteppers.mom_expl.update_solution(k)
                if self.options.use_limiter_for_velocity:
                    self.solver.uv_limiter.apply(self.fields.uv_3d)
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                # TODO need to update all dependencies here
                self._update_3d_elevation()
                self._update_vertical_velocity()

            self._compute_mesh_velocity_pre(k)
            # - IM: solve implicit tendency (this is implicit solve)
            self.timesteppers.swe2d.dirk.solve_tendency(k, t, update_forcings3d)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            self.timesteppers.swe2d.dirk.update_solution(k)
            self.compute_mesh_velocity(k)
            # TODO update all dependencies of implicit solutions here
            # NOTE if 3D implicit solves, must be done in new mesh!
            self._update_3d_elevation()

            # - EX: evaluate explicit tendency
            self.timesteppers.mom_expl.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.solve_tendency(k, t, update_forcings3d)

            last_step = (k == self.n_stages - 1)
            if last_step:
                self.timesteppers.swe2d.get_final_solution()
                self._update_3d_elevation()
            self._update_moving_mesh()
            if last_step:
                self.timesteppers.mom_expl.get_final_solution()
                if self.options.use_limiter_for_velocity:
                    self.solver.uv_limiter.apply(self.fields.uv_3d)
                self._update_vertical_velocity()
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.get_final_solution()
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.get_final_solution()

            self._update_2d_coupling()
            self._update_vertical_velocity()


class CoupledLeapFrogAM3(CoupledTimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 time integrator for coupled 2D-3D problem

    This is an ALE time integrator.
    Implementation follows the SLIM time integrator by Karna et al (2013)

    Karna, et al. (2013). A baroclinic discontinuous Galerkin finite element
    model for coastal flows. Ocean Modelling, 61(0):1-20.
    http://dx.doi.org/10.1016/j.ocemod.2012.09.009
    """
    integrator_2d = rungekutta.DIRK22
    integrator_3d = timeintegrator.LeapFrogAM3
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledLeapFrogAM3, self).__init__(solver)
        self.elev_old_3d = Function(self.fields.elev_3d)
        self.uv_old_2d = Function(self.fields.uv_2d)
        self.uv_new_2d = Function(self.fields.uv_2d)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        # -------------------------------------------------
        # Prediction step
        # - from t_{n-1/2} to t_{n+1/2}
        # - RHS evaluated at t_{n}
        # - Forward Euler step in fixed domain Omega_n
        # -------------------------------------------------

        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d.assign(0.0)

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.predict()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.predict()

        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.predict()
            if self.options.use_limiter_for_velocity:
                self.solver.uv_limiter.apply(self.fields.uv_3d)

        # dependencies for 2D update
        self._update_2d_coupling()
        self._update_baroclinicity()
        self._update_bottom_friction()

        # update 2D
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
        self.uv_old_2d.assign(self.fields.uv_2d)
        with timed_stage('mode2d'):
            self.timesteppers.swe2d.advance(t, update_forcings)
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_finalize()
        self.uv_new_2d.assign(self.fields.uv_2d)

        # set 3D elevation to half step
        gamma = self.timesteppers.mom_expl.gamma
        self.elev_old_3d.assign(self.fields.elev_3d)
        self.solver.copy_elev_to_3d.solve()
        self.fields.elev_3d *= (0.5 + 2*gamma)
        self.fields.elev_3d += (0.5 - 2*gamma)*self.elev_old_3d

        # correct uv_3d to uv_2d at t_{n+1/2}
        self.fields.uv_2d *= (0.5 + 2*gamma)
        self.fields.uv_2d += (0.5 - 2*gamma)*self.uv_old_2d
        self._update_2d_coupling()
        self.fields.uv_2d.assign(self.uv_new_2d)  # restore
        self._update_vertical_velocity()
        self._update_baroclinicity()

        # -------------------------------------------------
        # Correction step
        # - from t_{n} to t_{n+1}
        # - RHS evaluated at t_{n+1/2}
        # - Forward Euler ALE step from Omega_n to Omega_{n+1}
        # -------------------------------------------------

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.eval_rhs()
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.eval_rhs()
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.eval_rhs()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.eval_rhs()
        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.eval_rhs()

        self._update_3d_elevation()
        self._update_moving_mesh()

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.correct()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.correct()
        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.correct()
            if self.options.use_limiter_for_velocity:
                self.solver.uv_limiter.apply(self.fields.uv_3d)

        if self.options.use_implicit_vertical_diffusion:
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            self._update_turbulence(t)
            if self.options.solve_salinity:
                with timed_stage('impl_salt_vdiff'):
                    self.timesteppers.salt_impl.advance(t)
            if self.options.solve_temperature:
                with timed_stage('impl_temp_vdiff'):
                    self.timesteppers.temp_impl.advance(t)
            with timed_stage('impl_mom_vvisc'):
                self.timesteppers.mom_impl.advance(t)
            self._update_baroclinicity()
            self._update_vertical_velocity()
            self._update_stabilization_params()
        else:
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            self._update_vertical_velocity()
            self._update_stabilization_params()


class CoupledTwoStageRK(CoupledTimeIntegrator):
    """
    Coupled time integrator based on SSPRK(2,2) scheme

    This ALE time integration method uses SSPRK(2,2) scheme to advance the 3D
    equations and a compatible implicit Trapezoid method to advance the 2D
    equations. Backward Euler scheme is used for vertical diffusion.
    """
    integrator_2d = rungekutta.ESDIRKTrapezoid
    integrator_3d = timeintegrator.SSPRK22ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledTwoStageRK, self).__init__(solver)
        # allocate CG elevation fields for storing mesh geometry
        self.elev_fields = []
        for i in range(self.n_stages):
            e = Function(self.fields.elev_cg_2d)
            self.elev_fields.append(e)

    def store_elevation(self, istage):
        """
        Store current elevation field for computing mesh velocity

        Must be called before updating the 2D mode.

        :arg istage: stage of the Runge-Kutta iteration
        :type istage: int
        """
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
            self.elev_fields[istage].assign(self.fields.elev_cg_2d)

    def compute_mesh_velocity(self, istage):
        """
        Computes mesh velocity for stage i

        Must be called after updating the 2D mode.

        :arg istage: stage of the Runge-Kutta iteration
        :type istage: int
        """
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
            current_elev = self.fields.elev_cg_2d
            if istage == 0:
                # compute w_mesh at surface as (elev^{(1)} - elev^{n})/dt
                w_s = (current_elev - self.elev_fields[0])/self.solver.dt
            else:
                # compute w_mesh at surface as (2*elev^{n+1} - elev^{(1)} - elev^{n})/dt
                w_s = (2*current_elev - self.elev_fields[1] - self.elev_fields[0])/self.solver.dt
            self.fields.w_mesh_surf_2d.assign(w_s)
            # use that to compute w_mesh in whole domain
            self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
            # solve w_mesh at nodes
            w_mesh_surf = self.fields.w_mesh_surf_3d.dat.data[:]
            z_ref = self.fields.z_coord_ref_3d.dat.data[:]
            h = self.fields.bathymetry_3d.dat.data[:]
            self.fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for i_stage in range(self.n_stages):

            # solve 2D mode
            self.store_elevation(i_stage)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.solve_stage(i_stage, t, update_forcings)
            self.compute_mesh_velocity(i_stage)

            # solve 3D mode: preprocess in old mesh
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.prepare_stage(i_stage, t, update_forcings3d)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.prepare_stage(i_stage, t, update_forcings3d)

            # update mesh
            self._update_3d_elevation()
            self._update_moving_mesh()

            # solve 3D mode
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_stage(i_stage)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_stage(i_stage)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_stage(i_stage)
                if self.options.use_limiter_for_velocity:
                    self.solver.uv_limiter.apply(self.fields.uv_3d)

            last_stage = i_stage == self.n_stages - 1

            if last_stage:
                # compute final prognostic variables
                self._update_2d_coupling()  # due before impl. viscosity
                if self.options.use_implicit_vertical_diffusion:
                    if self.options.solve_salinity:
                        with timed_stage('impl_salt_vdiff'):
                            self.timesteppers.salt_impl.advance(t)
                    if self.options.solve_temperature:
                        with timed_stage('impl_temp_vdiff'):
                            self.timesteppers.temp_impl.advance(t)
                    with timed_stage('impl_mom_vvisc'):
                        self.timesteppers.mom_impl.advance(t)
                # compute final diagnostic fields
                self._update_baroclinicity()
                self._update_vertical_velocity()
                # update parametrizations
                self._update_turbulence(t)
                self._update_bottom_friction()
                self._update_stabilization_params()
            else:
                # update variables that explict solvers depend on
                self._update_2d_coupling()
                self._update_baroclinicity()
                self._update_vertical_velocity()
