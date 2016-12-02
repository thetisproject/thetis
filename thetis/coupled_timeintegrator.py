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


class CoupledTimeIntegratorBase(timeintegrator.TimeIntegratorBase):
    """
    Base class for coupled 2D-3D time integrators

    Provides common functionality for updating diagnostic fields etc.
    """
    def __init__(self, solver, options, fields):
        """
        :param solver: :class:`.FlowSolver` object
        :param options: :class:`.ModelOptions` object
        :param fields: :class:`.FieldDict` object
        """
        # TODO remove option, field args as these are members of solver
        self.solver = solver
        self.options = options
        self.fields = fields

    def _update_3d_elevation(self):
        """Projects elevation to 3D"""
        with timed_stage('aux_elev_3d'):
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}

    def _update_vertical_velocity(self):
        """Solve vertical velocity"""
        with timed_stage('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates 3D mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
                self.solver.mesh_updater.update_mesh_coordinates()

    def _update_bottom_friction(self):
        """Computes bottom friction related fields"""
        if self.options.use_bottom_friction:
            with timed_stage('aux_friction'):
                self.solver.uv_p1_projector.project()
                compute_bottom_friction(
                    self.solver,
                    self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                    self.fields.z_bottom_2d, self.fields.bathymetry_2d,
                    self.fields.bottom_drag_2d)
        if self.options.use_parabolic_viscosity:
            self.solver.parabolic_viscosity_solver.solve()

    def _update_2d_coupling(self):
        """Does 2D-3D coupling for the velocity field"""
        with timed_stage('aux_uv_coupling'):
            self._remove_depth_average_from_uv_3d()
            self._update_2d_coupling_term()
            self._copy_uv_2d_to_3d()

    def _remove_depth_average_from_uv_3d(self):
        """Computes depth averaged velocity and removes it from the 3D velocity field"""
        with timed_stage('aux_uv_coupling'):
            # compute depth averaged 3D velocity
            self.solver.uv_averager.solve()  # uv -> uv_dav_3d
            self.solver.extract_surf_dav_uv.solve()  # uv_dav_3d -> uv_dav_2d
            self.solver.copy_uv_dav_to_uv_dav_3d.solve()  # uv_dav_2d -> uv_dav_3d
            # remove depth average from 3D velocity
            self.fields.uv_3d -= self.fields.uv_dav_3d

    def _copy_uv_2d_to_3d(self):
        """Copies uv_2d to uv_dav_3d"""
        with timed_stage('aux_uv_coupling'):
            self.solver.copy_uv_to_uv_dav_3d.solve()

    def _update_2d_coupling_term(self):
        """Update split_residual_2d field for 2D-3D coupling"""
        with timed_stage('aux_uv_coupling'):
            # scale dav uv 2D to be used as a forcing in 2D mom eq.
            self.fields.split_residual_2d.assign(self.fields.uv_dav_2d)
            self.fields.split_residual_2d /= self.timestepper_mom_3d.dt_const

    def _update_baroclinicity(self):
        """Computes baroclinic head"""
        if self.options.baroclinic:
            with timed_stage('aux_baroclin'):
                compute_baroclinic_head(self.solver)

    def _update_turbulence(self, t):
        """
        Updates turbulence related fields

        :param t: simulation time
        """
        if self.options.use_turbulence:
            with timed_stage('turbulence'):
                self.solver.gls_model.preprocess()
                # NOTE psi must be solved first as it depends on tke
                self.timestepper_psi_3d.advance(t)
                self.timestepper_tke_3d.advance(t)
                self.solver.gls_model.postprocess()

    def _update_stabilization_params(self):
        """
        Computes Smagorinsky viscosity etc fields
        """
        with timed_stage('aux_stability'):
            self.solver.uv_mag_solver.solve()
            # update P1 velocity field
            self.solver.uv_p1_projector.project()
            if self.options.smagorinsky_factor is not None:
                self.solver.smagorinsky_diff_solver.solve()
            if self.options.salt_jump_diff_factor is not None:
                self.solver.horiz_jump_diff_solver.solve()

    def _update_all_dependencies(self, t,
                                 do_2d_coupling=False,
                                 do_vert_diffusion=False,
                                 do_ale_update=False,
                                 do_stab_params=False,
                                 do_turbulence=False):
        """Default routine for updating all dependent fields after a time step"""
        self._update_3d_elevation()
        if do_ale_update:
            self._update_moving_mesh()
        if do_2d_coupling:
            self._update_2d_coupling()
        self._update_vertical_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()
        if do_turbulence:
            self._update_turbulence(t)
        if do_vert_diffusion and self.options.solve_vert_diffusion:
            with timed_stage('impl_mom_vvisc'):
                self.timestepper_mom_vdff_3d.advance(t)
            if self.options.solve_salt:
                with timed_stage('impl_salt_vdiff'):
                    self.timestepper_salt_vdff_3d.advance(t)
            if self.options.solve_temp:
                with timed_stage('impl_temp_vdiff'):
                    self.timestepper_temp_vdff_3d.advance(t)
        if do_stab_params:
            self._update_stabilization_params()


class CoupledTimeIntegrator(CoupledTimeIntegratorBase):
    """
    Base class of mode-split time integrators that use 2D, 3D and implicit 3D
    time integrators.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def integrator_2d(self):
        """time integrator for 2D equations"""
        pass

    @abstractproperty
    def integrator_3d(self):
        """time integrator for explicit 3D equations (momentum, tracers)"""
        pass

    @abstractproperty
    def integrator_vert_3d(self):
        """time integrator for implicit 3D equations (vertical diffusion)"""
        pass

    def __init__(self, solver):
        """
        :param solver: :class:`.FlowSolver` object
        """
        super(CoupledTimeIntegrator, self).__init__(solver,
                                                    solver.options,
                                                    solver.fields)
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        print_output('  2D time integrator: {:}'.format(self.integrator_2d.__name__))
        print_output('  3D time integrator: {:}'.format(self.integrator_3d.__name__))
        print_output('  3D implicit time integrator: {:}'.format(self.integrator_vert_3d.__name__))
        self._initialized = False

        self._create_integrators()
        self.n_stages = self.timestepper2d.n_stages

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        # TODO refactor
        solver = self.solver
        uv_source_2d = solver.fields.split_residual_2d
        if self.options.uv_source_2d is not None:
            uv_source_2d = solver.fields.split_residual_2d + self.options.uv_source_2d
        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'baroc_head': solver.fields.get('baroc_head_2d'),
            'baroc_head_bot': solver.fields.get('baroc_head_bot_2d'),
            'viscosity_h': self.options.get('h_viscosity'),  # FIXME should be total h visc
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': uv_source_2d,
            'elev_source': self.options.elev_source_2d,
            'linear_drag': self.options.linear_drag}

        if issubclass(self.integrator_2d, (rungekutta.ERKSemiImplicitGeneric)):
            self.timestepper2d = self.integrator_2d(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.solver_parameters_sw,
                semi_implicit=self.options.use_linearized_semi_implicit_2d,
                theta=self.options.shallow_water_theta)
        else:
            self.timestepper2d = self.integrator_2d(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.solver_parameters_sw)

        # assign viscosity/diffusivity to correct equations
        if self.options.solve_vert_diffusion:
            implicit_v_visc = solver.tot_v_visc.get_sum()
            explicit_v_visc = None
            implicit_v_diff = solver.tot_v_diff.get_sum()
            explicit_v_diff = None
        else:
            implicit_v_visc = None
            explicit_v_visc = solver.tot_v_visc.get_sum()
            implicit_v_diff = None
            explicit_v_diff = solver.tot_v_diff.get_sum()

        fields = {'eta': self.fields.elev_3d,  # FIXME rename elev
                  'baroc_head': self.fields.get('baroc_head_3d'),
                  'mean_baroc_head': self.fields.get('baroc_head_int_3d'),
                  'uv_depth_av': self.fields.get('uv_dav_3d'),
                  'w': self.fields.w_3d,
                  'w_mesh': self.fields.get('w_mesh_3d'),
                  'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                  'viscosity_v': explicit_v_visc,
                  'viscosity_h': self.solver.tot_h_visc.get_sum(),
                  'source': self.options.uv_source_3d,
                  # uv_mag': self.fields.uv_mag_3d,
                  'uv_p1': self.fields.get('uv_p1_3d'),
                  'lax_friedrichs_factor': self.options.uv_lax_friedrichs,
                  'coriolis': self.fields.get('coriolis_3d'),
                  'linear_drag': self.options.linear_drag,
                  'quadratic_drag': self.options.quadratic_drag,
                  }
        self.timestepper_mom_3d = self.integrator_3d(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.solver_parameters_momentum_explicit)
        if self.solver.options.solve_vert_diffusion:
            fields = {'viscosity_v': implicit_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      }
            self.timestepper_mom_vdff_3d = self.integrator_vert_3d(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.solver_parameters_momentum_implicit)

        if self.solver.options.solve_salt:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': explicit_v_diff,
                      'source': self.options.salt_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                      }
            self.timestepper_salt_3d = self.integrator_3d(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_salt_vdff_3d = self.integrator_vert_3d(
                    solver.eq_salt_vdff, solver.fields.salt_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['salt'],
                    solver_parameters=self.options.solver_parameters_tracer_implicit)

        if self.solver.options.solve_temp:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': explicit_v_diff,
                      'source': self.options.temp_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                      }
            self.timestepper_temp_3d = self.integrator_3d(
                solver.eq_temp, solver.fields.temp_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['temp'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_temp_vdff_3d = self.integrator_vert_3d(
                    solver.eq_temp_vdff, solver.fields.temp_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['temp'],
                    solver_parameters=self.options.solver_parameters_tracer_implicit)

        if self.solver.options.use_turbulence:
            fields = {'diffusivity_v': implicit_v_diff,
                      'viscosity_v': implicit_v_visc,
                      'k': solver.fields.tke_3d,
                      'epsilon': solver.gls_model.epsilon,
                      'shear_freq2': solver.gls_model.m2,
                      'buoy_freq2_neg': solver.gls_model.n2_neg,
                      'buoy_freq2_pos': solver.gls_model.n2_pos
                      }
            self.timestepper_tke_3d = self.integrator_vert_3d(
                solver.eq_tke_diff, solver.fields.tke_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            self.timestepper_psi_3d = self.integrator_vert_3d(
                solver.eq_psi_diff, solver.fields.psi_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            if self.solver.options.use_turbulence_advection:
                fields = {'elev_3d': self.fields.elev_3d,
                          'uv_3d': self.fields.uv_3d,
                          'uv_depth_av': self.fields.get('uv_dav_3d'),
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                          # uv_mag': self.fields.uv_mag_3d,
                          'uv_p1': self.fields.get('uv_p1_3d'),
                          'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                          }
                self.timestepper_tke_adv_eq = self.integrator_3d(
                    solver.eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
                self.timestepper_psi_adv_eq = self.integrator_3d(
                    solver.eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
        self.cfl_coeff_3d = self.timestepper_mom_3d.cfl_coeff
        self.cfl_coeff_2d = self.timestepper2d.cfl_coeff

    def set_dt(self, dt, dt_2d):
        """
        Set time step for the coupled time integrator

        :param dt: Time step. This is the master (macro) time step used to
            march the 3D equations.
        :type dt: float
        :param dt_2d: Time step for 2D equations. For constency :attr:`dt_2d`
            must be an integer fraction of :attr:`dt`. If 2D solver is implicit
            use set :attr:`dt_2d` equal to :attr:`dt`.
        :type dt_2d: float

        """
        # TODO check mod(dt, dt_2d) == 0
        if dt != dt_2d:
            raise NotImplementedError('Case dt_2d < dt is not implemented yet')
        self.timestepper2d.set_dt(dt)
        self.timestepper_mom_3d.set_dt(dt)
        if self.solver.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.set_dt(dt)
        if self.solver.options.solve_salt:
            self.timestepper_salt_3d.set_dt(dt)
            if self.solver.options.solve_vert_diffusion:
                self.timestepper_salt_vdff_3d.set_dt(dt)
        if self.solver.options.solve_temp:
            self.timestepper_temp_3d.set_dt(dt)
            if self.solver.options.solve_vert_diffusion:
                self.timestepper_temp_vdff_3d.set_dt(dt)
        if self.solver.options.use_turbulence:
            self.timestepper_tke_3d.set_dt(dt)
            self.timestepper_psi_3d.set_dt(dt)
            if self.solver.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.set_dt(dt)
                self.timestepper_psi_adv_eq.set_dt(dt)

    def initialize(self):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
            if self.options.solve_vert_diffusion:
                self.timestepper_salt_vdff_3d.initialize(self.fields.salt_3d)
        if self.options.solve_temp:
            self.timestepper_temp_3d.initialize(self.fields.temp_3d)
            if self.options.solve_vert_diffusion:
                self.timestepper_temp_vdff_3d.initialize(self.fields.temp_3d)
        if self.solver.options.use_turbulence:
            self.timestepper_tke_3d.initialize(self.fields.tke_3d)
            self.timestepper_psi_3d.initialize(self.fields.psi_3d)
            if self.solver.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.initialize(self.fields.tke_3d)
                self.timestepper_psi_adv_eq.initialize(self.fields.psi_3d)

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

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    # explicit advection
                    self.timestepper_tke_adv_eq.solve_stage(k, t)
                    self.timestepper_psi_adv_eq.solve_stage(k, t)
                    # if self.options.use_limiter_for_tracers:
                    #    self.solver.tracer_limiter.apply(self.solver.fields.tke_3d)
                    #    self.solver.tracer_limiter.apply(self.solver.fields.psi_3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, update_forcings)
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
    surface gravity wave speed.
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
        ti = self.timestepper2d
        assert not ti.is_implicit
        a = ti.butcher[1:, :]
        self.a_inv = linalg.inv(a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :param i_stage: state of the Runge-Kutta iteration
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

        :param i_stage: state of the Runge-Kutta iteration
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

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :param t: simulation time
        :type t: float
        :param dt: time step
        :type dt: float
        :param update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :param update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        # TODO remove dt from args to comply with timeintegrator API
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            # FIXME mesh velocity is too high ~2x with EKRLPUM2
            last_step = (k == self.n_stages - 1)
            self._compute_mesh_velocity_pre(k)
            with timed_stage('mode2d'):
                self.timestepper2d.update_solution(k)
                self.timestepper2d.solve_tendency(k, t, update_forcings)
                if last_step:
                    self.timestepper2d.get_final_solution()
            self.compute_mesh_velocity(k)

            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_tendency(k, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.solve_tendency(k, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.solve_tendency(k, t, update_forcings3d)
                    self.timestepper_psi_adv_eq.solve_tendency(k, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_tendency(k, t, update_forcings3d)

            self._update_moving_mesh()

            if last_step:
                with timed_stage('salt_eq'):
                    if self.options.solve_salt:
                        self.timestepper_salt_3d.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temp:
                        self.timestepper_temp_3d.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if self.options.use_turbulence_advection:
                        self.timestepper_tke_adv_eq.get_final_solution()
                        self.timestepper_psi_adv_eq.get_final_solution()
                with timed_stage('momentum_eq'):
                    self.timestepper_mom_3d.get_final_solution()
            else:
                with timed_stage('salt_eq'):
                    if self.options.solve_salt:
                        self.timestepper_salt_3d.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temp:
                        self.timestepper_temp_3d.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if self.options.use_turbulence_advection:
                        self.timestepper_tke_adv_eq.update_solution(k)
                        self.timestepper_psi_adv_eq.update_solution(k)
                with timed_stage('momentum_eq'):
                    self.timestepper_mom_3d.update_solution(k)

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
        ti = self.timestepper2d.dirk
        assert ti.is_implicit
        self.a_inv = linalg.inv(ti.a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :param i_stage: state of the Runge-Kutta iteration
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

        :param i_stage: state of the Runge-Kutta iteration
        """
        # TODO remove dt from args to comply with timeintegrator API
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

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
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
                self.timestepper2d.erk.update_solution(k)
                self.timestepper_mom_3d.update_solution(k)
                if self.options.solve_salt:
                    self.timestepper_salt_3d.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temp:
                    self.timestepper_temp_3d.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                # TODO need to update all dependencies here
                self._update_3d_elevation()
                self._update_vertical_velocity()

            self._compute_mesh_velocity_pre(k)
            # - IM: solve implicit tendency (this is implicit solve)
            self.timestepper2d.dirk.solve_tendency(k, t, update_forcings3d)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            self.timestepper2d.dirk.update_solution(k)
            self.compute_mesh_velocity(k)
            # TODO update all dependencies of implicit solutions here
            # NOTE if 3D implicit solves, must be done in new mesh!
            self._update_3d_elevation()

            # - EX: evaluate explicit tendency
            self.timestepper_mom_3d.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_salt:
                self.timestepper_salt_3d.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_temp:
                self.timestepper_temp_3d.solve_tendency(k, t, update_forcings3d)

            last_step = (k == self.n_stages - 1)
            if last_step:
                self.timestepper2d.get_final_solution()
                self._update_3d_elevation()
            self._update_moving_mesh()
            if last_step:
                self.timestepper_mom_3d.get_final_solution()
                self._update_vertical_velocity()
                if self.options.solve_salt:
                    self.timestepper_salt_3d.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temp:
                    self.timestepper_temp_3d.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.get_final_solution()
                    self.timestepper_psi_adv_eq.get_final_solution()

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

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
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
            if self.options.solve_salt:
                self.timestepper_salt_3d.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temp:
                self.timestepper_temp_3d.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if self.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.predict()
                self.timestepper_psi_adv_eq.predict()
        with timed_stage('momentum_eq'):
            self.timestepper_mom_3d.predict()

        # dependencies for 2D update
        self._update_2d_coupling()
        self._update_baroclinicity()
        self._update_bottom_friction()

        # update 2D
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
        self.uv_old_2d.assign(self.fields.uv_2d)
        with timed_stage('mode2d'):
            self.timestepper2d.advance(t, update_forcings)
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_finalize()
        self.uv_new_2d.assign(self.fields.uv_2d)

        # set 3D elevation to half step
        gamma = self.timestepper_mom_3d.gamma
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
            if self.options.solve_salt:
                self.timestepper_salt_3d.eval_rhs()
        with timed_stage('temp_eq'):
            if self.options.solve_temp:
                self.timestepper_temp_3d.eval_rhs()
        with timed_stage('turb_advection'):
            if self.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.eval_rhs()
                self.timestepper_psi_adv_eq.eval_rhs()
        with timed_stage('momentum_eq'):
            self.timestepper_mom_3d.eval_rhs()

        self._update_3d_elevation()
        self._update_moving_mesh()

        with timed_stage('salt_eq'):
            if self.options.solve_salt:
                self.timestepper_salt_3d.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temp:
                self.timestepper_temp_3d.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if self.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.correct()
                self.timestepper_psi_adv_eq.correct()
        with timed_stage('momentum_eq'):
            self.timestepper_mom_3d.correct()

        if self.options.solve_vert_diffusion:
            # TODO figure out minimal set of dependency updates (costly)
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            self._update_turbulence(t)
            if self.options.solve_salt:
                with timed_stage('impl_salt_vdiff'):
                    self.timestepper_salt_vdff_3d.advance(t)
            if self.options.solve_temp:
                with timed_stage('impl_temp_vdiff'):
                    self.timestepper_temp_vdff_3d.advance(t)
            with timed_stage('impl_mom_vvisc'):
                self.timestepper_mom_vdff_3d.advance(t)
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
    integrator_2d = rungekutta.TwoStageTrapezoid
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

        :param istage: stage of the Runge-Kutta iteration
        :type istage: int
        """
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
            self.elev_fields[istage].assign(self.fields.elev_cg_2d)

    def compute_mesh_velocity(self, istage):
        """
        Computes mesh velocity for stage i

        Must be called after updating the 2D mode.

        :param istage: stage of the Runge-Kutta iteration
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

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        for i_stage in range(self.n_stages):

            # solve 2D mode
            self.store_elevation(i_stage)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(i_stage, t, update_forcings)
            self.compute_mesh_velocity(i_stage)

            # solve 3D mode: preprocess in old mesh
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.prepare_stage(i_stage, t, update_forcings3d)
                    self.timestepper_psi_adv_eq.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.prepare_stage(i_stage, t, update_forcings3d)

            # update mesh
            self._update_3d_elevation()
            self._update_moving_mesh()

            # solve 3D mode
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.solve_stage(i_stage)
                    self.timestepper_psi_adv_eq.solve_stage(i_stage)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(i_stage)

            # update coupling terms
            last = i_stage == self.n_stages - 1

            if last:
                self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            if i_stage == last and self.options.solve_vert_diffusion:
                self._update_turbulence(t)
                if self.options.solve_salt:
                    with timed_stage('impl_salt_vdiff'):
                        self.timestepper_salt_vdff_3d.advance(t)
                if self.options.solve_temp:
                    with timed_stage('impl_temp_vdiff'):
                        self.timestepper_temp_vdff_3d.advance(t)
                with timed_stage('impl_mom_vvisc'):
                    self.timestepper_mom_vdff_3d.advance(t)
                self._update_baroclinicity()
            self._update_vertical_velocity()
            if i_stage == last:
                self._update_stabilization_params()
