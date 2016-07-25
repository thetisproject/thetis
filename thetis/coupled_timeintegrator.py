"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from . import rungekutta
from . import implicitexplicit
from abc import ABCMeta, abstractproperty

# TODO turbulence update. move out from _update_all_dependencies ?
# TODO update solver interface to solve_stage(k, t, update_forcings)
# TODO - solution is not needed, it's a member of TimeIntegrator
# TODO - dt is usually the same, add change_dt(new_dt) method instead
# TODO make an helper method for creating timeintegrators (it's too long to cp)


class CoupledTimeIntegrator(timeintegrator.TimeIntegrator):
    """Base class for coupled time integrators"""
    def __init__(self, solver, options, fields):
        self.solver = solver
        self.options = options
        self.fields = fields

    def _update_3d_elevation(self):
        """Projects elevation to 3D"""
        with timed_stage('aux_elev_3d'):
            # elev_old_3d = Function(self.fields.elev_3d)
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}
            # self.fields.elev_3d *= 0.5
            # self.fields.elev_3d += 0.5*elev_old_3d

    def _update_vertical_velocity(self):
        with timed_stage('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
                self.solver.elev_2d_to_cg_projector.project()
                self.solver.copy_elev_cg_to_3d.solve()
                self.solver.mesh_coord_updater.solve()
                compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
                self.solver.copy_v_elem_size_to_2d.solve()

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
            # compute depth averaged 3D velocity
            self.solver.uv_averager.solve()
            self.solver.extract_surf_dav_uv.solve()
            self.solver.copy_uv_dav_to_uv_dav_3d.solve()
            # remore depth average from 3D velocity
            self.fields.uv_3d -= self.fields.uv_dav_3d
            self.solver.copy_uv_to_uv_dav_3d.solve()
            # add depth averaged 2D velocity
            self.fields.uv_3d += self.fields.uv_dav_3d

    def _update_mesh_velocity(self):
        """Computes ALE mesh velocity"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
                self.solver.w_mesh_solver.solve()

    def _update_baroclinicity(self):
        """Computes baroclinic head"""
        if self.options.baroclinic:
            with timed_stage('aux_baroclin'):
                compute_baroclinic_head(self.solver)

    def _update_turbulence(self, t):
        """Updates turbulence related fields"""
        if self.options.use_turbulence:
            with timed_stage('turbulence'):
                self.solver.gls_model.preprocess()
                # NOTE psi must be solved first as it depends on tke
                self.timestepper_psi_3d.advance(t, self.solver.dt, self.solver.fields.psi_3d)
                self.timestepper_tke_3d.advance(t, self.solver.dt, self.solver.fields.tke_3d)
                self.solver.gls_model.postprocess()

    def _update_stabilization_params(self):
        """Computes Smagorinsky viscosity etc fields"""
        # update velocity magnitude
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
                                 do_turbulence=False,
                                 do_mesh_velocity=True):
        """Default routine for updating all dependent fields after a time step"""
        self._update_3d_elevation()
        if do_ale_update:
            self._update_moving_mesh()
        if do_2d_coupling:
            self._update_2d_coupling()
        self._update_vertical_velocity()
        if do_mesh_velocity:
            self._update_mesh_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()
        if do_turbulence:
            self._update_turbulence(t)
        if do_vert_diffusion and self.options.solve_vert_diffusion:
            with timed_stage('impl_mom_vvisc'):
                self.timestepper_mom_vdff_3d.advance(t, self.solver.dt, self.fields.uv_3d)
            if self.options.solve_salt:
                with timed_stage('impl_salt_vdiff'):
                    self.timestepper_salt_vdff_3d.advance(t, self.solver.dt, self.fields.salt_3d)
            if self.options.solve_temp:
                with timed_stage('impl_temp_vdiff'):
                    self.timestepper_temp_vdff_3d.advance(t, self.solver.dt, self.fields.temp_3d)
        if do_stab_params:
            self._update_stabilization_params()


class CoupledSSPRKSync(CoupledTimeIntegrator):
    """
    Split-explicit SSPRK time integrator that sub-iterates 2D mode.
    3D time step is computed based on horizontal velocity. 2D mode is sub-iterated and hence has
    very little numerical diffusion.
    """
    def __init__(self, solver):
        super(CoupledSSPRKSync, self).__init__(solver, solver.options,
                                               solver.fields)
        self._initialized = False

        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'baroc_head': solver.fields.get('baroc_head_2d'),
            'viscosity_h': self.options.get('h_viscosity'),  # FIXME should be total h visc
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d,
            'linear_drag': self.options.linear_drag}
        self.timestepper2d = rungekutta.SSPRK33(
            solver.eq_sw, self.fields.solution_2d,
            fields, solver.dt,
            bnd_conditions=solver.bnd_functions['shallow_water'],
            solver_parameters=self.options.solver_parameters_sw)

        fs = self.timestepper2d.solution_old.function_space()
        self.sol2d_n = Function(fs, name='sol2dtmp')

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
        vert_timeintegrator = rungekutta.BackwardEuler

        fields = {'eta': self.fields.elev_3d,  # FIXME rename elev
                  'baroc_head': self.fields.get('baroc_head_3d'),
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
                  }
        self.timestepper_mom_3d = rungekutta.SSPRK33Stage(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.solver_parameters_momentum_explicit)
        if self.solver.options.solve_vert_diffusion:
            fields = {'viscosity_v': implicit_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      }
            self.timestepper_mom_vdff_3d = vert_timeintegrator(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.solver_parameters_momentum_implicit)

        if self.solver.options.solve_salt:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
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
            self.timestepper_salt_3d = rungekutta.SSPRK33Stage(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_salt_vdff_3d = vert_timeintegrator(
                    solver.eq_salt_vdff, solver.fields.salt_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['salt'],
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
            self.timestepper_tke_3d = vert_timeintegrator(
                solver.eq_tke_diff, solver.fields.tke_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            self.timestepper_psi_3d = vert_timeintegrator(
                solver.eq_psi_diff, solver.fields.psi_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            if self.solver.options.use_turbulence_advection:
                fields = {'elev_3d': self.fields.elev_3d,
                          'uv_3d': self.fields.uv_3d,
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                          # uv_mag': self.fields.uv_mag_3d,
                          'uv_p1': self.fields.get('uv_p1_3d'),
                          'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                          }
                self.timestepper_tke_adv_eq = rungekutta.SSPRK33Stage(
                    solver.eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
                self.timestepper_psi_adv_eq = rungekutta.SSPRK33Stage(
                    solver.eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)

        # ----- stage 1 -----
        # from n to n+1 with RHS at (u_n, t_n)
        # u_init = u_n
        # ----- stage 2 -----
        # from n+1/4 to n+1/2 with RHS at (u_(1), t_{n+1})
        # u_init = 3/4*u_n + 1/4*u_(1)
        # ----- stage 3 -----
        # from n+1/3 to n+1 with RHS at (u_(2), t_{n+1/2})
        # u_init = 1/3*u_n + 2/3*u_(2)
        # -------------------

        # length of each step (fraction of dt)
        self.dt_frac = [1.0, 1.0/4.0, 2.0/3.0]
        # start of each step (fraction of dt)
        self.start_frac = [0.0, 1.0/4.0, 1.0/3.0]
        # weight to multiply u_n in weighted average to obtain start value
        self.stage_w = [1.0 - self.start_frac[0]]
        for i in range(1, len(self.dt_frac)):
            prev_end_time = self.start_frac[i-1] + self.dt_frac[i-1]
            self.stage_w.append(prev_end_time*(1.0 - self.start_frac[i]))
        print_output('dt_frac ' + str(self.dt_frac))
        print_output('start_frac ' + str(self.start_frac))
        print_output('stage_w ' + str(self.stage_w))

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            m = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/m
            print_output('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, m, f))
            self.M.append(m)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.fields.solution_2d

        self.sol2d_n.assign(sol2d)  # keep copy of elev_n
        for k in range(len(self.dt_frac)):
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt,
                                                    self.fields.uv_3d)
            with timed_stage('mode2d'):
                t_rhs = t + self.start_frac[k]*self.solver.dt
                dt_2d = self.dt_2d[k]
                # initialize
                w = self.stage_w[k]
                sol2d.assign(w*self.sol2d_n + (1.0-w)*sol2d)

                # advance fields from T_{n} to T{n+1}
                for i in range(self.M[k]):
                    self.timestepper2d.advance(t_rhs + i*dt_2d, dt_2d, sol2d,
                                               update_forcings)
            last_step = (k == 2)
            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=last_step,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


# OBSOLETE
class CoupledSSPIMEX(CoupledTimeIntegrator):
    """
    Solves coupled 3D equations with SSP IMEX scheme by [1], method (17).

    With this scheme all the equations can be advanced in time synchronously.

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, solver):
        super(CoupledSSPIMEX, self).__init__(solver, solver.options,
                                             solver.fields)
        self._initialized = False
        # for 2d shallow water eqns
        sp_impl = self.options.solver_parameters_sw
        sp_expl = self.options.solver_parameters_sw
        fs_2d = self.fields.solution_2d.function_space()
        self.solution_2d_old = Function(fs_2d, name='old_sol_2d')

        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'baroc_head': solver.fields.get('baroc_head_2d'),
            'viscosity_h': self.options.get('h_viscosity'),  # FIXME should be total h visc
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d,
            'linear_drag': self.options.linear_drag}
        self.timestepper2d = implicitexplicit.SSPIMEX(
            solver.eq_sw, self.solution_2d_old,
            fields, solver.dt,
            bnd_conditions=solver.bnd_functions['shallow_water'],
            solver_parameters=sp_expl, solver_parameters_dirk=sp_impl)
        # for 3D equations
        fs_mom = self.fields.uv_3d.function_space()
        self.uv_3d_old = Function(fs_mom, name='old_sol_mom')

        fields = {'eta': self.fields.elev_3d,  # FIXME rename elev
                  'baroc_head': self.fields.get('baroc_head_3d'),
                  'w': self.fields.w_3d,
                  'w_mesh': self.fields.get('w_mesh_3d'),
                  'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                  'viscosity_v': self.solver.tot_v_visc.get_sum(),
                  'viscosity_h': self.solver.tot_h_visc.get_sum(),
                  'source': self.options.uv_source_3d,
                  # uv_mag': self.fields.uv_mag_3d,
                  'uv_p1': self.fields.get('uv_p1_3d'),
                  'lax_friedrichs_factor': self.options.uv_lax_friedrichs,
                  'coriolis': self.fields.get('coriolis_3d'),
                  'linear_drag': self.options.linear_drag,
                  'wind_stress': self.fields.get('wind_stress_3d'),
                  }
        solver.eq_momentum.use_bottom_friction = True
        self.timestepper_mom_3d = implicitexplicit.SSPIMEX(
            solver.eq_momentum, self.uv_3d_old, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'])
        if self.solver.options.solve_salt:
            fs = self.fields.salt_3d.function_space()
            self.salt_3d_old = Function(fs, name='old_sol_salt')

            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': self.solver.tot_v_diff.get_sum(),
                      'source': self.options.salt_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                      }
            self.timestepper_salt_3d = implicitexplicit.SSPIMEX(
                solver.eq_salt, self.salt_3d_old, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'])
        if self.solver.options.use_turbulence:
            raise NotImplementedError('turbulence time stepper not implemented yet')
        self.n_stages = self.timestepper_mom_3d.n_stages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            last_step = k == self.n_stages - 1
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt, self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt, self.fields.uv_3d)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, self.solver.dt, self.fields.solution_2d,
                                               update_forcings)
            self._update_all_dependencies(t, do_vert_diffusion=False,
                                          do_2d_coupling=last_step,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step)


class CoupledSSPRKSemiImplicit(CoupledTimeIntegrator):
    """
    Solves coupled equations with simultaneous SSPRK33 stages, where 2d gravity
    waves are solved semi-implicitly. This saves CPU cos diffuses gravity waves.
    """
    def __init__(self, solver):
        super(CoupledSSPRKSemiImplicit, self).__init__(solver,
                                                       solver.options,
                                                       solver.fields)
        self._initialized = False

        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'baroc_head': solver.fields.get('baroc_head_2d'),
            'viscosity_h': self.options.get('h_viscosity'),  # FIXME should be total h visc
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d,
            'linear_drag': self.options.linear_drag}

        # vert_timeintegrator = rungekutta.DIRKLSPUM2
        vert_timeintegrator = rungekutta.BackwardEuler
        expl_timeintegrator_2d = rungekutta.SSPRK33StageSemiImplicit
        expl_timeintegrator = rungekutta.SSPRK33Stage
        # expl_timeintegrator_2d = rungekutta.ERKLPUM2StageSemiImplicit
        # expl_timeintegrator = rungekutta.ERKLPUM2Stage

        self.timestepper2d = expl_timeintegrator_2d(
            solver.eq_sw, self.fields.solution_2d,
            fields, solver.dt,
            bnd_conditions=solver.bnd_functions['shallow_water'],
            solver_parameters=self.options.solver_parameters_sw,
            semi_implicit=self.options.use_linearized_semi_implicit_2d,
            theta=self.options.shallow_water_theta)

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
                  }
        self.timestepper_mom_3d = expl_timeintegrator(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.solver_parameters_momentum_explicit)
        if self.solver.options.solve_vert_diffusion:
            fields = {'viscosity_v': implicit_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      }
            self.timestepper_mom_vdff_3d = vert_timeintegrator(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.solver_parameters_momentum_implicit)

        if self.solver.options.solve_salt:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
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
            self.timestepper_salt_3d = expl_timeintegrator(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_salt_vdff_3d = vert_timeintegrator(
                    solver.eq_salt_vdff, solver.fields.salt_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['salt'],
                    solver_parameters=self.options.solver_parameters_tracer_implicit)

        if self.solver.options.solve_temp:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
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
            self.timestepper_temp_3d = expl_timeintegrator(
                solver.eq_temp, solver.fields.temp_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['temp'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_temp_vdff_3d = vert_timeintegrator(
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
            self.timestepper_tke_3d = vert_timeintegrator(
                solver.eq_tke_diff, solver.fields.tke_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            self.timestepper_psi_3d = vert_timeintegrator(
                solver.eq_psi_diff, solver.fields.psi_3d, fields, solver.dt,
                solver_parameters=self.options.solver_parameters_tracer_implicit)
            if self.solver.options.use_turbulence_advection:
                fields = {'elev_3d': self.fields.elev_3d,
                          'uv_3d': self.fields.uv_3d,
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                          # uv_mag': self.fields.uv_mag_3d,
                          'uv_p1': self.fields.get('uv_p1_3d'),
                          'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                          }
                self.timestepper_tke_adv_eq = expl_timeintegrator(
                    solver.eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
                self.timestepper_psi_adv_eq = expl_timeintegrator(
                    solver.eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)

        self.n_stages = self.timestepper_mom_3d.n_stages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
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

        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.solver.fields.solution_2d

        for k in range(self.n_stages):
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.solve_stage(k, t, self.solver.dt,
                                                         self.fields.temp_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    # explicit advection
                    self.timestepper_tke_adv_eq.solve_stage(k, t, self.solver.dt,
                                                            self.solver.fields.tke_3d)
                    self.timestepper_psi_adv_eq.solve_stage(k, t, self.solver.dt,
                                                            self.solver.fields.psi_3d)
                    # if self.options.use_limiter_for_tracers:
                    #    self.solver.tracer_limiter.apply(self.solver.fields.tke_3d)
                    #    self.solver.tracer_limiter.apply(self.solver.fields.psi_3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt,
                                                    self.fields.uv_3d)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, self.solver.dt, sol2d,
                                               update_forcings)
            last_step = (k == 2)
            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=last_step,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class CoupledSSPRKSingleMode(CoupledTimeIntegrator):
    """
    Split-explicit SSPRK33 solver without mode-splitting.
    Both 2D and 3D modes are advanced with the same time step, computed based on 2D gravity
    wave speed. This time integrator is therefore expensive and should be only used for debugging etc.
    """
    def __init__(self, solver):
        super(CoupledSSPRKSingleMode, self).__init__(solver,
                                                     solver.options,
                                                     solver.fields)
        self._initialized = False

        uv_2d, eta_2d = self.fields.solution_2d.split()
        fields = {'elev_source': self.options.elev_source_2d,
                  'uv': uv_2d}
        self.timestepper2d = rungekutta.SSPRK33Stage(
            solver.eq_sw, eta_2d, fields, solver.dt_2d,
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

        # vert_timeintegrator = rungekutta.DIRKLSPUM2
        vert_timeintegrator = rungekutta.BackwardEuler

        fields = {'eta': self.fields.elev_3d,  # FIXME rename elev
                  'baroc_head': self.fields.get('baroc_head_3d'),
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
                  }
        self.timestepper_mom_3d = rungekutta.SSPRK33Stage(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt_2d,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.solver_parameters_momentum_explicit)
        if self.solver.options.solve_vert_diffusion:
            fields = {'viscosity_v': implicit_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      }
            self.timestepper_mom_vdff_3d = vert_timeintegrator(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt_2d,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.solver_parameters_momentum_implicit)

        if self.solver.options.solve_salt:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
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
            self.timestepper_salt_3d = rungekutta.SSPRK33Stage(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt_2d,
                bnd_conditions=solver.bnd_functions['salt'],
                solver_parameters=self.options.solver_parameters_tracer_explicit)
            if self.solver.options.solve_vert_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': implicit_v_diff,
                          }
                self.timestepper_salt_vdff_3d = vert_timeintegrator(
                    solver.eq_salt_vdff, solver.fields.salt_3d, fields, solver.dt_2d,
                    bnd_conditions=solver.bnd_functions['salt'],
                    solver_parameters=self.options.solver_parameters_tracer_implicit)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d.split()[1])
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def _update_2d_coupling(self):
        """Overloaded coupling function"""
        with timed_stage('aux_mom_coupling'):
            self.solver.uv_averager.solve()
            self.solver.extract_surf_dav_uv.solve()
            self.fields.uv_2d.assign(self.fields.uv_dav_2d)

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        for k in range(self.timestepper2d.n_stages):
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt_2d,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt_2d,
                                                    self.fields.uv_3d)
            with timed_stage('mode2d'):
                uv, elev = self.fields.solution_2d.split()
                self.timestepper2d.solve_stage(k, t, self.solver.dt_2d, elev,
                                               update_forcings)
            last_step = (k == 2)
            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=True,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class NewCoupledTimeIntegrator(CoupledTimeIntegrator):
    __metaclass__ = ABCMeta

    @abstractproperty
    def integrator_2d(self):
        pass

    @abstractproperty
    def integrator_3d(self):
        pass

    @abstractproperty
    def integrator_vert_3d(self):
        pass

    def __init__(self, solver):
        super(NewCoupledTimeIntegrator, self).__init__(solver,
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
        solver = self.solver
        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'baroc_head': solver.fields.get('baroc_head_2d'),
            'viscosity_h': self.options.get('h_viscosity'),  # FIXME should be total h visc
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d,
            'linear_drag': self.options.linear_drag}

        if issubclass(self.integrator_2d, (rungekutta.ERKSemiImplicitGeneric,
                                           rungekutta.ForwardEulerSemiImplicit)):
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
                  }
        self.timestepper_mom_3d = self.integrator_3d(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.solver_parameters_momentum_explicit)
        if self.solver.options.solve_vert_diffusion:
            fields = {'viscosity_v': implicit_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      }
            self.timestepper_mom_vdff_3d = self.integrator_vert_3d(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.solver_parameters_momentum_implicit)

        if self.solver.options.solve_salt:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
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

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
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

        self._initialized = True


# OBSOLETE
class CoupledForwardEuler(NewCoupledTimeIntegrator):
    """
    Forward Euler time integrator with mode-splitting.
    """
    integrator_2d = rungekutta.ForwardEulerStage
    integrator_3d = rungekutta.ForwardEulerStage
    integrator_vert_3d = rungekutta.BackwardEuler

    def _update_mesh_velocity_pre(self):
        fields = self.solver.fields
        self.solver.elev_2d_to_cg_projector.project()
        fields.w_mesh_surf_2d.assign(fields.elev_cg_2d)

    def _update_mesh_velocity_finalize(self):
        fields = self.solver.fields
        # compute w_mesh_surf from (elev - elev_old)/dt
        self.solver.elev_2d_to_cg_projector.project()
        fields.w_mesh_surf_2d *= -1
        fields.w_mesh_surf_2d += fields.elev_cg_2d
        fields.w_mesh_surf_2d *= 1.0/self.solver.dt
        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def compute_mesh_velocity(self, i_stage):
        """Computes mesh velocity from the rhs of 2D elevation equation"""

        # TODO allocate delev_dt in solver class
        # TODO add projector in solver class
        fs = self.solver.function_spaces.V_2d
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        dswe_dt = Function(fs)
        a = inner(trial, test)*dx
        l = self.timestepper2d.L_RK
        solve(a == l, dswe_dt)
        delev_dt = dswe_dt.split()[1]

        fields = self.solver.fields
        fields.w_mesh_surf_2d.project(delev_dt)

        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.solver.fields.solution_2d

        for k in range(self.n_stages):
            # NOTE compute w_mesh as cg elev difference is more accurate
            # TODO generalize to generic explicit RK scheme
            self._update_mesh_velocity_pre()
            # self.compute_mesh_velocity(k)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, self.solver.dt, sol2d,
                                               update_forcings)
            self._update_mesh_velocity_finalize()

            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.pre_solve(k, t, dt)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.pre_solve(k, t, dt)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.pre_solve(k, t, dt)
                    self.timestepper_psi_adv_eq.pre_solve(k, t, dt)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.pre_solve(k, t, dt)

            self._update_3d_elevation()
            self._update_moving_mesh()

            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.finalize_solve(k, t, dt)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.finalize_solve(k, t, dt)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.finalize_solve(k, t, dt)
                    self.timestepper_psi_adv_eq.finalize_solve(k, t, dt)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.finalize_solve(k, t, dt)

            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=True,
                                          do_2d_coupling=True,
                                          do_ale_update=False,
                                          do_stab_params=True,
                                          do_turbulence=True,
                                          do_mesh_velocity=False)


class CoupledERKALE(NewCoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation
    """
    integrator_2d = rungekutta.ERKLPUM2Stage
    integrator_3d = rungekutta.ERKLPUM2ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledERKALE, self).__init__(solver)
        self.elev_cg_old_2d = Function(self.solver.fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """Computes mesh velocity from the rhs of 2D elevation equation"""

        # TODO allocate delev_dt in solver class
        # TODO add projector in solver class
        fs = self.solver.function_spaces.V_2d
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        dswe_dt = Function(fs)
        a = inner(trial, test)*dx
        l = self.timestepper2d.L_RK
        solve(a == l, dswe_dt)
        delev_dt = dswe_dt.split()[1]

        fields = self.solver.fields
        fields.w_mesh_surf_2d.project(delev_dt)

        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def _update_mesh_velocity_pre(self, i_stage):
        fields = self.solver.fields
        self.solver.elev_2d_to_cg_projector.project()
        if i_stage == 0:
            self.elev_cg_old_2d.assign(fields.elev_cg_2d)
        fields.w_mesh_surf_2d.assign(self.elev_cg_old_2d)

    def _update_mesh_velocity_finalize(self, i_stage):
        fields = self.solver.fields
        ti = self.timestepper2d

        if i_stage > 1 and ti.beta[i_stage + 1][i_stage - 1] != 0.0:
            raise Exception('Unsupported RK scheme: stage {:} solution depends on intermediate solution {:}'.format(i_stage, i_stage-1))

        # construct delev/dt from RK equation
        fields.w_mesh_surf_2d.assign(self.elev_cg_old_2d)
        fields.w_mesh_surf_2d *= -float(ti.alpha[i_stage + 1][0])
        if i_stage > 0:
            a = float(ti.alpha[i_stage + 1][i_stage])
            fields.w_mesh_surf_2d += -a*fields.elev_cg_2d
        self.solver.elev_2d_to_cg_projector.project()
        fields.w_mesh_surf_2d += fields.elev_cg_2d
        c = float(ti.beta[i_stage + 1][i_stage])
        fields.w_mesh_surf_2d *= 1.0/self.solver.dt/c

        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.solver.fields.solution_2d

        for k in range(self.n_stages):
            self._update_mesh_velocity_pre(k)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, dt, sol2d,
                                               update_forcings)
            self._update_mesh_velocity_finalize(k)
            # self.compute_mesh_velocity(k)  # very bad
            # self._update_mesh_velocity()  # worse than computing diff(elev_cg)

            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.pre_solve(k, t, dt, update_forcings)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.pre_solve(k, t, dt, update_forcings)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.pre_solve(k, t, dt, update_forcings)
                    self.timestepper_psi_adv_eq.pre_solve(k, t, dt, update_forcings)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.pre_solve(k, t, dt, update_forcings)

            self._update_3d_elevation()
            self._update_moving_mesh()

            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.finalize_solve(k, t, dt, update_forcings)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.finalize_solve(k, t, dt, update_forcings)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.finalize_solve(k, t, dt, update_forcings)
                    self.timestepper_psi_adv_eq.finalize_solve(k, t, dt, update_forcings)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.finalize_solve(k, t, dt, update_forcings)

            # move fields to next stage
            last_step = (k == self.n_stages - 1)
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=True,
                                          do_ale_update=False,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step,
                                          do_mesh_velocity=False)


class CoupledIMEXALE(NewCoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation
    """
    integrator_2d = implicitexplicit.IMEXLPUM2
    integrator_3d = rungekutta.ERKLPUM2ALE
    # integrator_2d = timeintegrator.IMEXEuler
    # integrator_3d = rungekutta.ERKEulerALE
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
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """Compute mesh velocity from 2D solver runge-kutta scheme"""
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velcity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
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
            self.timestepper2d.erk.update_solution(k)
            self.timestepper_mom_3d.update_solution(k)
            self.timestepper_salt_3d.update_solution(k)
            if self.options.use_limiter_for_tracers:
                self.solver.tracer_limiter.apply(self.fields.salt_3d)
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
            self.timestepper2d.dirk.update_solution(k, additive=True)
            self.compute_mesh_velocity(k)
            # TODO update all dependencies of implicit solutions here
            # NOTE if 3D implicit solves, must be done in new mesh!
            self._update_3d_elevation()

            # - EX: evaluate explicit tendency
            self.timestepper_mom_3d.solve_tendency(k, t, update_forcings3d)
            self.timestepper_salt_3d.solve_tendency(k, t, update_forcings3d)
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


# OBSOLETE
class CoupledIMEXEuler(NewCoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation
    """
    integrator_2d = implicitexplicit.IMEXEuler
    integrator_3d = rungekutta.ERKEulerALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledIMEXEuler, self).__init__(solver)

        self.elev_cg_old_2d = []
        for i in range(self.n_stages + 1):
            f = Function(self.solver.fields.elev_cg_2d)
            self.elev_cg_old_2d.append(f)

        import numpy.linalg as linalg
        ti = self.timestepper2d.dirk
        assert ti.is_implicit
        self.a_inv = linalg.inv(ti.a)

    def _compute_mesh_velocity_pre(self, i_stage):
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """Compute mesh velocity from 2D solver runge-kutta scheme"""
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velcity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
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
            # update 2D equations
            self._compute_mesh_velocity_pre(k)
            with timed_stage('mode2d'):
                self.timestepper2d.solve_stage(k, t, update_forcings)
                # self.timestepper2d.get_final_solution()
            self.compute_mesh_velocity(k)

            # self._update_moving_mesh()
            # self._update_mesh_velocity()

            # 3D update
            # - EX: set solution to u_n + dt*sum(a*k_erk)
            # - IM: solve implicit tendency (this is implicit solve)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            # - EX: evaluate explicit tendency
            with timed_stage('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                    # self.timestepper_salt_vdff_3d.solve_tendency(k, t)
                    # self.timestepper_salt_vdff_3d.update_solution(k, additive=True)
                    self.timestepper_salt_3d.solve_tendency(k, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temp:
                    self.timestepper_temp_3d.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                    # self.timestepper_temp_vdff_3d.solve_tendency(k, t)
                    # self.timestepper_temp_vdff_3d.update_solution(k, additive=True)
                    self.timestepper_temp_3d.solve_tendency(k, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.update_solution(k)
                    self.timestepper_psi_adv_eq.update_solution(k)
                    # self.timestepper_tke_3d.solve_tendency(k, t)
                    # self.timestepper_tke_3d.update_solution(k, additive=True)
                    # self.timestepper_psi_3d.solve_tendency(k, t)
                    # self.timestepper_psi_3d.update_solution(k, additive=True)
                    self.timestepper_tke_adv_eq.solve_tendency(k, t, update_forcings3d)
                    self.timestepper_psi_adv_eq.solve_tendency(k, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timestepper_mom_3d.update_solution(k)
                # self._update_vertical_velocity()
                # self.timestepper_mom_vdff_3d.solve_tendency(k, t)
                # self.timestepper_mom_vdff_3d.update_solution(k, additive=True)
                self.timestepper_mom_3d.solve_tendency(k, t, update_forcings3d)

            self._update_moving_mesh()

            # move fields to next stage
            last_step = (k == self.n_stages - 1)
            if last_step:
                # final solutions
                self.timestepper2d.get_final_solution()
                self._update_moving_mesh()
                if self.options.solve_salt:
                    self.timestepper_salt_3d.get_final_solution()
                    # self.timestepper_salt_vdff_3d.get_final_solution(additive=True)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temp:
                    self.timestepper_temp_3d.get_final_solution()
                    # self.timestepper_temp_vdff_3d.get_final_solution(additive=True)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                if self.options.use_turbulence_advection:
                    self.timestepper_tke_adv_eq.get_final_solution()
                    self.timestepper_psi_adv_eq.get_final_solution()
                    # self.timestepper_tke_eq.get_final_solution(additive=True)
                    # self.timestepper_psi_eq.get_final_solution(additive=True)
                self.timestepper_mom_3d.get_final_solution()
                # self.timestepper_mom_vdff_3d.get_final_solution(additive=True)

            # NOTE elev_3d is used in all equations
            # NOTE should only update after all equations have been evaluated
            # self._update_moving_mesh()

            self._update_all_dependencies(t, do_vert_diffusion=False,
                                          do_2d_coupling=True,
                                          do_ale_update=False,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step,
                                          do_mesh_velocity=False)


class CoupledLeapFrogAM3(NewCoupledTimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 time integrator for coupled 2D-3D problem
    """
    integrator_2d = rungekutta.CrankNicolsonRK
    integrator_3d = timeintegrator.LeapFrogAM3
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledLeapFrogAM3, self).__init__(solver)
        self.elev_old_3d = Function(self.fields.elev_3d)
        self.uv_old_2d = Function(self.fields.uv_2d)
        self.uv_new_2d = Function(self.fields.uv_2d)

    def _update_mesh_velocity_pre(self):
        fields = self.solver.fields
        self.solver.elev_2d_to_cg_projector.project()
        fields.w_mesh_surf_2d.assign(fields.elev_cg_2d)

    def _update_mesh_velocity_finalize(self):
        fields = self.solver.fields
        # compute w_mesh_surf from (elev - elev_old)/dt
        self.solver.elev_2d_to_cg_projector.project()
        fields.w_mesh_surf_2d *= -1
        fields.w_mesh_surf_2d += fields.elev_cg_2d
        fields.w_mesh_surf_2d *= 1.0/self.solver.dt
        # use that to compute w_mesh in whole domain
        self.solver.copy_surf_w_mesh_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        # -------------------
        # prediction step - from t_{n-1/2} to t_{n+1/2}, solution at t_{n}
        # -------------------

        self.fields.w_mesh_3d.assign(0.0)

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

        # update 2D
        self.uv_old_2d.assign(self.fields.uv_2d)
        self._update_mesh_velocity_pre()
        self.timestepper2d.advance(t, update_forcings)
        self._update_mesh_velocity_finalize()
        self.uv_new_2d.assign(self.fields.uv_2d)

        # set 3D elevation to half step
        self.elev_old_3d.assign(self.fields.elev_3d)
        self.solver.copy_elev_to_3d.solve()
        self.fields.elev_3d *= 0.5
        self.fields.elev_3d += 0.5*self.elev_old_3d

        # correct uv_3d to uv_2d at t_{n+1/2}
        self.fields.uv_2d *= 0.5
        self.fields.uv_2d += 0.5*self.uv_old_2d
        self._update_2d_coupling()
        self.fields.uv_2d.assign(self.uv_new_2d)  # restore
        self._update_vertical_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()

        self._update_moving_mesh()
        # self._update_mesh_velocity()

        # -------------------
        # correction step - from t_{n} to t_{n+1}, solution at t_{n+1/2}
        # -------------------

        with timed_stage('salt_eq'):
            if self.options.solve_salt:
                self.timestepper_salt_3d.eval_rhs()
                self.timestepper_salt_3d.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temp:
                self.timestepper_temp_3d.eval_rhs()
                self.timestepper_temp_3d.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if self.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq.eval_rhs()
                self.timestepper_psi_adv_eq.eval_rhs()
                self.timestepper_tke_adv_eq.correct()
                self.timestepper_psi_adv_eq.correct()
        with timed_stage('momentum_eq'):
            self.timestepper_mom_3d.eval_rhs()
            self.timestepper_mom_3d.correct()

        self._update_3d_elevation()
        self._update_2d_coupling()
        self._update_vertical_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()
        self._update_turbulence(t)
        if self.options.solve_vert_diffusion:
            with timed_stage('impl_mom_vvisc'):
                self.timestepper_mom_vdff_3d.advance(t, self.solver.dt, self.fields.uv_3d)
            if self.options.solve_salt:
                with timed_stage('impl_salt_vdiff'):
                    self.timestepper_salt_vdff_3d.advance(t, self.solver.dt, self.fields.salt_3d)
            if self.options.solve_temp:
                with timed_stage('impl_temp_vdiff'):
                    self.timestepper_temp_vdff_3d.advance(t, self.solver.dt, self.fields.temp_3d)
        self._update_stabilization_params()
