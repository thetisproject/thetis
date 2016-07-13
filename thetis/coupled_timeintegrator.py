"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator

# TODO turbulence update. move out from _update_all_dependencies ?


class CoupledTimeIntegrator(timeintegrator.TimeIntegrator):
    """Base class for coupled time integrators"""
    def __init__(self, solver, options, fields):
        self.solver = solver
        self.options = options
        self.fields = fields

    def _update_3d_elevation(self):
        """Projects elevation to 3D"""
        with timed_stage('aux_elev_3d'):
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}
            self.solver.elev_3d_to_cg_projector.project()

    def _update_vertical_velocity(self):
        with timed_stage('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
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
                                 do_turbulence=False):
        """Default routine for updating all dependent fields after a time step"""
        self._update_3d_elevation()
        if do_ale_update:
            self._update_moving_mesh()
        if do_vert_diffusion and self.options.solve_vert_diffusion:
            with timed_stage('impl_mom_vvisc'):
                self.timestepper_mom_vdff_3d.advance(t, self.solver.dt, self.fields.uv_3d)
            if self.options.solve_salt:
                with timed_stage('impl_salt_vdiff'):
                    self.timestepper_salt_vdff_3d.advance(t, self.solver.dt, self.fields.salt_3d)
            if self.options.solve_temp:
                with timed_stage('impl_temp_vdiff'):
                    self.timestepper_temp_vdff_3d.advance(t, self.solver.dt, self.fields.temp_3d)
        if do_2d_coupling:
            self._update_2d_coupling()
        self._update_vertical_velocity()
        if do_turbulence:
            self._update_turbulence(t)
        self._update_mesh_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()
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
        self.timestepper2d = timeintegrator.SSPRK33(
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
        vert_timeintegrator = timeintegrator.BackwardEuler

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
        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
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
            self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
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
                self.timestepper_tke_adv_eq = timeintegrator.SSPRK33Stage(
                    solver.eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
                self.timestepper_psi_adv_eq = timeintegrator.SSPRK33Stage(
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
        self.timestepper2d = timeintegrator.SSPIMEX(
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
        self.timestepper_mom_3d = timeintegrator.SSPIMEX(
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
            self.timestepper_salt_3d = timeintegrator.SSPIMEX(
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
                    if self.options.use_limiter_for_tracers and last_step:
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

        self.timestepper2d = timeintegrator.SSPRK33StageSemiImplicit(
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

        # vert_timeintegrator = timeintegrator.DIRKLSPUM2
        vert_timeintegrator = timeintegrator.BackwardEuler

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
        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
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
            self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
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
            self.timestepper_temp_3d = timeintegrator.SSPRK33Stage(
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
                self.timestepper_tke_adv_eq = timeintegrator.SSPRK33Stage(
                    solver.eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)
                self.timestepper_psi_adv_eq = timeintegrator.SSPRK33Stage(
                    solver.eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    solver_parameters=self.options.solver_parameters_tracer_explicit)

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
        sol2d = self.solver.fields.solution_2d

        for k in range(len(self.dt_frac)):
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
        self.timestepper2d = timeintegrator.SSPRK33Stage(
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

        # vert_timeintegrator = timeintegrator.DIRKLSPUM2
        vert_timeintegrator = timeintegrator.BackwardEuler

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
        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
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
            self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
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
