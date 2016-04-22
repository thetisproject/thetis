"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from utility import *
import timeintegrator

# TODO turbulence update. move out from _update_all_dependencies ?
# TODO add default functions for updating all eqns to base class?
# TODO add decorator for accessing s, o, f


class CoupledTimeIntegrator(timeintegrator.TimeIntegrator):
    """Base class for coupled time integrators"""
    def __init__(self, solver, options, fields):
        self.solver = solver
        self.options = options
        self.fields = fields

    def _update_3d_elevation(self):
        """Projects elevation to 3D"""
        with timed_region('aux_elev_3d'):
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}
            self.solver.elev_3d_to_cg_projector.project()

    def _update_vertical_velocity(self):
        with timed_region('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_region('aux_mesh_ale'):
                self.solver.mesh_coord_updater.solve()
                compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
                self.solver.copy_v_elem_size_to_2d.solve()

    def _update_bottom_friction(self):
        """Computes bottom friction related fields"""
        if self.options.use_bottom_friction:
            with timed_region('aux_friction'):
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
        with timed_region('aux_mom_coupling'):
            self.solver.uv_averager.solve()
            self.solver.extract_surf_dav_uv.solve()
            self.solver.copy_uv_dav_to_uv_dav_3d.solve()
            # 2d-3d coupling: restart 2d mode from depth ave uv_3d
            # NOTE unstable!
            # uv_2d_start = sol2d.split()[0]
            # uv_2d_start.assign(self.fields.uv_dav_2d)
            # 2d-3d coupling v2: force DAv(uv_3d) to uv_2d
            # self.solver.uv_dav_to_tmp_projector.project()  # project uv_dav to uv_3d_tmp
            # self.fields.uv_3d -= self.fields.uv_3d_tmp
            # self.solver.uv_2d_to_dav_projector.project()  # uv_2d to uv_dav_2d
            # copy_2d_field_to_3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
            #                 elem_height=self.fields.v_elem_size_3d)
            # self.fields.uv_dav_to_tmp_projector.project()  # project uv_dav to uv_3d_tmp
            # self.fields.uv_3d += self.fields.uv_3d_tmp
            self.fields.uv_3d -= self.fields.uv_dav_3d
            self.solver.copy_uv_to_uv_dav_3d.solve()
            self.fields.uv_3d += self.fields.uv_dav_3d

    def _update_mesh_velocity(self):
        """Computes ALE mesh velocity"""
        if self.options.use_ale_moving_mesh:
            with timed_region('aux_mesh_ale'):
                self.solver.w_mesh_solver.solve()

    def _update_baroclinicity(self):
        """Computes baroclinic head"""
        if self.options.baroclinic:
            with timed_region('aux_barolinicity'):
                compute_baroclinic_head(self.solver, self.fields.salt_3d, self.fields.baroc_head_3d,
                                        self.fields.baroc_head_2d, self.fields.baroc_head_int_3d,
                                        self.fields.bathymetry_3d)

    def _update_turbulence(self, t):
        """Updates turbulence related fields"""
        if self.options.use_turbulence:
            with timed_region('turbulence'):
                    self.solver.gls_model.preprocess()
                    # NOTE psi must be solved first as it depends on tke
                    self.timestepper_psi_3d.advance(t, self.solver.dt, self.solver.fields.psi_3d)
                    self.timestepper_tke_3d.advance(t, self.solver.dt, self.solver.fields.tke_3d)
                    # NOTE tke and eps are P0, nothing to limit
                    # if self.options.use_limiter_for_tracers:
                    #     self.solver.tracer_limiter.apply(self.solver.fields.tke_3d)
                    #     self.solver.tracer_limiter.apply(self.solver.fields.psi_3d)
                    self.solver.gls_model.postprocess()

    def _update_stabilization_params(self):
        """Computes Smagorinsky viscosity etc fields"""
        # update velocity magnitude
        self.solver.uv_mag_solver.solve()
        # update P1 velocity field
        self.solver.uv_p1_projector.project()
        if self.options.smagorinsky_factor is not None:
            with timed_region('aux_stabilization'):
                self.solver.smagorinsky_diff_solver.solve()
        if self.options.salt_jump_diff_factor is not None:
            with timed_region('aux_stabilization'):
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
        with timed_region('vert_diffusion'):
            if do_vert_diffusion and self.options.solve_vert_diffusion:
                self.timestepper_mom_vdff_3d.advance(t, self.solver.dt, self.fields.uv_3d)
                if self.options.solve_salt:
                    self.timestepper_salt_vdff_3d.advance(t, self.solver.dt, self.fields.salt_3d)
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
        self.timestepper2d = timeintegrator.SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        fs = self.timestepper2d.solution_old.function_space()
        self.sol2d_n = Function(fs, name='sol2dtmp')

        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.options.solve_salt:
            self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d = timeintegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

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
        print_info('dt_frac ' + str(self.dt_frac))
        print_info('start_frac ' + str(self.start_frac))
        print_info('stage_w ' + str(self.stage_w))

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
            print_info('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, m, f))
            self.M.append(m)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.timestepper2d.equation.solution

        self.sol2d_n.assign(sol2d)  # keep copy of elev_n
        for k in range(len(self.dt_frac)):
            with timed_region('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_region('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt,
                                                    self.fields.uv_3d)
            with timed_region('mode2d'):
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
        sp_impl = {'ksp_type': 'gmres',
                   'pc_type': 'fieldsplit',
                   'pc_fieldsplit_type': 'multiplicative',
                   }
        sp_expl = {'ksp_type': 'gmres',
                   'pc_type': 'fieldsplit',
                   'pc_fieldsplit_type': 'multiplicative',
                   }
        fs_2d = solver.eq_sw.solution.function_space()
        self.solution_2d_old = Function(fs_2d, name='old_sol_2d')
        self.timestepper2d = timeintegrator.SSPIMEX(
            solver.eq_sw, solver.dt,
            solver_parameters=sp_expl,
            solver_parameters_dirk=sp_impl,
            solution=self.solution_2d_old)
        # for 3D equations
        sp_impl = {'ksp_type': 'gmres',
                   }
        sp_expl = {'ksp_type': 'gmres',
                   }
        fs_mom = solver.eq_momentum.solution.function_space()
        self.uv_3d_old = Function(fs_mom, name='old_sol_mom')
        self.timestepper_mom_3d = timeintegrator.SSPIMEX(
            solver.eq_momentum, solver.dt,
            solver_parameters=sp_expl,
            solver_parameters_dirk=sp_impl,
            solution=self.uv_3d_old)
        if self.solver.options.solve_salt:
            fs = solver.eq_salt.solution.function_space()
            self.salt_3d_old = Function(fs, name='old_sol_salt')
            self.timestepper_salt_3d = timeintegrator.SSPIMEX(
                solver.eq_salt, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.salt_3d_old)
        if self.solver.options.solve_vert_diffusion:
            raise Exception('vert mom eq should not exist for this time integrator')
            self.timestepper_mom_vdff_3d = timeintegrator.SSPIMEX(
                solver.eq_vertmomentum, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl)
        if self.solver.options.use_turbulence:
            fs = solver.eq_tke_diff.solution.function_space()
            self.tke_3d_old = Function(fs, name='old_sol_tke')
            self.timestepper_tke_3d = timeintegrator.SSPIMEX(
                solver.eq_tke_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.tke_3d_old)
            self.psi_3d_old = Function(fs, name='old_sol_psi')
            self.timestepper_psi_3d = timeintegrator.SSPIMEX(
                solver.eq_psi_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.psi_3d_old)
        self.n_stages = self.timestepper_mom_3d.n_stages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            last_step = k == self.n_stages - 1
            with timed_region('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt, self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers and last_step:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_region('turbulence_advection'):
                if self.options.use_turbulence_advection:
                    # explicit advection
                    self.timestepper_tke_adv_eq.solve_stage(k, t, self.solver.dt, self.fields.tke_3d)
                    self.timestepper_psi_adv_eq.solve_stage(k, t, self.solver.dt, self.fields.psi_3d)
                    if self.options.use_limiter_for_tracers and last_step:
                        self.solver.tracer_limiter.apply(self.fields.tke_3d)
                        self.solver.tracer_limiter.apply(self.fields.psi_3d)
            with timed_region('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt, self.fields.uv_3d)
            with timed_region('mode2d'):
                self.timestepper2d.solve_stage(k, t, self.solver.dt, self.fields.solution_2d,
                                               update_forcings)
            with timed_region('turbulence'):
                if self.options.use_turbulence:
                    # NOTE psi must be solved first as it depends on tke
                    self.timestepper_psi_3d.solve_stage(k, t, self.solver.dt, self.fields.psi_3d)
                    self.timestepper_tke_3d.solve_stage(k, t, self.solver.dt, self.fields.tke_3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.tke_3d)
                        self.solver.tracer_limiter.apply(self.fields.psi_3d)
                    self.solver.gls_model.postprocess()
                    self.solver.gls_model.preprocess()  # for next iteration
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
        self.timestepper2d = timeintegrator.SSPRK33StageSemiImplicit(
            solver.eq_sw, solver.dt,
            solver.eq_sw.solver_parameters)

        vdiff_sp = {'snes_monitor': False,
                    'ksp_type': 'gmres',
                    'pc_type': 'ilu',
                    'snes_atol': 1e-27,
                    }
        # vert_timeintegrator = timeintegrator.DIRKLSPUM2
        vert_timeintegrator = timeintegrator.BackwardEuler
        # vert_timeintegrator = timeintegrator.ImplicitMidpoint
        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)

        # assign viscosity/diffusivity to correct equations
        if self.options.solve_vert_diffusion:
            # implicit_v_visc = solver.tot_v_visc.get_sum()
            # explicit_v_visc = None
            # implicit_v_diff = solver.tot_v_diff.get_sum()
            explicit_v_diff = None
        else:
            # implicit_v_visc = None
            # explicit_v_visc = solver.tot_v_visc.get_sum()
            # implicit_v_diff = None
            explicit_v_diff = solver.tot_v_diff.get_sum()

        if self.solver.options.solve_salt:
            # self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
            #     solver.eq_salt,
            #     solver.dt)
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'dw_mesh_dz': self.fields.get('w_mesh_ddz_3d'),
                      'diffusivity_h': solver.tot_h_diff.get_sum(),
                      'diffusivity_v': explicit_v_diff,
                      'source': self.options.salt_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_factor': self.options.tracer_lax_friedrichs,
                      }
            self.timestepper_salt_3d = timeintegrator.SSPRK33StageNew(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'])
            if self.solver.options.solve_vert_diffusion:
                self.timestepper_salt_vdff_3d = vert_timeintegrator(
                    solver.eq_salt_vdff, solver.dt, solver_parameters=vdiff_sp)

        if self.solver.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d = vert_timeintegrator(
                solver.eq_vertmomentum, solver.dt, solver_parameters=vdiff_sp)
        if self.solver.options.use_turbulence:
            self.timestepper_tke_3d = vert_timeintegrator(
                solver.eq_tke_diff, solver.dt, solver_parameters=vdiff_sp)
            self.timestepper_psi_3d = vert_timeintegrator(
                solver.eq_psi_diff, solver.dt, solver_parameters=vdiff_sp)
            if self.solver.options.use_turbulence_advection:
                self.timestepper_tke_adv_eq = timeintegrator.SSPRK33Stage(solver.eq_tke_adv, solver.dt)
                self.timestepper_psi_adv_eq = timeintegrator.SSPRK33Stage(solver.eq_psi_adv, solver.dt)
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
        print_info('dt_frac ' + str(self.dt_frac))
        print_info('start_frac ' + str(self.start_frac))
        print_info('stage_w ' + str(self.stage_w))
        self.n_stages = self.timestepper_mom_3d.n_stages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
            if self.options.solve_vert_diffusion:
                self.timestepper_salt_vdff_3d.initialize(self.fields.salt_3d)
        if self.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            m = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/m
            print_info('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, m, f))
            self.M.append(m)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.timestepper2d.equation.solution

        for k in range(len(self.dt_frac)):
            with timed_region('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_region('turbulence_advection'):
                if self.options.use_turbulence_advection:
                    # explicit advection
                    self.timestepper_tke_adv_eq.solve_stage(k, t, self.solver.dt,
                                                            self.solver.fields.tke_3d)
                    self.timestepper_psi_adv_eq.solve_stage(k, t, self.solver.dt,
                                                            self.solver.fields.psi_3d)
                    # if self.options.use_limiter_for_tracers:
                    #    self.solver.tracer_limiter.apply(self.solver.fields.tke_3d)
                    #    self.solver.tracer_limiter.apply(self.solver.fields.psi_3d)
            with timed_region('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt,
                                                    self.fields.uv_3d)
            with timed_region('mode2d'):
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
        self.timestepper2d = timeintegrator.SSPRK33Stage(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)

        self.timestepper_mom_3d = timeintegrator.SSPRK33Stage(
            solver.eq_momentum,
            solver.dt_2d)
        if self.solver.options.solve_salt:
            self.timestepper_salt_3d = timeintegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt_2d)
        if self.solver.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d = timeintegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt_2d, gamma=0.6)

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
        with timed_region('aux_mom_coupling'):
            self.solver.uv_averager.solve()
            self.solver.extract_surf_dav_uv.solve()
            self.fields.uv_2d.assign(self.fields.uv_dav_2d)

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        for k in range(self.timestepper2d.n_stages):
            with timed_region('salt_eq'):
                if self.options.solve_salt:
                    self.timestepper_salt_3d.solve_stage(k, t, self.solver.dt_2d,
                                                         self.fields.salt_3d,
                                                         update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_region('momentum_eq'):
                self.timestepper_mom_3d.solve_stage(k, t, self.solver.dt_2d,
                                                    self.fields.uv_3d)
            with timed_region('mode2d'):
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


class CoupledSSPRK(CoupledTimeIntegrator):
    """
    Split-explicit time integration that uses SSPRK for both 2d and 3d modes.
    2D mode is solved only once from t_{n} to t_{n+2} using shorter time steps.
    To couple with the 3D mode, 2D variables are averaged in time to get values at t_{n+1} and t_{n+1/2}.
    """
    def __init__(self, solver):
        super(CoupledSSPRK, self).__init__(solver,
                                           solver.options,
                                           solver.fields)
        self._initialized = False
        subiterator = SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        self.timestepper2d = timeintegrator.MacroTimeStepIntegrator(
            subiterator,
            solver.M_modesplit,
            restart_from_av=True)

        # FIXME self.fields.elev_3d_nplushalf is not defined!
        self.timestepper_mom_3d = timeintegrator.SSPRK33(
            solver.eq_momentum, solver.dt,
            funcs_nplushalf={'eta': self.fields.elev_3d_nplushalf})
        if self.solver.options.solve_salt:
            self.timestepper_salt_3d = timeintegrator.SSPRK33(
                solver.eq_salt,
                solver.dt)
        if self.solver.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d = timeintegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)
        elev_nph = self.timestepper2d.solution_nplushalf.split()[1]
        self.copy_elev_to_3d_nplushalf = ExpandFunctionTo3d(elev_nph, self.fields.elev_3d_nplushalf)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timestepper2d.initialize(self.fields.solution_2d)
        self.timestepper_mom_3d.initialize(self.fields.uv_3d)
        if self.options.options.solve_salt:
            self.timestepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.options.solve_vert_diffusion:
            self.timestepper_mom_vdff_3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def _update_2d_coupling(self):
        """Overloaded coupling function"""
        with timed_region('aux_mom_coupling'):
            self.solver.uv_averager.solve()
            self.solver.extract_surf_dav_uv.solve()
            self.solver.copy_uv_dav_to_uv_dav_3d.solve()
            # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
            uv_2d_start = self.timestepper2d.solution_start.split()[0]
            uv_2d_start.assign(self.fields.uv_dav_2d)

    def advance(self, t, dt, update_forcings=None, update_forcings3d=None):
        """Advances the equations for one time step"""
        # SSPRK33 time integration loop
        with timed_region('mode2d'):
            self.timestepper2d.advance(t, self.solver.dt_2d,
                                       self.fields.solution_2d,
                                       update_forcings)
        with timed_region('aux_elev_3d'):
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}
            self.copy_elev_to_3d_nplushalf.solve()  # at t_{n+1/2}
        self._update_moving_mesh()
        self._updateBottomFriction()
        self._updateBaroclinicity()
        with timed_region('momentum_eq'):
            self.timestepper_mom_3d.advance(t, self.solver.dt,
                                            self.fields.uv_3d,
                                            update_forcings3d)
        with timed_region('vert_diffusion'):
            if self.options.solve_vert_diffusion:
                self.timestepper_mom_vdff_3d.advance(t, self.solver.dt,
                                                     self.fields.uv_3d, None)
        self._updateVerticalVelocity()
        self._updateMeshVelocity()
        with timed_region('salt_eq'):
            if self.options.solve_salt:
                self.timestepper_salt_3d.advance(t, self.solver.dt,
                                                 self.fields.salt_3d,
                                                 update_forcings3d)
        self._update_2d_coupling()
