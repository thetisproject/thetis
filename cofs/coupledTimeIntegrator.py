"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from utility import *
import timeIntegrator

# TODO turbulence update. move out from _updateAllDependencies ?
# TODO add default functions for updating all eqns to base class?
# TODO add decorator for accessing s, o, f


class coupledTimeIntegrator(timeIntegrator.timeIntegrator):
    """Base class for coupled time integrators"""
    def __init__(self, solver, options, fields):
        self.solver = solver
        self.options = options
        self.fields = fields

    def _update3dElevation(self):
        """Projects elevation to 3D"""
        with timed_region('aux_elev_3d'):
            copy2dFieldTo3d(self.fields.elev_2d, self.fields.elev_3d)  # at t_{n+1}
            self.solver.elev_3d_to_CG_projector.project()

    def _updateVerticalVelocity(self):
        with timed_region('continuityEq'):
            computeVertVelocity(self.fields.w_3d, self.fields.uv_3d, self.fields.bathymetry_3d,
                                self.solver.eq_momentum.boundary_markers,
                                self.solver.eq_momentum.bnd_functions)

    def _updateMovingMesh(self):
        """Updates mesh to match elevation field"""
        if self.options.useALEMovingMesh:
            with timed_region('aux_mesh_ale'):
                updateCoordinates(
                    self.solver.mesh, self.fields.elev_3d, self.fields.bathymetry_3d,
                    self.fields.z_coord_3d, self.fields.z_coord_ref_3d)
                computeElemHeight(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
                copy3dFieldTo2d(self.fields.v_elem_size_3d, self.fields.v_elem_size_2d)

    def _updateBottomFriction(self):
        """Computes bottom friction related fields"""
        if self.options.useBottomFriction:
            with timed_region('aux_friction'):
                self.solver.uvP1_projector.project()
                computeBottomFriction(
                    self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                    self.fields.uv_bottom_3d, self.fields.z_coord_3d,
                    self.fields.z_bottom_2d,
                    self.fields.bathymetry_2d, self.fields.bottom_drag_2d,
                    self.fields.bottom_drag_3d,
                    self.fields.v_elem_size_2d, self.fields.v_elem_size_3d)
        if self.options.useParabolicViscosity:
            computeParabolicViscosity(
                self.fields.uv_bottom_3d, self.fields.bottom_drag_3d,
                self.fields.bathymetry_3d,
                self.fields.parab_visc_3d)

    def _update2DCoupling(self):
        """Does 2D-3D coupling for the velocity field"""
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(self.fields.uv_3d, self.fields.uv_dav_3d,
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=self.fields.bathymetry_3d)
            copy3dFieldTo2d(self.fields.uv_dav_3d, self.fields.uv_dav_2d,
                            useBottomValue=False,
                            elemHeight=self.fields.v_elem_size_2d)
            copy2dFieldTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                            elemHeight=self.fields.v_elem_size_3d)
            # 2d-3d coupling: restart 2d mode from depth ave uv_3d
            # NOTE unstable!
            #uv_2d_start = sol2d.split()[0]
            #uv_2d_start.assign(self.fields.uv_dav_2d)
            # 2d-3d coupling v2: force DAv(uv_3d) to uv_2d
            #self.solver.uvDAV_to_tmp_projector.project()  # project uv_dav to uv_3d_tmp
            #self.fields.uv_3d -= self.fields.uv_3d_tmp
            #self.solver.uv_2d_to_DAV_projector.project()  # uv_2d to uv_dav_2d
            #copy2dFieldTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                            #elemHeight=self.fields.v_elem_size_3d)
            #self.fields.uvDAV_to_tmp_projector.project()  # project uv_dav to uv_3d_tmp
            #self.fields.uv_3d += self.fields.uv_3d_tmp
            self.fields.uv_3d -= self.fields.uv_dav_3d
            copy2dFieldTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                            elemHeight=self.fields.v_elem_size_3d)
            self.fields.uv_3d += self.fields.uv_dav_3d

    def _updateMeshVelocity(self):
        """Computes ALE mesh velocity"""
        if self.options.useALEMovingMesh:
            with timed_region('aux_mesh_ale'):
                computeMeshVelocity(
                    self.fields.elev_3d, self.fields.uv_3d, self.fields.w_3d,
                    self.fields.w_mesh_3d, self.fields.w_mesh_surf_3d,
                    self.fields.w_mesh_surf_2d,
                    self.fields.w_mesh_ddz_3d, self.fields.bathymetry_3d,
                    self.fields.z_coord_ref_3d)

    def _updateBaroclinicity(self):
        """Computes baroclinic head"""
        if self.options.baroclinic:
            with timed_region('aux_barolinicity'):
                computeBaroclinicHead(self.fields.salt_3d, self.fields.baroc_head_3d,
                                      self.fields.baro_head_2d, self.fields.baroc_head_int_3d,
                                      self.fields.bathymetry_3d)

    def _updateTurbulence(self, t):
        """Updates turbulence related fields"""
        if self.options.useTurbulence:
            with timed_region('turbulence'):
                    self.solver.glsModel.preprocess()
                    # NOTE psi must be solved first as it depends on tke
                    self.timeStepper_psi_3d.advance(t, self.solver.dt, self.solver.fields.psi_3d)
                    self.timeStepper_tke_3d.advance(t, self.solver.dt, self.solver.fields.tke_3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.solver.fields.tke_3d)
                        self.solver.tracerLimiter.apply(self.solver.fields.psi_3d)
                    self.solver.glsModel.postprocess()

    def _updateStabilizationParams(self):
        """Computes Smagorinsky viscosity etc fields"""
        # update velocity magnitude
        computeVelMagnitude(self.fields.uv_mag_3d, u=self.fields.uv_3d)
        # update P1 velocity field
        self.solver.uvP1_projector.project()
        if self.options.smagorinskyFactor is not None:
            with timed_region('aux_stabilization'):
                smagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                     self.options.smagorinskyFactor, self.fields.h_elem_size_3d)
        if self.options.salt_jump_diffFactor is not None:
            with timed_region('aux_stabilization'):
                computeHorizJumpDiffusivity(self.options.salt_jump_diffFactor, self.fields.salt_3d,
                                            self.fields.salt_jump_diff, self.fields.h_elem_size_3d,
                                            self.fields.uv_mag_3d, self.options.saltRange,
                                            self.fields.max_h_diff)

    def _updateAllDependencies(self, t,
                               do2DCoupling=False,
                               doVertDiffusion=False,
                               doALEUpdate=False,
                               doStabParams=False,
                               doTurbulence=False):
        """Default routine for updating all dependent fields after a time step"""
        self._update3dElevation()
        if doALEUpdate:
            self._updateMovingMesh()
        with timed_region('vert_diffusion'):
            if doVertDiffusion and self.options.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, self.solver.dt, self.fields.uv_3d)
        if do2DCoupling:
            self._update2DCoupling()
        self._updateVerticalVelocity()
        if doTurbulence:
            self._updateTurbulence(t)
        self._updateMeshVelocity()
        self._updateBottomFriction()
        self._updateBaroclinicity()
        if doStabParams:
            self._updateStabilizationParams()


class coupledSSPRKSync(coupledTimeIntegrator):
    """
    Split-explicit SSPRK time integrator that sub-iterates 2D mode.
    3D time step is computed based on horizontal velocity. 2D mode is sub-iterated and hence has
    very little numerical diffusion.
    """
    def __init__(self, solver):
        super(coupledSSPRKSync, self).__init__(solver, solver.options,
                                               solver.fields)
        self._initialized = False
        self.timeStepper2d = timeIntegrator.SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()
        self.sol2d_n = Function(fs, name='sol2dtmp')

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.options.solveSalt:
            self.timeStepper_salt_3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.options.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.CrankNicolson(
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
        printInfo('dt_frac ' + str(self.dt_frac))
        printInfo('start_frac ' + str(self.start_frac))
        printInfo('stage_w ' + str(self.stage_w))

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.fields.solution2d)
        self.timeStepper_mom3d.initialize(self.fields.uv_3d)
        if self.options.solveSalt:
            self.timeStepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.fields.uv_3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            M = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/M
            printInfo('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, M, f))
            self.M.append(M)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        self.sol2d_n.assign(sol2d)  # keep copy of elev_n
        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if self.options.solveSalt:
                    self.timeStepper_salt_3d.solveStage(k, t, self.solver.dt,
                                                       self.fields.salt_3d,
                                                       updateForcings3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.fields.salt_3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, self.solver.dt,
                                                  self.fields.uv_3d)
            with timed_region('mode2d'):
                t_rhs = t + self.start_frac[k]*self.solver.dt
                dt_2d = self.dt_2d[k]
                # initialize
                w = self.stage_w[k]
                sol2d.assign(w*self.sol2d_n + (1.0-w)*sol2d)

                # advance fields from T_{n} to T{n+1}
                for i in range(self.M[k]):
                    self.timeStepper2d.advance(t_rhs + i*dt_2d, dt_2d, sol2d,
                                               updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            self._updateAllDependencies(t, doVertDiffusion=lastStep,
                                        do2DCoupling=lastStep,
                                        doALEUpdate=lastStep,
                                        doStabParams=lastStep,
                                        doTurbulence=lastStep)


class coupledSSPIMEX(coupledTimeIntegrator):
    """
    Solves coupled 3D equations with SSP IMEX scheme by [1], method (17).

    With this scheme all the equations can be advanced in time synchronously.

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, solver):
        super(coupledSSPIMEX, self).__init__(solver, solver.options,
                                             solver.fields)
        self._initialized = False
        # for 2d shallow water eqns
        sp_impl = {
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            }
        sp_expl = {
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            }
        fs_2d = solver.eq_sw.solution.function_space()
        self.solution2d_old = Function(fs_2d, name='old_sol_2d')
        self.timeStepper2d = timeIntegrator.SSPIMEX(
            solver.eq_sw, solver.dt,
            solver_parameters=sp_expl,
            solver_parameters_dirk=sp_impl,
            solution=self.solution2d_old)
        # for 3D equations
        sp_impl = {
            'ksp_type': 'gmres',
            }
        sp_expl = {
            'ksp_type': 'gmres',
            }
        fs_mom = solver.eq_momentum.solution.function_space()
        self.uv_3d_old = Function(fs_mom, name='old_sol_mom')
        self.timeStepper_mom3d = timeIntegrator.SSPIMEX(
            solver.eq_momentum, solver.dt,
            solver_parameters=sp_expl,
            solver_parameters_dirk=sp_impl,
            solution=self.uv_3d_old)
        if self.solver.options.solveSalt:
            fs = solver.eq_salt.solution.function_space()
            self.salt_3d_old = Function(fs, name='old_sol_salt')
            self.timeStepper_salt_3d = timeIntegrator.SSPIMEX(
                solver.eq_salt, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.salt_3d_old)
        if self.solver.options.solveVertDiffusion:
            raise Exception('vert mom eq should not exist for this time integrator')
            self.timeStepper_vmom3d = timeIntegrator.SSPIMEX(
                solver.eq_vertmomentum, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl)
        if self.solver.options.useTurbulence:
            fs = solver.eq_tke_diff.solution.function_space()
            self.tke_3d_old = Function(fs, name='old_sol_tke')
            self.timeStepper_tke_3d = timeIntegrator.SSPIMEX(
                solver.eq_tke_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.tke_3d_old)
            self.psi_3d_old = Function(fs, name='old_sol_psi')
            self.timeStepper_psi_3d = timeIntegrator.SSPIMEX(
                solver.eq_psi_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.psi_3d_old)
        self.nStages = self.timeStepper_mom3d.nStages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.fields.solution2d)
        self.timeStepper_mom3d.initialize(self.fields.uv_3d)
        if self.options.solveSalt:
            self.timeStepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()

        for k in range(self.nStages):
            lastStep = k == self.nStages - 1
            with timed_region('saltEq'):
                if self.options.solveSalt:
                    self.timeStepper_salt_3d.solveStage(k, t, self.solver.dt, self.fields.salt_3d,
                                                       updateForcings3d)
                    if self.options.useLimiterForTracers and lastStep:
                        self.solver.tracerLimiter.apply(self.fields.salt_3d)
            with timed_region('turbulenceAdvection'):
                if self.options.useTurbulenceAdvection:
                    # explicit advection
                    self.timeStepper_tkeAdvEq.solveStage(k, t, self.solver.dt, self.fields.tke_3d)
                    self.timeStepper_psiAdvEq.solveStage(k, t, self.solver.dt, self.fields.psi_3d)
                    if self.options.useLimiterForTracers and lastStep:
                        self.solver.tracerLimiter.apply(self.fields.tke_3d)
                        self.solver.tracerLimiter.apply(self.fields.psi_3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, self.solver.dt, self.fields.uv_3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, self.solver.dt, self.fields.solution2d,
                                              updateForcings)
            with timed_region('turbulence'):
                if self.options.useTurbulence:
                    # NOTE psi must be solved first as it depends on tke
                    self.timeStepper_psi_3d.solveStage(k, t, self.solver.dt, self.fields.psi_3d)
                    self.timeStepper_tke_3d.solveStage(k, t, self.solver.dt, self.fields.tke_3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.fields.tke_3d)
                        self.solver.tracerLimiter.apply(self.fields.psi_3d)
                    self.solver.glsModel.postprocess()
                    self.solver.glsModel.preprocess()  # for next iteration
            self._updateAllDependencies(t, doVertDiffusion=False,
                                        do2DCoupling=lastStep,
                                        doALEUpdate=lastStep,
                                        doStabParams=lastStep)


class coupledSSPRKSemiImplicit(coupledTimeIntegrator):
    """
    Solves coupled equations with simultaneous SSPRK33 stages, where 2d gravity
    waves are solved semi-implicitly. This saves CPU cos diffuses gravity waves.
    """
    def __init__(self, solver):
        super(coupledSSPRKSemiImplicit, self).__init__(solver,
                                                       solver.options,
                                                       solver.fields)
        self._initialized = False
        self.timeStepper2d = timeIntegrator.SSPRK33StageSemiImplicit(
            solver.eq_sw, solver.dt,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.options.solveSalt:
            self.timeStepper_salt_3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        vdiff_sp = {
            'ksp_type': 'gmres',
            'pc_type': 'ilu',
            #'snes_rtol': 1.0e-18,
            #'ksp_rtol': 1.0e-22,
            }
        if self.solver.options.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.DIRK_LSPUM2(
                solver.eq_vertmomentum, solver.dt, solver_parameters=vdiff_sp)
        if self.solver.options.useTurbulence:
            self.timeStepper_tke_3d = timeIntegrator.DIRK_LSPUM2(
                solver.eq_tke_diff, solver.dt, solver_parameters=vdiff_sp)
            self.timeStepper_psi_3d = timeIntegrator.DIRK_LSPUM2(
                solver.eq_psi_diff, solver.dt, solver_parameters=vdiff_sp)

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
        printInfo('dt_frac ' + str(self.dt_frac))
        printInfo('start_frac ' + str(self.start_frac))
        printInfo('stage_w ' + str(self.stage_w))

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.fields.solution2d)
        self.timeStepper_mom3d.initialize(self.fields.uv_3d)
        if self.options.solveSalt:
            self.timeStepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.fields.uv_3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            M = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/M
            printInfo('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, M, f))
            self.M.append(M)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        sol2d = self.timeStepper2d.equation.solution

        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if self.options.solveSalt:
                    self.timeStepper_salt_3d.solveStage(k, t, self.solver.dt,
                                                       self.fields.salt_3d,
                                                       updateForcings3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.fields.salt_3d)
            with timed_region('turbulenceAdvection'):
                if self.options.useTurbulenceAdvection:
                    # explicit advection
                    self.timeStepper_tkeAdvEq.solveStage(k, t, self.solver.dt,
                                                         self.solver.tke_3d)
                    self.timeStepper_psiAdvEq.solveStage(k, t, self.solver.dt,
                                                         self.solver.psi_3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.solver.tke_3d)
                        self.solver.tracerLimiter.apply(self.solver.psi_3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, self.solver.dt,
                                                  self.fields.uv_3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, self.solver.dt, sol2d,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            self._updateAllDependencies(t, doVertDiffusion=lastStep,
                                        do2DCoupling=lastStep,
                                        doALEUpdate=lastStep,
                                        doStabParams=lastStep,
                                        doTurbulence=lastStep)


class coupledSSPRKSingleMode(coupledTimeIntegrator):
    """
    Split-explicit SSPRK33 solver without mode-splitting.
    Both 2D and 3D modes are advanced with the same time step, computed based on 2D gravity
    wave speed. This time integrator is therefore expensive and should be only used for debugging etc.
    """
    def __init__(self, solver):
        super(coupledSSPRKSingleMode, self).__init__(solver,
                                                     solver.options,
                                                     solver.fields)
        self._initialized = False
        self.timeStepper2d = timeIntegrator.SSPRK33Stage(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum,
            solver.dt_2d)
        if self.solver.options.solveSalt:
            self.timeStepper_salt_3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt_2d)
        if self.solver.options.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.CrankNicolson(
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
        self.timeStepper2d.initialize(self.fields.solution2d.split()[1])
        self.timeStepper_mom3d.initialize(self.fields.uv_3d)
        if self.options.solveSalt:
            self.timeStepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def _update2DCoupling(self):
        """Overloaded coupling function"""
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(self.fields.uv_3d, self.fields.uv_dav_3d,
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=self.fields.bathymetry_3d)
            copy3dFieldTo2d(self.fields.uv_dav_3d, self.fields.uv_dav_2d,
                            useBottomValue=False,
                            elemHeight=self.fields.v_elem_size_2d)
            self.fields.uv_2d.assign(self.fields.uv_dav_2d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        for k in range(self.timeStepper2d.nstages):
            with timed_region('saltEq'):
                if self.options.solveSalt:
                    self.timeStepper_salt_3d.solveStage(k, t, self.solver.dt_2d,
                                                       self.fields.salt_3d,
                                                       updateForcings3d)
                    if self.options.useLimiterForTracers:
                        self.solver.tracerLimiter.apply(self.fields.salt_3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, self.solver.dt_2d,
                                                  self.fields.uv_3d)
            with timed_region('mode2d'):
                uv, elev = self.fields.solution2d.split()
                self.timeStepper2d.solveStage(k, t, self.solver.dt_2d, elev,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            self._updateAllDependencies(t, doVertDiffusion=lastStep,
                                        do2DCoupling=True,
                                        doALEUpdate=lastStep,
                                        doStabParams=lastStep,
                                        doTurbulence=lastStep)


class coupledSSPRK(coupledTimeIntegrator):
    """
    Split-explicit time integration that uses SSPRK for both 2d and 3d modes.
    2D mode is solved only once from t_{n} to t_{n+2} using shorter time steps.
    To couple with the 3D mode, 2D variables are averaged in time to get values at t_{n+1} and t_{n+1/2}. 
    """
    def __init__(self, solver):
        super(coupledSSPRK, self).__init__(solver,
                                           solver.options,
                                           solver.fields)
        self._initialized = False
        subIterator = SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        self.timeStepper2d = timeIntegrator.macroTimeStepIntegrator(
            subIterator,
            solver.M_modesplit,
            restartFromAv=True)

        self.timeStepper_mom3d = timeIntegrator.SSPRK33(
            solver.eq_momentum, solver.dt,
            funcs_nplushalf={'eta': solver.elev_3d_nplushalf})
        if self.solver.options.solveSalt:
            self.timeStepper_salt_3d = timeIntegrator.SSPRK33(
                solver.eq_salt,
                solver.dt)
        if self.solver.options.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.fields.solution2d)
        self.timeStepper_mom3d.initialize(self.fields.uv_3d)
        if self.options.options.solveSalt:
            self.timeStepper_salt_3d.initialize(self.fields.salt_3d)
        if self.options.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.fields.uv_3d)
        self._initialized = True

    def _update2DCoupling(self):
        """Overloaded coupling function"""
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(self.fields.uv_3d, self.fields.uv_dav_3d,
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=self.fields.bathymetry_3d)
            copy3dFieldTo2d(self.fields.uv_dav_3d, self.fields.uv_dav_2d,
                            useBottomValue=False)
            copy2dFieldTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d)
            # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
            uv_2d_start = self.timeStepper2d.solution_start.split()[0]
            uv_2d_start.assign(self.fields.uv_dav_2d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        # SSPRK33 time integration loop
        with timed_region('mode2d'):
            self.timeStepper2d.advance(t, self.solver.dt_2d,
                                       self.fields.solution2d,
                                       updateForcings)
        with timed_region('aux_elev_3d'):
            elev_n = self.fields.solution2d.split()[1]
            copy2dFieldTo3d(elev_n, self.fields.elev_3d)  # at t_{n+1}
            elev_nph = self.timeStepper2d.solution_nplushalf.split()[1]
            copy2dFieldTo3d(elev_nph, self.fields.elev_3d_nplushalf)  # at t_{n+1/2}
        self._updateMovingMesh()
        self._updateBottomFriction()
        self._updateBaroclinicity()
        with timed_region('momentumEq'):
            self.timeStepper_mom3d.advance(t, self.solver.dt,
                                           self.fields.uv_3d,
                                           updateForcings3d)
        with timed_region('vert_diffusion'):
            if self.options.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, self.solver.dt,
                                                self.fields.uv_3d, None)
        self._updateVerticalVelocity()
        self._updateMeshVelocity()
        with timed_region('saltEq'):
            if self.options.solveSalt:
                self.timeStepper_salt_3d.advance(t, self.solver.dt,
                                                self.fields.salt_3d,
                                                updateForcings3d)
        self._update2DCoupling()
