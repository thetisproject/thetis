"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from utility import *
import timeIntegrator

# TODO write all common operations to helper functions _computeFriction etc


class coupledSSPRKSync(timeIntegrator.timeIntegrator):
    """
    Split-explicit SSPRK time integrator that sub-iterates 2D mode.
    3D time step is computed based on horizontal velocity. 2D mode is sub-iterated and hence has
    very little numerical diffusion.
    """
    def __init__(self, solver):
        self._initialized = False
        self.solver = solver
        self.timeStepper2d = timeIntegrator.SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()
        self.sol2d_n = Function(fs, name='sol2dtmp')

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.options.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
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
        f = self.solver.functions
        o = self.solver.options
        self.timeStepper2d.initialize(f.solution2d)
        self.timeStepper_mom3d.initialize(f.uv3d)
        if o.solveSalt:
            self.timeStepper_salt3d.initialize(f.salt3d)
        if o.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(f.uv3d)

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
        s = self.solver
        o = self.solver.options
        f = self.solver.functions
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_elev3d'):
                elev = sol2d.split()[1]
                copy2dFieldTo3d(elev, f.elev3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, f.elev3d, f.bathymetry3d,
                        f.z_coord3d, f.z_coord_ref3d)
                    computeElemHeight(f.z_coord3d, f.vElemSize3d)
                    copy3dFieldTo2d(f.vElemSize3d, f.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if o.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if o.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and o.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt, f.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(f.uv3d, f.uvDav3d,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=f.bathymetry3d)
                    copy3dFieldTo2d(f.uvDav3d, f.uvDav2d,
                                    useBottomValue=False, elemHeight=f.vElemSize2d)
                    copy2dFieldTo3d(f.uvDav2d, f.uvDav3d, elemHeight=f.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(f.uvDav2d)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    #s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    #f.uv3d -= f.uv3d_tmp
                    #s.uv2d_to_DAV_projector.project()  # uv2d to uvDav2d
                    #copy2dFieldTo3d(f.uvDav2d, f.uvDav3d,
                                    #elemHeight=f.vElemSize3d)
                    #s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    #f.uv3d += f.uv3d_tmp
                    f.uv3d -= f.uvDav3d
                    copy2dFieldTo3d(sol2d.split()[0], f.uvDav3d,
                                    elemHeight=f.vElemSize3d)
                    f.uv3d += f.uvDav3d

            with timed_region('continuityEq'):
                computeVertVelocity(f.w3d, f.uv3d, f.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh:
                    computeMeshVelocity(
                        f.elev3d, f.uv3d, f.w3d,
                        f.w_mesh3d, f.w_mesh_surf3d,
                        f.w_mesh_surf2d,
                        f.dw_mesh_dz_3d, f.bathymetry3d,
                        f.z_coord_ref3d)
            with timed_region('aux_friction'):
                if o.useBottomFriction:
                    s.uvP1_projector.project()
                    computeBottomFriction(
                        f.uv3d_P1, f.uv_bottom2d,
                        f.uv_bottom3d, f.z_coord3d,
                        f.z_bottom2d, f.z_bottom3d,
                        f.bathymetry2d, f.bottom_drag2d,
                        f.bottom_drag3d,
                        f.vElemSize2d, f.vElemSize3d)
                if o.useParabolicViscosity:
                    computeParabolicViscosity(
                        f.uv_bottom3d, f.bottom_drag3d,
                        f.bathymetry3d,
                        f.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if o.baroclinic:
                    computeBaroclinicHead(f.salt3d, f.baroHead3d,
                                          f.baroHead2d, f.baroHeadInt3d,
                                          f.bathymetry3d)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(f.uv3d_mag, u=f.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if o.smagorinskyFactor is not None:
                        smagorinskyViscosity(f.uv3d_P1, f.smag_viscosity,
                                             f.smagorinskyFactor, f.hElemSize3d)
                    if o.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(f.saltJumpDiffFactor, f.salt3d,
                                                    f.saltJumpDiff, f.hElemSize3d,
                                                    f.uv3d_mag, o.saltRange,
                                                    f.maxHDiffusivity)

        self.sol2d_n.assign(sol2d)  # keep copy of elev_n
        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if o.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, f.salt3d,
                                                       updateForcings3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(f.salt3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, f.uv3d)
            with timed_region('mode2d'):
                t_rhs = t + self.start_frac[k]*s.dt
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
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=lastStep,
                               doStabParams=lastStep)


class coupledSSPIMEX(timeIntegrator.timeIntegrator):
    """
    Solves coupled 3D equations with SSP IMEX scheme by [1], method (17).

    With this scheme all the equations can be advanced in time synchronously.

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, solver):
        self._initialized = False
        self.solver = solver
        self.functions = self.solver.functions
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
        self.uv3d_old = Function(fs_mom, name='old_sol_mom')
        self.timeStepper_mom3d = timeIntegrator.SSPIMEX(
            solver.eq_momentum, solver.dt,
            solver_parameters=sp_expl,
            solver_parameters_dirk=sp_impl,
            solution=self.uv3d_old)
        if self.solver.options.solveSalt:
            fs = solver.eq_salt.solution.function_space()
            self.salt3d_old = Function(fs, name='old_sol_salt')
            self.timeStepper_salt3d = timeIntegrator.SSPIMEX(
                solver.eq_salt, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.salt3d_old)
        if self.solver.options.solveVertDiffusion:
            raise Exception('vert mom eq should not exist for this time integrator')
            self.timeStepper_vmom3d = timeIntegrator.SSPIMEX(
                solver.eq_vertmomentum, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl)
        if self.solver.options.useTurbulence:
            fs = solver.eq_tke_diff.solution.function_space()
            self.tke3d_old = Function(fs, name='old_sol_tke')
            self.timeStepper_tke3d = timeIntegrator.SSPIMEX(
                solver.eq_tke_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.tke3d_old)
            self.psi3d_old = Function(fs, name='old_sol_psi')
            self.timeStepper_psi3d = timeIntegrator.SSPIMEX(
                solver.eq_psi_diff, solver.dt,
                solver_parameters=sp_expl,
                solver_parameters_dirk=sp_impl,
                solution=self.psi3d_old)
        self.nStages = self.timeStepper_mom3d.nStages

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        f = self.solver.functions
        o = self.solver.options
        self.timeStepper2d.initialize(f.solution2d)
        self.timeStepper_mom3d.initialize(f.uv3d)
        if o.solveSalt:
            self.timeStepper_salt3d.initialize(f.salt3d)
        if o.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(f.uv3d)
        self._initialized = True

    def _updateDependencies(self, do2DCoupling=False,
                            doVertDiffusion=False,
                            doALEUpdate=False,
                            doStabParams=False,
                            doTurbulence=False):
        """Updates all dependencies of the primary variables"""
        s = self.solver
        o = self.solver.options
        f = self.solver.functions
        sol2d = self.timeStepper2d.equation.solution
        with timed_region('aux_elev3d'):
            elev = sol2d.split()[1]
            copy2dFieldTo3d(elev, f.elev3d)  # at t_{n+1}
            s.elev3d_to_CG_projector.project()
        with timed_region('aux_mesh_ale'):
            if o.useALEMovingMesh and doALEUpdate:
                updateCoordinates(
                    s.mesh, f.elev3dCG, f.bathymetry3d,
                    f.z_coord3d, f.z_coord_ref3d)
                computeElemHeight(f.z_coord3d, f.vElemSize3d)
                copy3dFieldTo2d(f.vElemSize3d, f.vElemSize2d)
                # need to destroy all cached solvers!
                linProblemCache.clear()
                self.timeStepper_mom3d.updateSolver()
                if o.solveSalt:
                    self.timeStepper_salt3d.updateSolver()
                if o.solveVertDiffusion:
                    self.timeStepper_vmom3d.updateSolver()
        with timed_region('vert_diffusion'):
            if doVertDiffusion and o.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, s.dt, f.uv3d)
        with timed_region('aux_mom_coupling'):
            if do2DCoupling:
                bndValue = Constant((0.0, 0.0, 0.0))
                computeVerticalIntegral(f.uv3d, f.uvDav3d,
                                        bottomToTop=True, bndValue=bndValue,
                                        average=True,
                                        bathymetry=f.bathymetry3d)
                copy3dFieldTo2d(f.uvDav3d, f.uvDav2d,
                                useBottomValue=False,
                                elemHeight=f.vElemSize2d)
                copy2dFieldTo3d(f.uvDav2d, f.uvDav3d,
                                elemHeight=f.vElemSize3d)
                # 2d-3d coupling: restart 2d mode from depth ave uv3d
                # NOTE unstable!
                #uv2d_start = sol2d.split()[0]
                #uv2d_start.assign(f.uvDav2d)
                # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                #s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                #f.uv3d -= f.uv3d_tmp
                #s.uv2d_to_DAV_projector.project()  # uv2d to uvDav2d
                #copy2dFieldTo3d(f.uvDav2d, f.uvDav3d,
                                #elemHeight=f.vElemSize3d)
                #f.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                #f.uv3d += f.uv3d_tmp
                f.uv3d -= f.uvDav3d
                copy2dFieldTo3d(sol2d.split()[0], f.uvDav3d,
                                elemHeight=f.vElemSize3d)
                f.uv3d += f.uvDav3d
        with timed_region('continuityEq'):
            computeVertVelocity(f.w3d, f.uv3d, f.bathymetry3d,
                                s.eq_momentum.boundary_markers,
                                s.eq_momentum.bnd_functions)
        with timed_region('aux_mesh_ale'):
            if o.useALEMovingMesh:
                computeMeshVelocity(
                    f.elev3d, f.uv3d, f.w3d,
                    f.w_mesh3d, f.w_mesh_surf3d,
                    f.w_mesh_surf2d,
                    f.dw_mesh_dz_3d, f.bathymetry3d,
                    f.z_coord_ref3d)
        with timed_region('aux_friction'):
            if o.useBottomFriction and doVertDiffusion:
                s.uvP1_projector.project()
                computeBottomFriction(
                    f.uv3d_P1, f.uv_bottom2d,
                    f.uv_bottom3d, f.z_coord3d,
                    f.z_bottom2d, f.z_bottom3d,
                    f.bathymetry2d, f.bottom_drag2d,
                    f.bottom_drag3d,
                    f.vElemSize2d, f.vElemSize3d)
            if o.useParabolicViscosity:
                computeParabolicViscosity(
                    f.uv_bottom3d, f.bottom_drag3d,
                    f.bathymetry3d,
                    f.parabViscosity_v)
        with timed_region('aux_barolinicity'):
            if o.baroclinic:
                computeBaroclinicHead(f.salt3d, f.baroHead3d,
                                      f.baroHead2d, f.baroHeadInt3d,
                                      f.bathymetry3d)
        with timed_region('aux_stabilization'):
            if doStabParams:
                # update velocity magnitude
                computeVelMagnitude(f.uv3d_mag, u=f.uv3d)
                # update P1 velocity field
                s.uvP1_projector.project()
                if o.smagorinskyFactor is not None:
                    smagorinskyViscosity(f.uv3d_P1, f.smag_viscosity,
                                         f.smagorinskyFactor, f.hElemSize3d)
                if o.saltJumpDiffFactor is not None:
                    computeHorizJumpDiffusivity(f.saltJumpDiffFactor, f.salt3d,
                                                f.saltJumpDiff, f.hElemSize3d,
                                                f.uv3d_mag, o.saltRange,
                                                f.maxHDiffusivity)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        s = self.solver
        o = self.solver.options
        f = self.functions

        for k in range(self.nStages):
            lastStep = k == self.nStages - 1
            with timed_region('saltEq'):
                if o.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, f.salt3d,
                                                       updateForcings3d)
                    if o.useLimiterForTracers and lastStep:
                        s.tracerLimiter.apply(f.salt3d)
            with timed_region('turbulenceAdvection'):
                if o.useTurbulenceAdvection:
                    # explicit advection
                    self.timeStepper_tkeAdvEq.solveStage(k, t, s.dt, f.tke3d)
                    self.timeStepper_psiAdvEq.solveStage(k, t, s.dt, f.psi3d)
                    if o.useLimiterForTracers and lastStep:
                        s.tracerLimiter.apply(f.tke3d)
                        s.tracerLimiter.apply(f.psi3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, f.uv3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, s.dt, f.solution2d,
                                              updateForcings)
            with timed_region('turbulence'):
                if o.useTurbulence:
                    # NOTE psi must be solved first as it depends on tke
                    self.timeStepper_psi3d.solveStage(k, t, s.dt, f.psi3d)
                    self.timeStepper_tke3d.solveStage(k, t, s.dt, f.tke3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(f.tke3d)
                        s.tracerLimiter.apply(f.psi3d)
                    s.glsModel.postprocess()
                    s.glsModel.preprocess() # for next iteration
            self._updateDependencies(doVertDiffusion=False,
                                     do2DCoupling=lastStep,
                                     doALEUpdate=lastStep,
                                     doStabParams=lastStep)


class coupledSSPRKSemiImplicit(timeIntegrator.timeIntegrator):
    """
    Solves coupled equations with simultaneous SSPRK33 stages, where 2d gravity
    waves are solved semi-implicitly. This saves CPU cos diffuses gravity waves.
    """
    def __init__(self, solver):
        self._initialized = False
        self.solver = solver
        self.timeStepper2d = timeIntegrator.SSPRK33StageSemiImplicit(
            solver.eq_sw, solver.dt,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.options.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
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
            self.timeStepper_tke3d = timeIntegrator.DIRK_LSPUM2(
                solver.eq_tke_diff, solver.dt, solver_parameters=vdiff_sp)
            self.timeStepper_psi3d = timeIntegrator.DIRK_LSPUM2(
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
        f = self.solver.functions
        o = self.solver.options
        self.timeStepper2d.initialize(f.solution2d)
        self.timeStepper_mom3d.initialize(f.uv3d)
        if o.solveSalt:
            self.timeStepper_salt3d.initialize(f.salt3d)
        if o.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(f.uv3d)

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
        s = self.solver
        o = self.solver.options
        f = self.solver.functions
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doALEUpdate=False,
                               doStabParams=False,
                               doTurbulence=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_elev3d'):
                elev = sol2d.split()[1]
                copy2dFieldTo3d(elev, f.elev3d)  # at t_{n+1}
                s.elev3d_to_CG_projector.project()
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh and doALEUpdate:
                    updateCoordinates(
                        s.mesh, f.elev3dCG, f.bathymetry3d,
                        f.z_coord3d, f.z_coord_ref3d)
                    computeElemHeight(f.z_coord3d, f.vElemSize3d)
                    copy3dFieldTo2d(f.vElemSize3d, f.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if o.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if o.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and o.solveVertDiffusion:
                    self.timeStepper_vmom3d.advance(t, s.dt, f.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(f.uv3d, f.uvDav3d,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=f.bathymetry3d)
                    copy3dFieldTo2d(f.uvDav3d, f.uvDav2d,
                                    useBottomValue=False,
                                    elemHeight=f.vElemSize2d)
                    copy2dFieldTo3d(f.uvDav2d, f.uvDav3d,
                                    elemHeight=f.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(f.uvDav2d)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    #s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    #f.uv3d -= f.uv3d_tmp
                    #s.uv2d_to_DAV_projector.project()  # uv2d to uvDav2d
                    #copy2dFieldTo3d(f.uvDav2d, f.uvDav3d,
                                    #elemHeight=f.vElemSize3d)
                    #s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    #f.uv3d += f.uv3d_tmp
                    f.uv3d -= f.uvDav3d
                    copy2dFieldTo3d(sol2d.split()[0], f.uvDav3d,
                                    elemHeight=f.vElemSize3d)
                    f.uv3d += f.uvDav3d
            with timed_region('continuityEq'):
                computeVertVelocity(f.w3d, f.uv3d, f.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('turbulence'):
                if o.useTurbulence and doTurbulence:
                    s.glsModel.preprocess()
                    # NOTE psi must be solved first as it depends on tke
                    self.timeStepper_psi3d.advance(t, s.dt, s.psi3d)
                    self.timeStepper_tke3d.advance(t, s.dt, s.tke3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(s.tke3d)
                        s.tracerLimiter.apply(s.psi3d)
                    s.glsModel.postprocess()
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh:
                    computeMeshVelocity(
                        f.elev3d, f.uv3d, f.w3d,
                        f.w_mesh3d, f.w_mesh_surf3d,
                        f.w_mesh_surf2d,
                        f.dw_mesh_dz_3d, f.bathymetry3d,
                        f.z_coord_ref3d)
            with timed_region('aux_friction'):
                if o.useBottomFriction and doVertDiffusion:
                    s.uvP1_projector.project()
                    computeBottomFriction(
                        f.uv3d_P1, f.uv_bottom2d,
                        f.uv_bottom3d, f.z_coord3d,
                        f.z_bottom2d, f.z_bottom3d,
                        f.bathymetry2d, f.bottom_drag2d,
                        f.bottom_drag3d,
                        f.vElemSize2d, f.vElemSize3d)
                if o.useParabolicViscosity:
                    computeParabolicViscosity(
                        f.uv_bottom3d, f.bottom_drag3d,
                        f.bathymetry3d,
                        f.parabViscosity_v)
            with timed_region('aux_barolinicity'):
                if o.baroclinic:
                    computeBaroclinicHead(f.salt3d, f.baroHead3d,
                                          f.baroHead2d, f.baroHeadInt3d,
                                          f.bathymetry3d)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(f.uv3d_mag, u=f.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if o.smagorinskyFactor is not None:
                        smagorinskyViscosity(f.uv3d_P1, f.smag_viscosity,
                                             f.smagorinskyFactor, f.hElemSize3d)
                    if o.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(f.saltJumpDiffFactor, f.salt3d,
                                                    f.saltJumpDiff, f.hElemSize3d,
                                                    f.uv3d_mag, o.saltRange,
                                                    f.maxHDiffusivity)

        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if o.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, f.salt3d,
                                                       updateForcings3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(f.salt3d)
            with timed_region('turbulenceAdvection'):
                if o.useTurbulenceAdvection:
                    # explicit advection
                    self.timeStepper_tkeAdvEq.solveStage(k, t, s.dt, s.tke3d)
                    self.timeStepper_psiAdvEq.solveStage(k, t, s.dt, s.psi3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(s.tke3d)
                        s.tracerLimiter.apply(s.psi3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, f.uv3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, s.dt, sol2d,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=lastStep,
                               doALEUpdate=lastStep,
                               doStabParams=lastStep,
                               doTurbulence=lastStep)


class coupledSSPRKSingleMode(timeIntegrator.timeIntegrator):
    """
    Split-explicit SSPRK33 solver without mode-splitting.
    Both 2D and 3D modes are advanced with the same time step, computed based on 2D gravity
    wave speed. This time integrator is therefore expensive and should be only used for debugging etc.
    """
    def __init__(self, solver):
        self.solver = solver
        self.timeStepper2d = timeIntegrator.SSPRK33Stage(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)

        self.timeStepper_mom3d = timeIntegrator.SSPRK33Stage(
            solver.eq_momentum,
            solver.dt_2d)
        if self.solver.options.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
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
        f = self.solver.functions
        o = self.solver.options
        self.timeStepper2d.initialize(f.solution2d.split()[1])
        self.timeStepper_mom3d.initialize(f.uv3d)
        if o.solveSalt:
            self.timeStepper_salt3d.initialize(f.salt3d)
        if o.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(f.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        o = self.solver.options
        f = self.solver.functions

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_elev3d'):
                copy2dFieldTo3d(f.elev2d, f.elev3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, f.elev3d, f.bathymetry3d,
                        f.z_coord3d, f.z_coord_ref3d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if o.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if o.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and o.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt_2d, f.uv3d)
            with timed_region('continuityEq'):
                computeVertVelocity(f.w3d, f.uv3d, f.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if o.useALEMovingMesh:
                    computeMeshVelocity(
                        f.elev3d, f.uv3d, f.w3d,
                        f.w_mesh3d, f.w_mesh_surf3d,
                        f.w_mesh_surf2d,
                        f.dw_mesh_dz_3d, f.bathymetry3d,
                        f.z_coord_ref3d)
            with timed_region('aux_friction'):
                if o.useBottomFriction:
                    s.uvP1_projector.project()
                    computeBottomFriction(
                        f.uv3d_P1, f.uv_bottom2d,
                        f.uv_bottom3d, f.z_coord3d,
                        f.z_bottom2d, f.z_bottom3d,
                        f.bathymetry2d, f.bottom_drag2d,
                        f.bottom_drag3d,
                        f.vElemSize2d, f.vElemSize3d)
                if o.useParabolicViscosity:
                    computeParabolicViscosity(
                        f.uv_bottom3d, f.bottom_drag3d,
                        f.bathymetry3d,
                        f.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if o.baroclinic:
                    computeBaroclinicHead(f.salt3d, f.baroHead3d,
                                          f.baroHead2d, f.baroHeadInt3d,
                                          f.bathymetry3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(f.uv3d, f.uvDav3d,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=f.bathymetry3d)
                    copy3dFieldTo2d(f.uvDav3d, f.uvDav2d,
                                    useBottomValue=False, elemHeight=f.vElemSize2d)
                    f.uv2d.assign(f.uvDav2d)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(f.uv3d_mag, u=f.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if o.smagorinskyFactor is not None:
                        smagorinskyViscosity(f.uv3d_P1, f.smag_viscosity,
                                             f.smagorinskyFactor, f.hElemSize3d)
                    if o.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(f.saltJumpDiffFactor, f.salt3d,
                                                    f.saltJumpDiff, f.hElemSize3d,
                                                    f.uv3d_mag, o.saltRange,
                                                    f.maxHDiffusivity)

        for k in range(self.timeStepper2d.nstages):
            with timed_region('saltEq'):
                if o.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt_2d, f.salt3d,
                                                       updateForcings3d)
                    if o.useLimiterForTracers:
                        s.tracerLimiter.apply(f.salt3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt_2d, f.uv3d)
            with timed_region('mode2d'):
                uv, elev = f.solution2d.split()
                self.timeStepper2d.solveStage(k, t, s.dt_2d, elev,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=True,
                               doStabParams=lastStep)


class coupledSSPRK(timeIntegrator.timeIntegrator):
    """
    Split-explicit time integration that uses SSPRK for both 2d and 3d modes.
    2D mode is solved only once from t_{n} to t_{n+2} using shorter time steps.
    To couple with the 3D mode, 2D variables are averaged in time to get values at t_{n+1} and t_{n+1/2}. 
    """
    def __init__(self, solver):
        self.solver = solver
        subIterator = SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        self.timeStepper2d = timeIntegrator.macroTimeStepIntegrator(
            subIterator,
            solver.M_modesplit,
            restartFromAv=True)

        self.timeStepper_mom3d = timeIntegrator.SSPRK33(
            solver.eq_momentum, solver.dt,
            funcs_nplushalf={'eta': solver.elev3d_nplushalf})
        if self.solver.options.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33(
                solver.eq_salt,
                solver.dt)
        if self.solver.options.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        f = self.solver.functions
        o = self.solver.options
        self.timeStepper2d.initialize(f.solution2d)
        self.timeStepper_mom3d.initialize(f.uv3d)
        if o.options.solveSalt:
            self.timeStepper_salt3d.initialize(f.salt3d)
        if o.options.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(f.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        o = self.solver.options
        f = self.solver.functions
        # SSPRK33 time integration loop
        with timed_region('mode2d'):
            self.timeStepper2d.advance(t, s.dt_2d, f.solution2d,
                                       updateForcings)
        with timed_region('aux_elev3d'):
            elev_n = f.solution2d.split()[1]
            copy2dFieldTo3d(elev_n, f.elev3d)  # at t_{n+1}
            elev_nph = self.timeStepper2d.solution_nplushalf.split()[1]
            copy2dFieldTo3d(elev_nph, f.elev3d_nplushalf)  # at t_{n+1/2}
        with timed_region('aux_mesh_ale'):
            if o.useALEMovingMesh:
                updateCoordinates(
                    s.mesh, f.elev3d, f.bathymetry3d,
                    f.z_coord3d, f.z_coord_ref3d)
        with timed_region('aux_friction'):
            if o.useBottomFriction:
                s.uvP1_projector.project()
                computeBottomFriction(
                    f.uv3d_P1, f.uv_bottom2d,
                    f.uv_bottom3d, f.z_coord3d,
                    f.z_bottom2d, f.z_bottom3d,
                    f.bathymetry2d, f.bottom_drag2d,
                    f.bottom_drag3d,
                    f.vElemSize2d, f.vElemSize3d)
            if o.useParabolicViscosity:
                computeParabolicViscosity(
                    f.uv_bottom3d, f.bottom_drag3d,
                    f.bathymetry3d,
                    f.viscosity_v3d)
        with timed_region('aux_barolinicity'):
            if o.baroclinic:
                computeBaroclinicHead(f.salt3d, f.baroHead3d,
                                      f.baroHead2d, f.baroHeadInt3d,
                                      f.bathymetry3d)

        with timed_region('momentumEq'):
            self.timeStepper_mom3d.advance(t, s.dt, f.uv3d,
                                           updateForcings3d)
        with timed_region('vert_diffusion'):
            if o.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, s.dt, f.uv3d, None)
        with timed_region('continuityEq'):
            computeVertVelocity(f.w3d, f.uv3d, f.bathymetry3d,
                                s.eq_momentum.boundary_markers,
                                s.eq_momentum.bnd_functions)
        with timed_region('aux_mesh_ale'):
            if o.useALEMovingMesh:
                computeMeshVelocity(
                    f.elev3d, f.uv3d, f.w3d,
                    f.w_mesh3d, f.w_mesh_surf3d,
                    f.dw_mesh_dz_3d, f.bathymetry3d,
                    f.z_coord_ref3d)

        with timed_region('saltEq'):
            if o.solveSalt:
                self.timeStepper_salt3d.advance(t, s.dt, f.salt3d,
                                                updateForcings3d)
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(f.uv3d, f.uvDav3d,
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=f.bathymetry3d)
            copy3dFieldTo2d(f.uvDav3d, f.uvDav2d,
                            useBottomValue=False)
            copy2dFieldTo3d(f.uvDav2d, f.uvDav3d)
            # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
            uv2d_start = self.timeStepper2d.solution_start.split()[0]
            uv2d_start.assign(f.uvDav2d)
