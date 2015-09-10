"""
Time integrators for solving coupled 2D-3D-tracer equations.

Tuomas Karna 2015-07-06
"""
from utility import *
import timeIntegrator


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
        if self.solver.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
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
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

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
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d.split()[1]
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, s.eta3d, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    computeElemHeight(s.z_coord3d, s.vElemSize3d)
                    copy3dFieldTo2d(s.vElemSize3d, s.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False, elemHeight=s.vElemSize2d)
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav, elemHeight=s.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(s.uv2d_dav)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    s.uv3d -= s.uv3d_tmp
                    s.uv2d_to_DAV_projector.project()  # uv2d to uv2d_dav
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav,
                                    elemHeight=s.vElemSize3d)
                    s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    s.uv3d += s.uv3d_tmp

            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    computeBottomFriction(
                        s.uv3d, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange,
                                                    s.maxHDiffusivity)

        self.sol2d_n.assign(sol2d)  # keep copy of eta_n
        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, s.salt3d,
                                                       updateForcings3d)
                    if s.useLimiterForTracers:
                        s.tracerLimiter.apply(s.salt3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, s.uv3d)
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
        if self.solver.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
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
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

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
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doALEUpdate=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d.split()[1]
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
                s.eta3d_to_CG_projector.project()
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh and doALEUpdate:
                    updateCoordinates(
                        s.mesh, s.eta3dCG, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    computeElemHeight(s.z_coord3d, s.vElemSize3d)
                    copy3dFieldTo2d(s.vElemSize3d, s.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                    self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False,
                                    elemHeight=s.vElemSize2d)
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav,
                                    elemHeight=s.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(s.uv2d_dav)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    s.uv3d -= s.uv3d_tmp
                    s.uv2d_to_DAV_projector.project()  # uv2d to uv2d_dav
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav,
                                    elemHeight=s.vElemSize3d)
                    s.uvDAV_to_tmp_projector.project()  # project uv_dav to uv3d_tmp
                    s.uv3d += s.uv3d_tmp
            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    s.uvP1_projector.project()
                    computeBottomFriction(
                        s.uv3d_P1, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d,
                        s.vElemSize2d, s.vElemSize3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.parabViscosity_v)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange,
                                                    s.maxHDiffusivity)

        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, s.salt3d,
                                                       updateForcings3d)
                    if s.useLimiterForTracers:
                        s.tracerLimiter.apply(s.salt3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, s.uv3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, s.dt, sol2d,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=lastStep,
                               doALEUpdate=lastStep,
                               doStabParams=lastStep)


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
        if self.solver.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33Stage(
                solver.eq_salt,
                solver.dt_2d)
        if self.solver.solveVertDiffusion:
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
        self.timeStepper2d.initialize(self.solver.solution2d.split()[1])
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, s.eta3d, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt_2d, s.uv3d)
            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    computeBottomFriction(
                        s.uv3d, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False, elemHeight=s.vElemSize2d)
                    s.uv2dDAV_to_uv2d_projector.project()
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange,
                                                    s.maxHDiffusivity)

        for k in range(self.timeStepper2d.nstages):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt_2d, s.salt3d,
                                                       updateForcings3d)
                    if s.useLimiterForTracers:
                        s.tracerLimiter.apply(s.salt3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt_2d, s.uv3d)
            with timed_region('mode2d'):
                uv, eta = s.solution2d.split()
                self.timeStepper2d.solveStage(k, t, s.dt_2d, eta,
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
            funcs_nplushalf={'eta': solver.eta3d_nplushalf})
        if self.solver.solveSalt:
            self.timeStepper_salt3d = timeIntegrator.SSPRK33(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d = timeIntegrator.CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        # SSPRK33 time integration loop
        with timed_region('mode2d'):
            self.timeStepper2d.advance(t, s.dt_2d, s.solution2d,
                                       updateForcings)
        with timed_region('aux_eta3d'):
            eta_n = s.solution2d.split()[1]
            copy2dFieldTo3d(eta_n, s.eta3d)  # at t_{n+1}
            eta_nph = self.timeStepper2d.solution_nplushalf.split()[1]
            copy2dFieldTo3d(eta_nph, s.eta3d_nplushalf)  # at t_{n+1/2}
        with timed_region('aux_mesh_ale'):
            if s.useALEMovingMesh:
                updateCoordinates(
                    s.mesh, s.eta3d, s.bathymetry3d,
                    s.z_coord3d, s.z_coord_ref3d)
        with timed_region('aux_friction'):
            if s.useBottomFriction:
                computeBottomFriction(
                    s.uv3d, s.uv_bottom2d,
                    s.uv_bottom3d, s.z_coord3d,
                    s.z_bottom2d, s.z_bottom3d,
                    s.bathymetry2d, s.bottom_drag2d,
                    s.bottom_drag3d)
            if s.useParabolicViscosity:
                computeParabolicViscosity(
                    s.uv_bottom3d, s.bottom_drag3d,
                    s.bathymetry3d,
                    s.viscosity_v3d)
        with timed_region('aux_barolinicity'):
            if s.baroclinic:
                computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                      s.baroHead2d, s.baroHeadInt3d,
                                      s.bathymetry3d)

        with timed_region('momentumEq'):
            self.timeStepper_mom3d.advance(t, s.dt, s.uv3d,
                                           updateForcings3d)
        with timed_region('vert_diffusion'):
            if s.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d, None)
        with timed_region('continuityEq'):
            computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                s.eq_momentum.boundary_markers,
                                s.eq_momentum.bnd_functions)
        with timed_region('aux_mesh_ale'):
            if s.useALEMovingMesh:
                computeMeshVelocity(
                    s.eta3d, s.uv3d, s.w3d,
                    s.w_mesh3d, s.w_mesh_surf3d,
                    s.dw_mesh_dz_3d, s.bathymetry3d,
                    s.z_coord_ref3d)
        with timed_region('aux_friction'):
            if s.useBottomFriction:
                computeBottomFriction(
                    s.uv3d, s.uv_bottom2d,
                    s.uv_bottom3d, s.z_coord3d,
                    s.z_bottom2d, s.z_bottom3d,
                    s.bathymetry2d, s.bottom_drag2d,
                    s.bottom_drag3d)

        with timed_region('saltEq'):
            if s.solveSalt:
                self.timeStepper_salt3d.advance(t, s.dt, s.salt3d,
                                                updateForcings3d)
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=s.bathymetry3d)
            copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                            useBottomValue=False)
            copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav)
            # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
            uv2d_start = self.timeStepper2d.solution_start.split()[0]
            uv2d_start.assign(s.uv2d_dav)
