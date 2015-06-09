"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from utility import *
import module_2d
import module_3d
import timeIntegration as timeIntegration
import time as timeMod
from mpi4py import MPI


# TODO rip name from here, use the name of the funct
class exportManager(object):
    """Handles a list of file exporter objects"""
    # maps each fieldname to long name and filename
    exportRules = {
        'uv2d': {'name': 'Depth averaged velocity',
                 'file': 'Velocity2d.pvd'},
        'elev2d': {'name': 'Elevation',
                   'file': 'Elevation2d.pvd'},
        'elev3d': {'name': 'Elevation',
                   'file': 'Elevation3d.pvd'},
        'uv3d': {'name': 'Velocity',
                 'file': 'Velocity3d.pvd'},
        'w3d': {'name': 'V.Velocity',
                'file': 'VertVelo3d.pvd'},
        'w3d_mesh': {'name': 'Mesh Velocity',
                     'file': 'MeshVelo3d.pvd'},
        'salt3d': {'name': 'Salinity',
                   'file': 'Salinity3d.pvd'},
        'uv2d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity2d.pvd'},
        'uv3d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity3d.pvd'},
        'uv2d_bot': {'name': 'Bottom Velocity',
                     'file': 'BotVelocity2d.pvd'},
        'nuv3d': {'name': 'Vertical Viscosity',
                  'file': 'Viscosity3d.pvd'},
        'barohead3d': {'name': 'Baroclinic head',
                       'file': 'Barohead3d.pvd'},
        'barohead2d': {'name': 'Dav baroclinic head',
                       'file': 'Barohead2d.pvd'},
        'gjvAlphaH3d': {'name': 'GJV Parameter h',
                        'file': 'GJVParamH.pvd'},
        'gjvAlphaV3d': {'name': 'GJV Parameter v',
                        'file': 'GJVParamV.pvd'},
        'smagViscosity': {'name': 'Smagorinsky viscosity',
                          'file': 'SmagViscosity3d.pvd'},
        'saltJumpDiff': {'name': 'Salt Jump Diffusivity',
                         'file': 'SaltJumpDiff3d.pvd'},
        }

    def __init__(self, outputDir, fieldsToExport, exportFunctions,
                 verbose=False):
        self.outputDir = outputDir
        self.fieldsToExport = fieldsToExport
        self.exportFunctions = exportFunctions
        self.verbose = verbose
        # for each field create an exporter
        self.exporters = {}
        for key in fieldsToExport:
            name = self.exportRules[key]['name']
            fn = self.exportRules[key]['file']
            space = self.exportFunctions[key][1]
            self.exporters[key] = exporter(space, name, outputDir, fn)

    def export(self):
        if self.verbose and commrank == 0:
            sys.stdout.write('Exporting: ')
        for key in self.exporters:
            field = self.exportFunctions[key][0]
            if field is not None:
                if self.verbose and commrank == 0:
                    sys.stdout.write(key+' ')
                    sys.stdout.flush()
                self.exporters[key].export(field)
        if self.verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def exportBathymetry(self, bathymetry2d):
        bathfile = File(os.path.join(self.outputDir, 'bath.pvd'))
        bathfile << bathymetry2d


class flowSolver(object):
    """Creates and solves coupled 2D-3D equations"""
    def __init__(self, mesh2d, bathymetry2d, n_layers):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.bathymetry2d = bathymetry2d
        self.mesh = extrudeMeshSigma(mesh2d, n_layers, bathymetry2d)

        # Time integrator setup
        self.TExport = 100.0  # export interval
        self.T = 1000.0  # Simulation duration
        self.uAdvection = Constant(0.0)  # magnitude of max horiz. velocity
        self.dt = None
        self.dt_2d = None
        self.M_modesplit = None

        # options
        self.cfl_2d = 1.0  # factor to scale the 2d time step
        self.cfl_3d = 1.0  # factor to scale the 2d time step
        self.nonlin = True  # use nonlinear shallow water equations
        self.use_wd = False  # use wetting-drying
        self.solveSalt = True  # solve salt transport
        self.solveVertDiffusion = True  # solve implicit vert diffusion
        self.useBottomFriction = True  # apply log layer bottom stress
        self.useParabolicViscosity = False  # compute parabolic eddy viscosity
        self.useALEMovingMesh = True  # 3D mesh tracks free surface
        self.useModeSplit = True  # run 2D/3D modes with different dt
        self.useSemiImplicit2D = True  # implicit 2D waves (only w. mode split)
        self.lin_drag = None  # 2D linear drag parameter tau/H/rho_0 = -drag*u
        self.hDiffusivity = None  # background diffusivity (set to Constant)
        self.vDiffusivity = None  # background diffusivity (set to Constant)
        self.hViscosity = None  # background viscosity (set to Constant)
        self.vViscosity = None  # background viscosity (set to Constant)
        self.coriolis = None  # Coriolis parameter (Constant or 2D Function)
        self.wind_stress = None  # stress at free surface (2D vector function)
        self.useSUPG = False  # SUPG stabilization for tracer advection
        self.useGJV = False  # nonlin gradient jump viscosity
        self.baroclinic = False  # comp and use internal pressure gradient
        self.uvLaxFriedrichs = Constant(1.0)  # scales uv stab. None omits
        self.checkVolConservation2d = False
        self.checkVolConservation3d = False
        self.checkSaltConservation = False
        self.checkSaltDeviation = False
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_eta3d',
                            'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                            'aux_mom_coupling',
                            'func_copy2dTo3d', 'func_copy3dTo2d',
                            'func_vert_int',
                            'supg', 'gjv']
        self.outputDir = 'outputs'
        self.fieldsToExport = ['elev2d', 'uv2d', 'uv3d', 'w3d']
        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}
        self.verbose = 0

    def setTimeStep(self):
        if self.useModeSplit:
            mesh_dt = self.eq_sw.getTimeStepAdvection(Umag=self.uAdvection)
            dt = self.cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
            dt = round(comm.allreduce(dt, dt, op=MPI.MIN))
            if dt == 0:
                raise Exception('3d advective time step is zero after rounding')
            if self.dt is None:
                self.dt = dt
            else:
                dt = self.dt
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
            if self.dt_2d is None:
                self.dt_2d = dt_2d
            self.M_modesplit = int(np.ceil(dt/self.dt_2d))
            self.dt_2d = dt/self.M_modesplit
        else:
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
            if self.dt is None:
                self.dt = dt_2d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        if commrank == 0:
            print 'dt =', self.dt
            print '2D dt =', self.dt_2d, self.M_modesplit
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.H_2d = FunctionSpace(self.mesh2d, 'CG', 2)
        self.W_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        self.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0)
        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.U = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
        self.U_visu = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1)
        self.U_scalar = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
        self.H = FunctionSpace(self.mesh, 'CG', 2, vfamily='CG', vdegree=1)

        # ----- fields
        self.solution2d = Function(self.W_2d, name='solution2d')
        if self.useBottomFriction:
            self.uv_bottom2d = Function(self.U_2d, name='Bottom Velocity')
            self.z_bottom2d = Function(self.P1_2d, name='Bot. Vel. z coord')
            self.bottom_drag2d = Function(self.P1_2d, name='Bottom Drag')
        else:
            self.uv_bottom2d = None
            self.z_bottom2d = None
            self.bottom_drag2d = None

        self.eta3d = Function(self.H, name='Elevation')
        self.eta3d_nplushalf = Function(self.H, name='Elevation')
        self.bathymetry3d = Function(self.P1, name='Bathymetry')
        self.uv3d = Function(self.U, name='Velocity')
        if self.useBottomFriction:
            self.uv_bottom3d = Function(self.U, name='Bottom Velocity')
            self.z_bottom3d = Function(self.P1, name='Bot. Vel. z coord')
            self.bottom_drag3d = Function(self.P1, name='Bottom Drag')
        else:
            self.uv_bottom3d = None
            self.z_bottom3d = None
            self.bottom_drag3d = None
        # z coordinate in the strecthed mesh
        self.z_coord3d = Function(self.P1, name='Bot. Vel. z coord')
        # z coordinate in the reference mesh (eta=0)
        self.z_coord_ref3d = Function(self.P1, name='Bot. Vel. z coord')
        self.uv3d_dav = Function(self.U, name='Depth Averaged Velocity 3d')
        self.uv2d_dav = Function(self.U_2d, name='Depth Averaged Velocity 2d')
        self.w3d = Function(self.H, name='Vertical Velocity')
        if self.useALEMovingMesh:
            self.w_mesh3d = Function(self.H, name='Vertical Velocity')
            self.dw_mesh_dz_3d = Function(self.H, name='Vertical Velocity dz')
            self.w_mesh_surf3d = Function(self.H, name='Vertical Velocity Surf')
            self.w_mesh_surf2d = Function(self.H_2d, name='Vertical Velocity Surf')
        else:
            self.w_mesh3d = self.dw_mesh_dz_3d = self.w_mesh_surf3d = None
        if self.solveSalt:
            self.salt3d = Function(self.H, name='Salinity')
        else:
            self.salt3d = None
        if self.solveVertDiffusion and self.useParabolicViscosity:
            self.viscosity_v3d = Function(self.P1, name='Eddy viscosity')
        else:
            self.viscosity_v3d = self.vViscosity
        if self.baroclinic:
            self.baroHead3d = Function(self.H, name='Baroclinic head')
            self.baroHeadInt3d = Function(self.H, name='V.int. baroclinic head')
            self.baroHead2d = Function(self.H_2d, name='DAv baroclinic head')
        else:
            self.baroHead3d = self.baroHead2d = None
        if self.coriolis is not None:
            if isinstance(self.coriolis, Constant):
                self.coriolis3d = self.coriolis
            else:
                self.coriolis3d = Function(self.P1, name='Coriolis parameter')
                copy2dFieldTo3d(self.coriolis, self.coriolis3d)
        else:
            self.coriolis3d = None
        if self.wind_stress is not None:
            self.wind_stress3d = Function(self.U, name='Wind stress')
            copy2dFieldTo3d(self.wind_stress, self.wind_stress3d)
        else:
            self.wind_stress3d = None
        if self.useSUPG:
            # TODO move these somewhere else? All form are now in equations...
            test = TestFunction(self.H)
            self.u_mag_func = Function(self.U_scalar, name='uvw magnitude')
            self.u_mag_func_h = Function(self.U_scalar, name='uv magnitude')
            self.u_mag_func_v = Function(self.U_scalar, name='w magnitude')
            self.SUPG_alpha = Constant(0.1)  # between 0 and 1
            self.hElemSize3d = getHorzontalElemSize(self.P1_2d, self.P1)
            self.vElemSize3d = getVerticalElemSize(self.P1_2d, self.P1)
            self.supg_gamma_h = Function(self.P1, name='gamma_h')
            self.supg_gamma_v = Function(self.P1, name='gamma_v')
            self.test_supg_h = self.supg_gamma_h*(self.uv3d[0]*Dx(test, 0) +
                                                  self.uv3d[1]*Dx(test, 1))
            self.test_supg_v = self.supg_gamma_v*(self.w3d*Dx(test, 2))
            self.test_supg_mass = self.SUPG_alpha/self.u_mag_func*(
                self.hElemSize3d/2*self.uv3d[0]*Dx(test, 0) +
                self.hElemSize3d/2*self.uv3d[1]*Dx(test, 1) +
                self.vElemSize3d/2*self.w3d*Dx(test, 2))
        else:
            self.test_supg_h = self.test_supg_v = self.test_supg_mass = None
        if self.useGJV:
            self.gjv_alpha = Constant(1.0)
            self.nonlinStab_h = Function(self.P0, name='GJV parameter h')
            self.nonlinStab_v = Function(self.P0, name='GJV parameter v')
        else:
            self.gjv_alpha = self.nonlinStab_h = self.nonlinStab_v = None

        # set initial values
        copy2dFieldTo3d(self.bathymetry2d, self.bathymetry3d)
        getZCoordFromMesh(self.z_coord_ref3d)

        # ----- Equations
        if self.useModeSplit:
            # full 2D shallow water equations
            self.eq_sw = module_2d.shallowWaterEquations(
                self.mesh2d, self.W_2d, self.solution2d, self.bathymetry2d,
                self.uv_bottom2d, self.bottom_drag2d,
                baro_head=self.baroHead2d,
                viscosity_h=self.hViscosity,
                uvLaxFriedrichs=self.uvLaxFriedrichs,
                coriolis=self.coriolis,
                wind_stress=self.wind_stress,
                lin_drag=self.lin_drag,
                nonlin=self.nonlin, use_wd=self.use_wd)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.solution2d.split()
            self.eq_sw = module_2d.freeSurfaceEquation(
                self.mesh2d, self.H_2d, eta, uv, self.bathymetry2d,
                nonlin=self.nonlin, use_wd=self.use_wd)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = module_3d.momentumEquation(
            self.mesh, self.U, self.U_scalar, bnd_markers,
            bnd_len, self.uv3d, self.eta3d,
            self.bathymetry3d, w=self.w3d,
            baro_head=self.baroHead3d,
            w_mesh=self.w_mesh3d,
            dw_mesh_dz=self.dw_mesh_dz_3d,
            viscosity_v=self.vViscosity, viscosity_h=self.hViscosity,
            uvLaxFriedrichs=self.uvLaxFriedrichs,
            coriolis=self.coriolis3d,
            lin_drag=self.lin_drag,
            nonlin=self.nonlin)
        if self.solveSalt:
            self.eq_salt = module_3d.tracerEquation(
                self.mesh, self.H, self.salt3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                diffusivity_h=self.hDiffusivity,
                diffusivity_v=self.vDiffusivity,
                test_supg_h=self.test_supg_h,
                test_supg_v=self.test_supg_v,
                test_supg_mass=self.test_supg_mass,
                nonlinStab_h=self.nonlinStab_h,
                nonlinStab_v=self.nonlinStab_v,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.solveVertDiffusion:
            self.eq_vertmomentum = module_3d.verticalMomentumEquation(
                self.mesh, self.U, self.U_scalar, self.uv3d, w=None,
                viscosity_v=self.viscosity_v3d,
                uv_bottom=self.uv_bottom3d,
                bottom_drag=self.bottom_drag3d,
                wind_stress=self.wind_stress3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']

        # ----- Time integrators
        self.setTimeStep()
        if self.useModeSplit:
            if self.useSemiImplicit2D:
                self.timeStepper = timeIntegration.coupledSSPRKSemiImplicit(self)
            else:
                self.timeStepper = timeIntegration.coupledSSPRKSync(self)
        else:
            self.timeStepper = timeIntegration.coupledSSPRKSingleMode(self)

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.P1_2d),
            'elev3d': (self.eta3d, self.P1),
            'uv3d': (self.uv3d, self.U_visu),
            'uv3d_dav': (self.uv3d_dav, self.U_visu),
            'w3d': (self.w3d, self.P1),
            'w3d_mesh': (self.w_mesh3d, self.P1),
            'salt3d': (self.salt3d, self.P1),
            'uv2d_dav': (self.uv2d_dav, self.U_visu_2d),
            'uv2d_bot': (self.uv_bottom2d, self.U_visu_2d),
            'nuv3d': (self.viscosity_v3d, self.P1),
            'barohead3d': (self.baroHead3d, self.P1),
            'barohead2d': (self.baroHead2d, self.P1_2d),
            'gjvAlphaH3d': (self.nonlinStab_h, self.P0),
            'gjvAlphaV3d': (self.nonlinStab_v, self.P0),
            }
        self.exporter = exportManager(self.outputDir, self.fieldsToExport,
                                      exportFuncs, verbose=self.verbose > 0)

        self._initialized = True

    def assignInitialConditions(self, elev=None, salt=None):
        if not self._initialized:
            self.mightyCreator()
        if elev is not None:
            uv2d, eta2d = self.solution2d.split()
            eta2d.project(elev)
            copy2dFieldTo3d(eta2d, self.eta3d)
            if self.useALEMovingMesh:
                updateCoordinates(self.mesh, self.eta3d, self.bathymetry3d,
                                  self.z_coord3d, self.z_coord_ref3d)
        if salt is not None and self.solveSalt:
            self.salt3d.project(salt)
        computeVertVelocity(self.w3d, self.uv3d, self.bathymetry3d)
        if self.useALEMovingMesh:
            computeMeshVelocity(self.eta3d, self.uv3d, self.w3d, self.w_mesh3d,
                                self.w_mesh_surf3d, self.w_mesh_surf2d,
                                self.dw_mesh_dz_3d,
                                self.bathymetry3d, self.z_coord_ref3d)
        if self.baroclinic:
            computeBaroclinicHead(self.salt3d, self.baroHead3d,
                                  self.baroHead2d, self.baroHeadInt3d,
                                  self.bathymetry3d)
        if self.useSUPG:
            updateSUPGGamma(self.uv3d, self.w3d, self.u_mag_func,
                            self.u_mag_func_h, self.u_mag_func_v,
                            self.hElemSize3d, self.vElemSize3d,
                            self.SUPG_alpha,
                            self.supg_gamma_h, self.supg_gamma_v)
        if self.useGJV:
            computeHorizGJVParameter(
                self.gjv_alpha, self.salt3d, self.nonlinStab_h,
                self.hElemSize3d, self.u_mag_func_h,
                maxval=800.0*self.uAdvection.dat.data[0])
            computeVertGJVParameter(
                self.gjv_alpha, self.salt3d, self.nonlinStab_v,
                self.vElemSize3d, self.u_mag_func_v,
                maxval=800.0*self.uAdvection.dat.data[0])
        if self.useGJV and not self.useSUPG:
            raise Exception('Currently GJV requires SUPG (comp of umag)')

        self.timeStepper.initialize()

        self.checkSaltConservation *= self.solveSalt
        self.checkSaltDeviation *= self.solveSalt
        self.checkVolConservation3d *= self.useALEMovingMesh

    def iterate(self, updateForcings=None, updateForcings3d=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.TExport

        # initialize conservation checks
        dx_3d = self.eq_momentum.dx
        dx_2d = self.eq_sw.dx
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d, dx_2d)
            print 'Initial volume 2d', Vol2d_0
        if self.checkVolConservation3d:
            Vol3d_0 = compVolume3d(dx_3d)
            print 'Initial volume 3d', Vol3d_0
        if self.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.salt3d, dx_3d)
            print 'Initial salt mass', Mass3d_0
        if self.checkSaltDeviation:
            saltVal = self.salt3d.dat.data.mean()
            print 'Initial mean salt value', saltVal

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.bathymetry2d)

        while t <= self.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, updateForcings,
                                     updateForcings3d)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.solution2d.split()[1])
                norm_u = norm(self.solution2d.split()[0])

                if self.checkVolConservation2d:
                    Vol2d = compVolume2d(self.solution2d.split()[1],
                                       self.bathymetry2d, dx_2d)
                if self.checkVolConservation3d:
                    Vol3d = compVolume3d(dx_3d)
                if self.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.salt3d, dx_3d)
                if self.checkSaltDeviation:
                    saltMin = self.salt3d.dat.data.min()
                    saltMax = self.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    if self.checkVolConservation3d:
                        print(line.format('vol 3d', (Vol3d_0 - Vol3d)/Vol3d_0))
                    if self.checkSaltConservation:
                        print(line.format('mass ',
                                          (Mass3d_0 - Mass3d)/Mass3d_0))
                    if self.checkSaltDeviation:
                        print('salt deviation {:g} {:g}'.format(*saltDev))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.TExport
                iExp += 1

                if commrank == 0 and len(self.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()


class flowSolverMimetic(object):
    """Creates and solves coupled 2D-3D equations"""
    def __init__(self, mesh2d, bathymetry2d, n_layers, order=1):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.bathymetry2d = bathymetry2d
        self.mesh = extrudeMeshSigma(mesh2d, n_layers, bathymetry2d)

        # Time integrator setup
        self.TExport = 100.0  # export interval
        self.T = 1000.0  # Simulation duration
        self.uAdvection = Constant(0.0)  # magnitude of max horiz. velocity
        self.dt = None
        self.dt_2d = None
        self.M_modesplit = None

        # options
        self.cfl_2d = 1.0  # factor to scale the 2d time step
        self.cfl_3d = 1.0  # factor to scale the 2d time step
        self.order = order  # polynomial order of elements
        self.nonlin = True  # use nonlinear shallow water equations
        self.use_wd = False  # use wetting-drying
        self.solveSalt = True  # solve salt transport
        self.solveVertDiffusion = True  # solve implicit vert diffusion
        self.useBottomFriction = True  # apply log layer bottom stress
        self.useParabolicViscosity = False  # compute parabolic eddy viscosity
        self.useALEMovingMesh = True  # 3D mesh tracks free surface
        self.useModeSplit = True  # run 2D/3D modes with different dt
        self.useSemiImplicit2D = True  # implicit 2D waves (only w. mode split)
        self.lin_drag = None  # 2D linear drag parameter tau/H/rho_0 = -drag*u
        self.hDiffusivity = None  # background diffusivity (set to Constant)
        self.vDiffusivity = None  # background diffusivity (set to Constant)
        self.hViscosity = None  # background viscosity (set to Constant)
        self.vViscosity = None  # background viscosity (set to Constant)
        self.coriolis = None  # Coriolis parameter (Constant or 2D Function)
        self.wind_stress = None  # stress at free surface (2D vector function)
        self.useSUPG = False  # SUPG stabilization for tracer advection
        self.useGJV = False  # nonlin gradient jump viscosity
        self.baroclinic = False  # comp and use internal pressure gradient
        self.smagorinskyFactor = None  # set to a Constant to use smag. visc.
        self.saltJumpDiffFactor = None  # set to a Constant to use nonlin diff.
        self.saltRange = Constant(30.0)  # value scale for salt to scale jumps
        self.uvLaxFriedrichs = Constant(1.0)  # scales uv stab. None omits
        self.tracerLaxFriedrichs = Constant(1.0)  # scales tracer stab. None omits
        self.checkVolConservation2d = False
        self.checkVolConservation3d = False
        self.checkSaltConservation = False
        self.checkSaltDeviation = False
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_eta3d',
                            'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                            'aux_mom_coupling',
                            'func_copy2dTo3d', 'func_copy3dTo2d',
                            'func_vert_int',
                            'supg', 'gjv']
        self.outputDir = 'outputs'
        self.fieldsToExport = ['elev2d', 'uv2d', 'uv3d', 'w3d']
        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}
        self.verbose = 0

    def setTimeStep(self):
        if self.useModeSplit:
            mesh_dt = self.eq_sw.getTimeStepAdvection(Umag=self.uAdvection)
            dt = self.cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
            dt = comm.allreduce(dt, dt, op=MPI.MIN)
            if round(dt) > 0:
                dt = round(dt)
            if self.dt is None:
                self.dt = dt
            else:
                dt = float(self.dt)
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
            if self.dt_2d is None:
                self.dt_2d = dt_2d
            self.M_modesplit = int(np.ceil(dt/self.dt_2d))
            self.dt_2d = dt/self.M_modesplit
        else:
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
            if self.dt is None:
                self.dt = dt_2d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        if commrank == 0:
            print 'dt =', self.dt
            print '2D dt =', self.dt_2d, self.M_modesplit
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.order+1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', self.order)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_visu_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.W_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        self.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0)
        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
        Uh_elt = FiniteElement('RT', triangle, self.order+1)
        Uv_elt = FiniteElement('DG', interval, self.order)
        U_elt = HDiv(OuterProductElement(Uh_elt, Uv_elt))
        self.U = FunctionSpace(self.mesh, U_elt)
        Uvint_elt = FiniteElement('DG', interval, self.order+1)
        Uint_elt = HDiv(OuterProductElement(Uh_elt, Uv_elt))
        self.Uint = FunctionSpace(self.mesh, Uint_elt)
        self.U_visu = VectorFunctionSpace(self.mesh, 'CG', self.order, vfamily='CG', vdegree=self.order)
        self.U_scalar = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=self.order)
        self.H = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=max(0, self.order))
        self.Hint = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=self.order+1)
        self.H_visu = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        # TODO w must live in a HDiv space as well, like this (a 3d vector field)
        Hh_elt = FiniteElement('DG', triangle, self.order)
        Hv_elt = FiniteElement('CG', interval, self.order+1)
        H_elt = HDiv(OuterProductElement(Hh_elt, Hv_elt))
        self.Hvec = FunctionSpace(self.mesh, H_elt)
        self.Hvec_visu = VectorFunctionSpace(self.mesh, 'CG',
                                             1,
                                             vfamily='CG',
                                             vdegree=1)

        # ----- fields
        self.solution2d = Function(self.W_2d, name='solution2d')
        if self.useBottomFriction:
            self.uv_bottom2d = Function(self.U_2d, name='Bottom Velocity')
            self.z_bottom2d = Function(self.P1_2d, name='Bot. Vel. z coord')
            self.bottom_drag2d = Function(self.P1_2d, name='Bottom Drag')
        else:
            self.uv_bottom2d = None
            self.z_bottom2d = None
            self.bottom_drag2d = None

        self.eta3d = Function(self.H, name='Elevation')
        self.eta3dCG = Function(self.P1, name='Elevation')
        self.eta3d_nplushalf = Function(self.H, name='Elevation')
        self.bathymetry3d = Function(self.P1, name='Bathymetry')
        self.uv3d = Function(self.U, name='Velocity')
        if self.useBottomFriction:
            self.uv_bottom3d = Function(self.U, name='Bottom Velocity')
            self.z_bottom3d = Function(self.P1, name='Bot. Vel. z coord')
            self.bottom_drag3d = Function(self.P1, name='Bottom Drag')
        else:
            self.uv_bottom3d = None
            self.z_bottom3d = None
            self.bottom_drag3d = None
        # z coordinate in the strecthed mesh
        self.z_coord3d = Function(self.P1, name='z coord')
        # z coordinate in the reference mesh (eta=0)
        self.z_coord_ref3d = Function(self.P1, name='ref z coord')
        self.uv3d_dav = Function(self.U, name='Depth Averaged Velocity 3d')
        self.uv2d_dav = Function(self.U_2d, name='Depth Averaged Velocity 2d')
        self.uv3d_mag = Function(self.P0, name='Velocity magnitude')
        self.uv3d_P1 = Function(self.P1v, name='Smoothed Velocity')
        self.w3d = Function(self.Hvec, name='Vertical Velocity')
        if self.useALEMovingMesh:
            self.w_mesh3d = Function(self.H, name='Vertical Velocity')
            self.dw_mesh_dz_3d = Function(self.H, name='Vertical Velocity dz')
            self.w_mesh_surf3d = Function(self.H, name='Vertical Velocity Surf')
            self.w_mesh_surf2d = Function(self.H_2d, name='Vertical Velocity Surf')
        else:
            self.w_mesh3d = self.dw_mesh_dz_3d = self.w_mesh_surf3d = None
        if self.solveSalt:
            self.salt3d = Function(self.H, name='Salinity')
        else:
            self.salt3d = None
        if self.solveVertDiffusion and self.useParabolicViscosity:
            self.viscosity_v3d = Function(self.P1, name='Eddy viscosity')
        else:
            self.viscosity_v3d = self.vViscosity
        if self.baroclinic:
            self.baroHead3d = Function(self.H, name='Baroclinic head')
            self.baroHeadInt3d = Function(self.H, name='V.int. baroclinic head')
            self.baroHead2d = Function(self.H_2d, name='DAv baroclinic head')
        else:
            self.baroHead3d = self.baroHead2d = None
        if self.coriolis is not None:
            if isinstance(self.coriolis, Constant):
                self.coriolis3d = self.coriolis
            else:
                self.coriolis3d = Function(self.P1, name='Coriolis parameter')
                copy2dFieldTo3d(self.coriolis, self.coriolis3d)
        else:
            self.coriolis3d = None
        if self.wind_stress is not None:
            self.wind_stress3d = Function(self.U_visu, name='Wind stress')
            copy2dFieldTo3d(self.wind_stress, self.wind_stress3d)
        else:
            self.wind_stress3d = None
        self.vElemSize3d = Function(self.P1DG, name='element height')
        self.vElemSize2d = Function(self.P1DG_2d, name='element height')
        self.hElemSize3d = getHorzontalElemSize(self.P1_2d, self.P1)
        if self.useSUPG:
            # TODO move these somewhere else? All form are now in equations...
            test = TestFunction(self.H)
            self.u_mag_func = Function(self.U_scalar, name='uvw magnitude')
            self.u_mag_func_h = Function(self.U_scalar, name='uv magnitude')
            self.u_mag_func_v = Function(self.U_scalar, name='w magnitude')
            self.SUPG_alpha = Constant(0.1)  # between 0 and 1
            # FIXME clashes vElemSize3d above
            self.vElemSize3d = getVerticalElemSize(self.P1_2d, self.P1)
            self.supg_gamma_h = Function(self.P1, name='gamma_h')
            self.supg_gamma_v = Function(self.P1, name='gamma_v')
            self.test_supg_h = self.supg_gamma_h*(self.uv3d[0]*Dx(test, 0) +
                                                  self.uv3d[1]*Dx(test, 1))
            self.test_supg_v = self.supg_gamma_v*(self.w3d*Dx(test, 2))
            self.test_supg_mass = self.SUPG_alpha/self.u_mag_func*(
                self.hElemSize3d/2*self.uv3d[0]*Dx(test, 0) +
                self.hElemSize3d/2*self.uv3d[1]*Dx(test, 1) +
                self.vElemSize3d/2*self.w3d*Dx(test, 2))
        else:
            self.test_supg_h = self.test_supg_v = self.test_supg_mass = None
        if self.useGJV:
            self.gjv_alpha = Constant(1.0)
            self.nonlinStab_h = Function(self.P0, name='GJV parameter h')
            self.nonlinStab_v = Function(self.P0, name='GJV parameter v')
        else:
            self.gjv_alpha = self.nonlinStab_h = self.nonlinStab_v = None
        if self.smagorinskyFactor is not None:
            self.smag_viscosity = Function(self.P1, name='Smagorinsky viscosity')
        # total horizontal viscosity
        self.tot_h_visc = None
        if self.hViscosity is not None and self.smagorinskyFactor is not None:
            self.tot_h_visc = self.hViscosity + self.smag_viscosity
        elif self.hViscosity is None and self.smagorinskyFactor is not None:
            self.tot_h_visc = self.smag_viscosity
        elif self.hViscosity is not None and self.smagorinskyFactor is None:
            self.tot_h_visc = self.hViscosity
        if self.saltJumpDiffFactor is not None:
            self.saltJumpDiff = Function(self.P1, name='Salt Jump Diffusivity')
            self.tot_salt_h_diff = self.saltJumpDiff
            if self.hViscosity is not None:
                self.tot_salt_h_diff += self.hViscosity
        else:
            self.tot_salt_h_diff = self.hDiffusivity

        # set initial values
        copy2dFieldTo3d(self.bathymetry2d, self.bathymetry3d)
        getZCoordFromMesh(self.z_coord_ref3d)
        self.z_coord3d.assign(self.z_coord_ref3d)
        computeElemHeight(self.z_coord3d, self.vElemSize3d)
        copy3dFieldTo2d(self.vElemSize3d, self.vElemSize2d)

        # ----- Equations
        if self.useModeSplit:
            # full 2D shallow water equations
            self.eq_sw = module_2d.shallowWaterEquations(
                self.mesh2d, self.W_2d, self.solution2d, self.bathymetry2d,
                self.uv_bottom2d, self.bottom_drag2d,
                baro_head=self.baroHead2d,
                viscosity_h=self.hViscosity,
                uvLaxFriedrichs=self.uvLaxFriedrichs,
                coriolis=self.coriolis,
                wind_stress=self.wind_stress,
                lin_drag=self.lin_drag,
                nonlin=self.nonlin, use_wd=self.use_wd)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.solution2d.split()
            self.eq_sw = module_2d.freeSurfaceEquation(
                self.mesh2d, self.H_2d, eta, uv, self.bathymetry2d,
                nonlin=self.nonlin, use_wd=self.use_wd)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = module_3d.momentumEquation(
            self.mesh, self.U, self.U_scalar, bnd_markers,
            bnd_len, self.uv3d, self.eta3d,
            self.bathymetry3d, w=self.w3d,
            baro_head=self.baroHead3d,
            w_mesh=self.w_mesh3d,
            dw_mesh_dz=self.dw_mesh_dz_3d,
            viscosity_v=self.vViscosity, viscosity_h=self.tot_h_visc,
            uvLaxFriedrichs=self.uvLaxFriedrichs,
            #uvMag=self.uv3d_mag,
            uvP1=self.uv3d_P1,
            coriolis=self.coriolis3d,
            lin_drag=self.lin_drag,
            nonlin=self.nonlin)
        if self.solveSalt:
            self.eq_salt = module_3d.tracerEquation(
                self.mesh, self.H, self.salt3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                diffusivity_h=self.tot_salt_h_diff,
                diffusivity_v=self.vDiffusivity,
                #uvMag=self.uv3d_mag,
                uvP1=self.uv3d_P1,
                tracerLaxFriedrichs=self.tracerLaxFriedrichs,
                test_supg_h=self.test_supg_h,
                test_supg_v=self.test_supg_v,
                test_supg_mass=self.test_supg_mass,
                nonlinStab_h=self.nonlinStab_h,
                nonlinStab_v=self.nonlinStab_v,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.solveVertDiffusion:
            self.eq_vertmomentum = module_3d.verticalMomentumEquation(
                self.mesh, self.U, self.U_scalar, self.uv3d, w=None,
                viscosity_v=self.viscosity_v3d,
                uv_bottom=self.uv_bottom3d,
                bottom_drag=self.bottom_drag3d,
                wind_stress=self.wind_stress3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']

        # ----- Time integrators
        self.setTimeStep()
        if self.useModeSplit:
            if self.useSemiImplicit2D:
                self.timeStepper = timeIntegration.coupledSSPRKSemiImplicit(self)
            else:
                self.timeStepper = timeIntegration.coupledSSPRKSync(self)
        else:
            self.timeStepper = timeIntegration.coupledSSPRKSingleMode(self)

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.H_visu_2d),
            'elev3d': (self.eta3d, self.H_visu),
            'uv3d': (self.uv3d, self.U_visu),
            'uv3d_dav': (self.uv3d_dav, self.U_visu),
            'w3d': (self.w3d, self.Hvec_visu),
            'w3d_mesh': (self.w_mesh3d, self.P1),
            'salt3d': (self.salt3d, self.H_visu),
            'uv2d_dav': (self.uv2d_dav, self.U_visu_2d),
            'uv2d_bot': (self.uv_bottom2d, self.U_visu_2d),
            'nuv3d': (self.viscosity_v3d, self.P1),
            'barohead3d': (self.baroHead3d, self.P1),
            'barohead2d': (self.baroHead2d, self.P1_2d),
            'gjvAlphaH3d': (self.nonlinStab_h, self.P0),
            'gjvAlphaV3d': (self.nonlinStab_v, self.P0),
            'smagViscosity': (self.smag_viscosity, self.P1),
            'saltJumpDiff': (self.saltJumpDiff, self.P1),
            }
        self.exporter = exportManager(self.outputDir, self.fieldsToExport,
                                      exportFuncs, verbose=self.verbose > 0)
        self.uvP1_projector = projector(self.uv3d, self.uv3d_P1)

        self._initialized = True

    def assignInitialConditions(self, elev=None, salt=None):
        if not self._initialized:
            self.mightyCreator()
        if elev is not None:
            uv2d, eta2d = self.solution2d.split()
            eta2d.project(elev)
            copy2dFieldTo3d(eta2d, self.eta3d)
            self.eta3dCG.project(self.eta3d)
            if self.useALEMovingMesh:
                updateCoordinates(self.mesh, self.eta3dCG, self.bathymetry3d,
                                  self.z_coord3d, self.z_coord_ref3d)
                computeElemHeight(self.z_coord3d, self.vElemSize3d)
                copy3dFieldTo2d(self.vElemSize3d, self.vElemSize2d)

        if salt is not None and self.solveSalt:
            self.salt3d.project(salt)
        computeVertVelocity(self.w3d, self.uv3d, self.bathymetry3d)
        if self.useALEMovingMesh:
            computeMeshVelocity(self.eta3d, self.uv3d, self.w3d, self.w_mesh3d,
                                self.w_mesh_surf3d, self.w_mesh_surf2d,
                                self.dw_mesh_dz_3d,
                                self.bathymetry3d, self.z_coord_ref3d)
        if self.baroclinic:
            computeBaroclinicHead(self.salt3d, self.baroHead3d,
                                  self.baroHead2d, self.baroHeadInt3d,
                                  self.bathymetry3d)
        if self.useSUPG:
            updateSUPGGamma(self.uv3d, self.w3d, self.u_mag_func,
                            self.u_mag_func_h, self.u_mag_func_v,
                            self.hElemSize3d, self.vElemSize3d,
                            self.SUPG_alpha,
                            self.supg_gamma_h, self.supg_gamma_v)
        if self.useGJV:
            computeHorizGJVParameter(
                self.gjv_alpha, self.salt3d, self.nonlinStab_h,
                self.hElemSize3d, self.u_mag_func_h,
                maxval=800.0*self.uAdvection.dat.data[0])
            computeVertGJVParameter(
                self.gjv_alpha, self.salt3d, self.nonlinStab_v,
                self.vElemSize3d, self.u_mag_func_v,
                maxval=800.0*self.uAdvection.dat.data[0])
        if self.useGJV and not self.useSUPG:
            raise Exception('Currently GJV requires SUPG (comp of umag)')

        self.timeStepper.initialize()

        self.checkSaltConservation *= self.solveSalt
        self.checkSaltDeviation *= self.solveSalt
        self.checkVolConservation3d *= self.useALEMovingMesh

    def iterate(self, updateForcings=None, updateForcings3d=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.TExport

        # initialize conservation checks
        dx_3d = self.eq_momentum.dx
        dx_2d = self.eq_sw.dx
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d, dx_2d)
            print 'Initial volume 2d', Vol2d_0
        if self.checkVolConservation3d:
            Vol3d_0 = compVolume3d(dx_3d)
            print 'Initial volume 3d', Vol3d_0
        if self.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.salt3d, dx_3d)
            print 'Initial salt mass', Mass3d_0
        if self.checkSaltDeviation:
            saltVal = self.salt3d.dat.data.mean()
            print 'Initial mean salt value', saltVal

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.bathymetry2d)

        while t <= self.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, updateForcings,
                                     updateForcings3d)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.solution2d.split()[1])
                norm_u = norm(self.solution2d.split()[0])

                if self.checkVolConservation2d:
                    Vol2d = compVolume2d(self.solution2d.split()[1],
                                       self.bathymetry2d, dx_2d)
                if self.checkVolConservation3d:
                    Vol3d = compVolume3d(dx_3d)
                if self.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.salt3d, dx_3d)
                if self.checkSaltDeviation:
                    saltMin = self.salt3d.dat.data.min()
                    saltMax = self.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    if self.checkVolConservation3d:
                        print(line.format('vol 3d', (Vol3d_0 - Vol3d)/Vol3d_0))
                    if self.checkSaltConservation:
                        print(line.format('mass ',
                                          (Mass3d_0 - Mass3d)/Mass3d_0))
                    if self.checkSaltDeviation:
                        print('salt deviation {:g} {:g}'.format(*saltDev))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.TExport
                iExp += 1

                if commrank == 0 and len(self.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()


class flowSolver2d(object):
    """Creates and solves 2D depth averaged equations"""
    def __init__(self, mesh2d, bathymetry2d):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.bathymetry2d = bathymetry2d

        # Time integrator setup
        self.TExport = 100.0  # export interval
        self.T = 1000.0  # Simulation duration
        self.uAdvection = Constant(0.0)  # magnitude of max horiz. velocity
        self.dt = None

        # options
        self.cfl_2d = 1.0  # factor to scale the 2d time step
        self.nonlin = True  # use nonlinear shallow water equations
        self.use_wd = False  # use wetting-drying
        self.lin_drag = None  # linear drag parameter tau/H/rho_0 = -drag*u
        self.hDiffusivity = None  # background diffusivity (set to Constant)
        self.hViscosity = None  # background viscosity (set to Constant)
        self.coriolis = None  # Coriolis parameter (Constant or 2D Function)
        self.wind_stress = None  # stress at free surface (2D vector function)
        self.uvLaxFriedrichs = Constant(1.0)  # scales uv stab. None omits
        self.checkVolConservation2d = False
        self.timeStepperType = 'SSPRK33'
        self.timerLabels = ['mode2d']
        self.outputDir = 'outputs'
        self.fieldsToExport = ['elev2d', 'uv2d']
        self.bnd_functions = {'shallow_water': {}}
        self.verbose = 0

    def setTimeStep(self):
        mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
        dt = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
        dt = comm.allreduce(dt, dt, op=MPI.MIN)
        if self.dt is None:
            self.dt = dt
        if commrank == 0:
            print 'dt =', self.dt
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.H_2d = FunctionSpace(self.mesh2d, 'CG', 2)
        self.W_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.solution2d = Function(self.W_2d, name='solution2d')

        # ----- Equations
        self.eq_sw = module_2d.shallowWaterEquations(
            self.mesh2d, self.W_2d, self.solution2d, self.bathymetry2d,
            lin_drag=self.lin_drag,
            viscosity_h=self.hViscosity,
            uvLaxFriedrichs=self.uvLaxFriedrichs,
            coriolis=self.coriolis,
            wind_stress=self.wind_stress,
            nonlin=self.nonlin, use_wd=self.use_wd)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeIntegration.SSPRK33Stage(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeIntegration.ForwardEuler(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeIntegration.CrankNicolson(self.eq_sw, self.dt,
                                                             self.eq_sw.solver_parameters)
        else:
            raise Exception('Unknown time integrator type: '+str(self.timeStepperType))

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.P1_2d),
            }
        self.exporter = exportManager(self.outputDir, self.fieldsToExport,
                                      exportFuncs, verbose=self.verbose > 0)
        self._initialized = True

    def assignInitialConditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.mightyCreator()
        uv2d, eta2d = self.solution2d.split()
        if elev is not None:
            eta2d.project(elev)
        if uv_init is not None:
            uv2d.project(uv_init)

        self.timeStepper.initialize(self.solution2d)

    def iterate(self, updateForcings=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.TExport

        # initialize conservation checks
        dx_2d = self.eq_sw.dx
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d, dx_2d)
            print 'Initial volume 2d', Vol2d_0

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.bathymetry2d)

        while t <= self.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, self.solution2d,
                                     updateForcings)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.solution2d.split()[1])
                norm_u = norm(self.solution2d.split()[0])

                if self.checkVolConservation2d:
                    Vol2d = compVolume2d(self.solution2d.split()[1],
                                       self.bathymetry2d, dx_2d)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.TExport
                iExp += 1

                if commrank == 0 and len(self.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()


class flowSolver2dMimetic(object):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry2d, order=1):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.bathymetry2d = bathymetry2d

        # Time integrator setup
        self.TExport = 100.0  # export interval
        self.T = 1000.0  # Simulation duration
        self.uAdvection = Constant(0.0)  # magnitude of max horiz. velocity
        self.dt = None

        # options
        self.cfl_2d = 1.0  # factor to scale the 2d time step
        self.order = order  # polynomial order of elements
        self.nonlin = True  # use nonlinear shallow water equations
        self.use_wd = False  # use wetting-drying
        self.lin_drag = None  # linear drag parameter tau/H/rho_0 = -drag*u
        self.hDiffusivity = None  # background diffusivity (set to Constant)
        self.hViscosity = None  # background viscosity (set to Constant)
        self.coriolis = None  # Coriolis parameter (Constant or 2D Function)
        self.wind_stress = None  # stress at free surface (2D vector function)
        self.uvLaxFriedrichs = Constant(1.0)  # scales uv stab. None omits
        self.checkVolConservation2d = False
        self.timeStepperType = 'SSPRK33'
        self.timerLabels = ['mode2d']
        self.outputDir = 'outputs'
        self.fieldsToExport = ['elev2d', 'uv2d']
        self.bnd_functions = {'shallow_water': {}}
        self.verbose = 0

    def setTimeStep(self):
        mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
        dt = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
        dt = comm.allreduce(dt, dt, op=MPI.MIN)
        if self.dt is None:
            self.dt = dt
        if commrank == 0:
            print 'dt =', self.dt
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0)
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.order+1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', self.order)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_visu_2d = FunctionSpace(self.mesh2d, 'CG', max(self.order, 1))
        self.W_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.solution2d = Function(self.W_2d, name='solution2d')
        self.volumeFlux2d = Function(self.U_2d, name='volumeFlux2d')

        # ----- Equations
        self.eq_sw = module_2d.shallowWaterEquations(
            self.mesh2d, self.W_2d, self.solution2d, self.bathymetry2d,
            lin_drag=self.lin_drag,
            viscosity_h=self.hViscosity,
            uvLaxFriedrichs=self.uvLaxFriedrichs,
            coriolis=self.coriolis,
            wind_stress=self.wind_stress,
            volumeFlux=self.volumeFlux2d,
            nonlin=self.nonlin, use_wd=self.use_wd)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeIntegration.SSPRK33Stage(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeIntegration.ForwardEuler(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeIntegration.CrankNicolson(self.eq_sw, self.dt,
                                                             self.eq_sw.solver_parameters)
        else:
            raise Exception('Unknown time integrator type: '+str(self.timeStepperType))

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.H_visu_2d),
            }
        self.exporter = exportManager(self.outputDir, self.fieldsToExport,
                                      exportFuncs, verbose=self.verbose > 0)
        self._initialized = True

    def assignInitialConditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.mightyCreator()
        uv2d, eta2d = self.solution2d.split()
        if elev is not None:
            eta2d.project(elev)
        if uv_init is not None:
            uv2d.project(uv_init)

        self.timeStepper.initialize(self.solution2d)

    def iterate(self, updateForcings=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.TExport

        # initialize conservation checks
        dx_2d = self.eq_sw.dx
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d, dx_2d)
            print 'Initial volume 2d', Vol2d_0

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.bathymetry2d)

        while t <= self.T + T_epsilon:

            #self.timeStepper.advance(t, self.dt, self.solution2d,
                                     #updateForcings)
            for i in range(3):
                uv, eta = self.solution2d.split()
                if self.nonlin:
                    computeVolumeFlux(uv, (eta+self.bathymetry2d),
                                      self.volumeFlux2d, self.eq_sw.dx)
                else:
                    computeVolumeFlux(uv, self.bathymetry2d,
                                      self.volumeFlux2d, self.eq_sw.dx)
                self.timeStepper.solveStage(i, t, self.dt, self.solution2d,
                                            updateForcings)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.solution2d.split()[1])
                norm_u = norm(self.solution2d.split()[0])

                if self.checkVolConservation2d:
                    Vol2d = compVolume2d(self.solution2d.split()[1],
                                       self.bathymetry2d, dx_2d)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.TExport
                iExp += 1

                if commrank == 0 and len(self.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
