"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from cofs.utility import *
import cofs.module_2d as mode2d
import cofs.module_3d as mode3d
import cofs.timeIntegration as timeIntegration
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
        'uv2d_bot': {'name': 'Bottom Velocity',
                     'file': 'BotVelocity2d.pvd'},
        'nuv3d': {'name': 'Vertical Viscosity',
                  'file': 'Viscosity3d.pvd'},
        }

    def __init__(self, outputDir, fieldsToExport, exportFunctions,
                 verbose=False):
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
        self.nonlin = True  # use nonlinear shallow water equations
        self.use_wd = False  # use wetting-drying
        self.solveSalt = True  # solve salt transport
        self.solveVertDiffusion = True  # solve implicit vert diffusion
        self.useBottomFriction = True  # apply log layer bottom stress
        self.useALEMovingMesh = True  # 3D mesh tracks free surface
        self.checkVolConservation2d = False
        self.checkVolConservation3d = False
        self.checkSaltConservation = False
        self.checkSaltDeviation = False
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_functions']
        self.outputDir = 'outputs'
        self.fieldsToExport = ['elev2d', 'uv2d', 'uv3d', 'w3d']
        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}

        # solver parameters
        self.solver_parameters2d = {'ksp_rtol': 1e-12,
                                    'ksp_atol': 1e-16}

    def setTimeStep(self, cfl_3d=1.0, cfl_2d=1.0):
        mesh_dt = self.eq_sw.getTimeStepAdvection(Umag=self.uAdvection)
        dt = cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
        dt = round(comm.allreduce(dt, dt, op=MPI.MIN))
        if self.dt is None:
            self.dt = dt
        else:
            dt = self.dt
        mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
        dt_2d = cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
        dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
        if self.dt_2d is None:
            self.dt_2d = dt_2d
        self.M_modesplit = int(np.ceil(dt/self.dt_2d))
        self.dt_2d = dt/self.M_modesplit

        if commrank == 0:
            print 'dt =', self.dt
            print '2D dt =', self.dt_2d, self.M_modesplit
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.H_2d = FunctionSpace(self.mesh2d, 'CG', 2)
        self.W_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.U = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
        self.U_visu = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
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
        self.uv3d_dav = Function(self.U, name='Depth Averaged Velocity')
        self.uv2d_dav = Function(self.U_2d, name='Depth Averaged Velocity')
        self.uv2d_dav_old = Function(self.U_2d, name='Depth Averaged Velocity')
        self.w3d = Function(self.H, name='Vertical Velocity')
        if self.useALEMovingMesh:
            self.w_mesh3d = Function(self.H, name='Vertical Velocity')
            self.dw_mesh_dz_3d = Function(self.H, name='Vertical Velocity')
            self.w_mesh_surf3d = Function(self.H, name='Vertical Velocity')
        else:
            self.w_mesh3d = None
            self.dw_mesh_dz_3d = None
            self.w_mesh_surf3d = None
        if self.solveSalt:
            self.salt3d = Function(self.H, name='Salinity')
        else:
            self.salt3d = None
        if self.solveVertDiffusion:
            self.viscosity_v3d = Function(self.P1, name='Vertical Velocity')
        else:
            self.viscosity_v3d = None
        # set initial values
        copy2dFieldTo3d(self.bathymetry2d, self.bathymetry3d)
        getZCoordFromMesh(self.z_coord_ref3d)

        # ----- Equations
        self.eq_sw = mode2d.freeSurfaceEquations(
            self.mesh2d, self.W_2d, self.solution2d, self.bathymetry2d,
            self.uv_bottom2d, self.bottom_drag2d,
            nonlin=self.nonlin, use_wd=self.use_wd)
        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = mode3d.momentumEquation(
            self.mesh, self.U, self.U_scalar, bnd_markers,
            bnd_len, self.uv3d, self.eta3d,
            self.bathymetry3d, w=self.w3d,
            w_mesh=self.w_mesh3d,
            dw_mesh_dz=self.dw_mesh_dz_3d,
            viscosity_v=None,
            nonlin=self.nonlin)
        if self.solveSalt:
            self.eq_salt = mode3d.tracerEquation(
                self.mesh, self.H, self.salt3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.solveVertDiffusion:
            self.eq_vertmomentum = mode3d.verticalMomentumEquation(
                self.mesh, self.U, self.U_scalar, self.uv3d, w=None,
                viscosity_v=self.viscosity_v3d,
                uv_bottom=self.uv_bottom3d,
                bottom_drag=self.bottom_drag3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']

        # ----- Time integrators
        self.setTimeStep()
        self.timeStepper = timeIntegration.coupledSSPRK(self)

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.P1_2d),
            'elev3d': (self.eta3d, self.P1),
            'uv3d': (self.uv3d, self.U_visu),
            'w3d': (self.w3d, self.P1),
            'w3d_mesh': (self.w_mesh3d, self.P1),
            'salt3d': (self.salt3d, self.P1),
            'uv2d_dav': (self.uv2d_dav, self.U_visu_2d),
            'uv2d_bot': (self.uv_bottom2d, self.U_visu_2d),
            'nuv3d': (self.viscosity_v3d, self.P1),
            }
        self.exporter = exportManager(self.outputDir, self.fieldsToExport,
                                      exportFuncs, verbose=True)

        self._initialized = True

    def assingInitialConditions(self, elev=None, salt=None):
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
                                self.w_mesh_surf3d, self.dw_mesh_dz_3d,
                                self.bathymetry3d, self.z_coord_ref3d)

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
                        relcost = c/totcost
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
