"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from utility import *
import module_2d
import module_3d
import tracerEquation
import turbulence
import timeIntegrator as timeIntegrator
import coupledTimeIntegrator as coupledTimeIntegrator
import limiter
import time as timeMod
from mpi4py import MPI
import exporter
import ufl


class sumFunction(object):
    """
    Class to keep track of sum of Coefficients.
    """
    def __init__(self):
        """
        Initialize empty sum.

        get operation returns Constant(0)
        """
        self.coeffList = []

    def add(self, coeff):
        """
        Adds a coefficient to self
        """
        if coeff is None:
            return
        #classes = (Function, Constant, ufl.algebra.Sum, ufl.algebra.Product)
        #assert not isinstance(coeff, classes), \
            #('bad argument type: ' + str(type(coeff)))
        self.coeffList.append(coeff)

    def getSum(self):
        """
        Returns a sum of all added Coefficients
        """
        if len(self.coeffList) == 0:
            return None
        return sum(self.coeffList)


class flowSolver(object):
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
        self.solveSalt = True  # solve salt transport
        self.solveVertDiffusion = True  # solve implicit vert diffusion
        self.useBottomFriction = True  # apply log layer bottom stress
        self.useParabolicViscosity = False  # compute parabolic eddy viscosity
        self.useALEMovingMesh = True  # 3D mesh tracks free surface
        self.useModeSplit = True  # run 2D/3D modes with different dt
        self.useSemiImplicit2D = True  # implicit 2D waves (only w. mode split)
        self.useTurbulence = False  # GLS turbulence model
        self.useTurbulenceAdvection = False  # Advect tke,psi with velocity
        self.lin_drag = None  # 2D linear drag parameter tau/H/rho_0 = -drag*u
        self.hDiffusivity = None  # background diffusivity (set to Constant)
        self.vDiffusivity = None  # background diffusivity (set to Constant)
        self.hViscosity = None  # background viscosity (set to Constant)
        self.vViscosity = None  # background viscosity (set to Constant)
        self.coriolis = None  # Coriolis parameter (Constant or 2D Function)
        self.wind_stress = None  # stress at free surface (2D vector function)
        # NOTE 'baroclinic' means that salt3d field is treated as density [kg/m3]
        self.baroclinic = False  # comp and use internal pressure gradient
        self.smagorinskyFactor = None  # set to a Constant to use smag. visc.
        self.saltJumpDiffFactor = None  # set to a Constant to use nonlin diff.
        self.saltRange = Constant(30.0)  # value scale for salt to scale jumps
        self.useLimiterForTracers = False  # apply P1DG limiter
        self.uvLaxFriedrichs = Constant(1.0)  # scales uv stab. None omits
        self.tracerLaxFriedrichs = Constant(1.0)  # scales tracer stab. None omits
        self.checkVolConservation2d = False
        self.checkVolConservation3d = False
        self.checkSaltConservation = False
        self.checkSaltDeviation = False  # print deviation from mean of initial value
        self.checkSaltOvershoot = False  # print overshoots that exceed initial range  
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_eta3d',
                            'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                            'aux_mom_coupling',
                            'func_copy2dTo3d', 'func_copy3dTo2d',
                            'func_vert_int']
        self.outputDir = 'outputs'
        # list of fields to export in VTK format
        self.fieldsToExport = ['elev2d', 'uv2d', 'uv3d', 'w3d']
        # list of fields to export in numpy format
        self.fieldsToExportNumpy = []
        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}
        self.verbose = 0

    def setTimeStep(self):
        if self.useModeSplit:
            mesh_dt = self.eq_sw.getTimeStepAdvection(Umag=self.uAdvection)
            dt = self.cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
            dt = comm.allreduce(dt, op=MPI.MIN)
            if round(dt) > 0:
                dt = round(dt)
            if self.dt is None:
                self.dt = dt
            else:
                dt = float(self.dt)
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, op=MPI.MIN)
            if self.dt_2d is None:
                self.dt_2d = dt_2d
            self.M_modesplit = int(np.ceil(dt/self.dt_2d))
            self.dt_2d = dt/self.M_modesplit
        else:
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.uAdvection)
            dt_2d = self.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, op=MPI.MIN)
            if self.dt is None:
                self.dt = dt_2d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        printInfo('dt = {0:f}'.format(self.dt))
        printInfo('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
        sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0)
        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1)
        self.P1DGv = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1)

        # Construct HDiv OuterProductElements
        # for horizontal velocity component
        Uh_elt = FiniteElement('RT', triangle, self.order+1)
        Uv_elt = FiniteElement('DG', interval, self.order)
        U_elt = HDiv(OuterProductElement(Uh_elt, Uv_elt))
        # for vertical velocity component
        Wh_elt = FiniteElement('DG', triangle, self.order)
        Wv_elt = FiniteElement('CG', interval, self.order+1)
        W_elt = HDiv(OuterProductElement(Wh_elt, Wv_elt))
        # in deformed mesh horiz. velocity must actually live in U + W
        UW_elt = EnrichedElement(U_elt, W_elt)
        # final spaces
        self.U = FunctionSpace(self.mesh, UW_elt)  # uv
        self.W = FunctionSpace(self.mesh, W_elt)  # w
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.Uproj = VectorFunctionSpace(self.mesh, 'DG', self.order,
                                         vfamily='DG', vdegree=self.order)

        self.Uint = self.U  # vertical integral of uv
        # tracers
        self.H = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=max(0, self.order))
        # vertical integral of tracers
        self.Hint = FunctionSpace(self.mesh, 'DG', self.order, vfamily='CG', vdegree=self.order+1)
        # for scalar fields to be used in momentum eq NOTE could be omitted ? 
        self.U_scalar = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=self.order)
        # spaces for visualization
        self.U_visu = self.P1v
        self.H_visu = self.P1
        self.W_visu = self.P1v

        # 2D spaces
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        self.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        # 2D velocity space
        # NOTE this is not compatible with enriched UW space used in 3D
        self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.order+1)
        self.Uproj_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.order)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_visu_2d = self.P1_2d
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.solution2d = Function(self.V_2d, name='solution2d')
        if self.useBottomFriction:
            self.uv_bottom2d = Function(self.P1v_2d, name='Bottom Velocity')
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
            self.uv_bottom3d = Function(self.P1v, name='Bottom Velocity')
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
        self.uv3d_dav = Function(self.Uproj, name='Depth Averaged Velocity 3d')
        self.uv2d_dav = Function(self.Uproj_2d, name='Depth Averaged Velocity 2d')
        self.uv3d_tmp = Function(self.U, name='Velocity')
        self.uv3d_mag = Function(self.P0, name='Velocity magnitude')
        self.uv3d_P1 = Function(self.P1v, name='Smoothed Velocity')
        self.w3d = Function(self.W, name='Vertical Velocity')
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
            # FIXME useParabolicViscosity is OBSOLETE
            self.parabViscosity_v = Function(self.P1, name='Eddy viscosity')
        else:
            self.parabViscosity_v = None
        if self.baroclinic:
            self.baroHead3d = Function(self.Hint, name='Baroclinic head')
            self.baroHeadInt3d = Function(self.Hint, name='V.int. baroclinic head')
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
        self.maxHDiffusivity = Function(self.P1, name='Maximum h. Diffusivity')
        if self.smagorinskyFactor is not None:
            self.smag_viscosity = Function(self.P1, name='Smagorinsky viscosity')
        else:
            self.smag_viscosity = None
        if self.saltJumpDiffFactor is not None:
            self.saltJumpDiff = Function(self.P1, name='Salt Jump Diffusivity')
        else:
            self.saltJumpDiff = None
        if self.useLimiterForTracers:
            self.tracerLimiter = limiter.vertexBasedP1DGLimiter(self.H,
                                                                self.P1,
                                                                self.P0)
        else:
            self.tracerLimiter = None
        if self.useTurbulence:
            # NOTE tke and psi should be in H as tracers ??
            self.tke3d = Function(self.H, name='Turbulent kinetic energy')
            self.psi3d = Function(self.H, name='Turbulence psi variable')
            # NOTE other turb. quantities should share the same nodes ??
            self.epsilon3d = Function(self.H, name='TKE dissipation rate')
            self.len3d = Function(self.H, name='Turbulent lenght scale')
            self.eddyVisc_v = Function(self.H, name='Vertical eddy viscosity')
            self.eddyDiff_v = Function(self.H, name='Vertical eddy diffusivity')
            # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
            self.shearFreq2_3d = Function(self.H, name='Shear frequency squared')
            self.buoyancyFreq2_3d = Function(self.H, name='Buoyancy frequency squared')
            glsParameters = {}  # use default parameters for now
            self.glsModel = turbulence.genericLengthScaleModel(self,
                self.tke3d, self.psi3d, self.uv3d_P1, self.len3d, self.epsilon3d,
                self.eddyDiff_v, self.eddyVisc_v,
                self.buoyancyFreq2_3d, self.shearFreq2_3d,
                **glsParameters)
        else:
            self.tke3d = self.psi3d = self.epsilon3d = self.len3d = None
            self.eddyVisc_v = self.eddyDiff_v = None
            self.shearFreq2_3d = self.buoyancyFreq2_3d = None
            self.glsModel = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = sumFunction()
        self.tot_h_visc.add(self.hViscosity)
        self.tot_h_visc.add(self.smag_viscosity)
        self.tot_v_visc = sumFunction()
        self.tot_v_visc.add(self.vViscosity)
        self.tot_v_visc.add(self.eddyVisc_v)
        self.tot_v_visc.add(self.parabViscosity_v)
        self.tot_salt_h_diff = sumFunction()
        self.tot_salt_h_diff.add(self.hDiffusivity)
        self.tot_salt_v_diff = sumFunction()
        self.tot_salt_v_diff.add(self.vDiffusivity)
        self.tot_salt_v_diff.add(self.eddyDiff_v)

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
                self.mesh2d, self.V_2d, self.solution2d, self.bathymetry2d,
                self.uv_bottom2d, self.bottom_drag2d,
                baro_head=self.baroHead2d,
                viscosity_h=self.hViscosity,  # FIXME add 2d smag
                uvLaxFriedrichs=self.uvLaxFriedrichs,
                coriolis=self.coriolis,
                wind_stress=self.wind_stress,
                lin_drag=self.lin_drag,
                nonlin=self.nonlin)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.solution2d.split()
            self.eq_sw = module_2d.freeSurfaceEquation(
                self.mesh2d, self.H_2d, eta, uv, self.bathymetry2d,
                nonlin=self.nonlin)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = module_3d.momentumEquation(
            self.mesh, self.U, self.U_scalar, bnd_markers,
            bnd_len, self.uv3d, self.eta3d,
            self.bathymetry3d, w=self.w3d,
            baro_head=self.baroHead3d,
            w_mesh=self.w_mesh3d,
            dw_mesh_dz=self.dw_mesh_dz_3d,
            viscosity_v=self.tot_v_visc.getSum(),
            viscosity_h=self.tot_h_visc.getSum(),
            laxFriedrichsFactor=self.uvLaxFriedrichs,
            #uvMag=self.uv3d_mag,
            uvP1=self.uv3d_P1,
            coriolis=self.coriolis3d,
            lin_drag=self.lin_drag,
            nonlin=self.nonlin)
        if self.solveSalt:
            self.eq_salt = tracerEquation.tracerEquation(
                self.mesh, self.H, self.salt3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                diffusivity_h=self.tot_salt_h_diff.getSum(),
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                #uvMag=self.uv3d_mag,
                uvP1=self.uv3d_P1,
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.solveVertDiffusion:
            self.eq_vertmomentum = module_3d.verticalMomentumEquation(
                self.mesh, self.U, self.U_scalar, self.uv3d, w=None,
                viscosity_v=self.tot_v_visc.getSum(),
                uv_bottom=self.uv_bottom3d,
                bottom_drag=self.bottom_drag3d,
                wind_stress=self.wind_stress3d,
                vElemSize=self.vElemSize3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.useTurbulence:
            # explicit advection equations
            self.eq_tke_adv = tracerEquation.tracerEquation(
                self.mesh, self.H, self.tke3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.uv3d_P1,
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            self.eq_psi_adv = tracerEquation.tracerEquation(
                self.mesh, self.H, self.psi3d, self.eta3d, self.uv3d,
                w=self.w3d, w_mesh=self.w_mesh3d,
                dw_mesh_dz=self.dw_mesh_dz_3d,
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.uv3d_P1,
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.tkeEquation(
                self.mesh, self.H, self.tke3d, self.eta3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)
            self.eq_psi_diff = turbulence.psiEquation(
                self.mesh, self.H, self.psi3d, self.eta3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)

        # ----- Time integrators
        self.setTimeStep()
        if self.useModeSplit:
            if self.useSemiImplicit2D:
                printInfo('using coupledSSPRKSemiImplicit time integrator')
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSemiImplicit(self)
            else:
                printInfo('using coupledSSPRKSync time integrator')
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSync(self)
        else:
            printInfo('using coupledSSPRKSingleMode time integrator')
            self.timeStepper = coupledTimeIntegrator.coupledSSPRKSingleMode(self)

        # compute maximal diffusivity for explicit schemes
        maxDiffAlpha = 1.0/100.0  # FIXME depends on element type and order
        self.maxHDiffusivity.assign(maxDiffAlpha/self.dt * self.hElemSize3d**2)

        # ----- File exporters
        uv2d, eta2d = self.solution2d.split()
        # dictionary of all exportable functions and their visualization space
        exportFuncs = {
            'uv2d': (uv2d, self.U_visu_2d),
            'elev2d': (eta2d, self.H_visu_2d),
            'elev3d': (self.eta3d, self.H_visu),
            'uv3d': (self.uv3d, self.U_visu),
            'uv3d_dav': (self.uv3d_dav, self.U_visu),
            'w3d': (self.w3d, self.W_visu),
            'w3d_mesh': (self.w_mesh3d, self.P1),
            'salt3d': (self.salt3d, self.H_visu),
            'uv2d_dav': (self.uv2d_dav, self.U_visu_2d),
            'uv2d_bot': (self.uv_bottom2d, self.U_visu_2d),
            'parabNuv3d': (self.parabViscosity_v, self.P1),
            'eddyNuv3d': (self.eddyVisc_v, self.P1DG),
            'shearFreq3d': (self.shearFreq2_3d, self.P1DG),
            'tke3d': (self.tke3d, self.P1DG),
            'psi3d': (self.psi3d, self.P1DG),
            'eps3d': (self.epsilon3d, self.P1DG),
            'len3d': (self.len3d, self.P1DG),
            'barohead3d': (self.baroHead3d, self.P1),
            'barohead2d': (self.baroHead2d, self.P1_2d),
            'smagViscosity': (self.smag_viscosity, self.P1),
            'saltJumpDiff': (self.saltJumpDiff, self.P1),
            }
        # create exportManagers and store in a list
        self.exporters = {}
        e = exporter.exportManager(self.outputDir,
                                   self.fieldsToExport,
                                   exportFuncs,
                                   exportType='vtk',
                                   verbose=self.verbose > 0)
        self.exporters['vtk'] = e
        numpyDir = os.path.join(self.outputDir, 'numpy')
        e = exporter.exportManager(numpyDir,
                                   self.fieldsToExportNumpy,
                                   exportFuncs,
                                   exportType='numpy',
                                   verbose=self.verbose > 0)
        self.exporters['numpy'] = e

        self.uvP1_projector = projector(self.uv3d, self.uv3d_P1)
        self.uvDAV_to_tmp_projector = projector(self.uv3d_dav, self.uv3d_tmp)
        self.uv2d_to_DAV_projector = projector(self.solution2d.split()[0],
                                               self.uv2d_dav)
        self.uv2dDAV_to_uv2d_projector = projector(self.uv2d_dav,
                                                   self.solution2d.split()[0])
        self.eta3d_to_CG_projector = projector(self.eta3d, self.eta3dCG)

        self._initialized = True

    def assignInitialConditions(self, elev=None, salt=None, uv2d=None):
        if not self._initialized:
            self.mightyCreator()
        if elev is not None:
            eta2d = self.solution2d.split()[1]
            eta2d.project(elev)
            copy2dFieldTo3d(eta2d, self.eta3d)
            self.eta3dCG.project(self.eta3d)
            if self.useALEMovingMesh:
                updateCoordinates(self.mesh, self.eta3dCG, self.bathymetry3d,
                                  self.z_coord3d, self.z_coord_ref3d)
                computeElemHeight(self.z_coord3d, self.vElemSize3d)
                copy3dFieldTo2d(self.vElemSize3d, self.vElemSize2d)
        if uv2d is not None:
            uv2d_field = self.solution2d.split()[0]
            uv2d_field.project(uv2d)
            copy2dFieldTo3d(uv2d_field, self.uv3d,
                            elemHeight=self.vElemSize3d)

        if salt is not None and self.solveSalt:
            self.salt3d.project(salt)
        computeVertVelocity(self.w3d, self.uv3d, self.bathymetry3d,
                            self.eq_momentum.boundary_markers,
                            self.eq_momentum.bnd_functions)
        if self.useALEMovingMesh:
            computeMeshVelocity(self.eta3d, self.uv3d, self.w3d, self.w_mesh3d,
                                self.w_mesh_surf3d, self.w_mesh_surf2d,
                                self.dw_mesh_dz_3d,
                                self.bathymetry3d, self.z_coord_ref3d)
        if self.baroclinic:
            computeBaroclinicHead(self.salt3d, self.baroHead3d,
                                  self.baroHead2d, self.baroHeadInt3d,
                                  self.bathymetry3d)

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
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))
        if self.checkVolConservation3d:
            Vol3d_0 = compVolume3d(self.mesh)
            printInfo('Initial volume 3d {0:f}'.format(Vol3d_0))
        if self.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.salt3d)
            printInfo('Initial salt mass {0:f}'.format(Mass3d_0))
        if self.checkSaltDeviation:
            saltSum = self.salt3d.dat.data.sum()
            saltSum = op2.MPI.COMM.allreduce(saltSum, op=MPI.SUM)
            nbNodes = self.salt3d.dat.data.shape[0]
            nbNodes = op2.MPI.COMM.allreduce(nbNodes, op=MPI.SUM)
            saltVal = saltSum/nbNodes
            printInfo('Initial mean salt value {0:f}'.format(saltVal))
        if self.checkSaltOvershoot:
            saltMin0 = self.salt3d.dat.data.min()
            saltMax0 = self.salt3d.dat.data.max()
            saltMin0 = op2.MPI.COMM.allreduce(saltMin0, op=MPI.MIN)
            saltMax0 = op2.MPI.COMM.allreduce(saltMax0, op=MPI.MAX)
            printInfo('Initial salt value range {0:.3f}-{1:.3f}'.format(saltMin0, saltMax0))

        # initial export
        for key in self.exporters:
            self.exporters[key].export()
        if exportFunc is not None:
            exportFunc()
        self.exporters['vtk'].exportBathymetry(self.bathymetry2d)

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
                                         self.bathymetry2d)
                if self.checkVolConservation3d:
                    Vol3d = compVolume3d(self.mesh)
                if self.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.salt3d)
                if self.checkSaltDeviation:
                    saltMin = self.salt3d.dat.data.min()
                    saltMax = self.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if self.checkSaltOvershoot:
                    saltMin = self.salt3d.dat.data.min()
                    saltMax = self.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    overshoot = max(saltMax-saltMax0, 0.0)
                    undershoot = min(saltMin-saltMin0, 0.0)
                    saltOversh = (undershoot, overshoot)
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
                    if self.checkSaltOvershoot:
                        print('salt overshoots {:g} {:g}'.format(*saltOversh))
                    sys.stdout.flush()

                for key in self.exporters:
                    self.exporters[key].export()
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
        dt = comm.allreduce(dt, op=MPI.MIN)
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
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', max(self.order,1))
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.order)
        self.H_visu_2d = self.P1_2d
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.solution2d = Function(self.V_2d, name='solution2d')

        # ----- Equations
        self.eq_sw = module_2d.shallowWaterEquations(
            self.mesh2d, self.V_2d, self.solution2d, self.bathymetry2d,
            lin_drag=self.lin_drag,
            viscosity_h=self.hViscosity,
            uvLaxFriedrichs=self.uvLaxFriedrichs,
            coriolis=self.coriolis,
            wind_stress=self.wind_stress,
            nonlin=self.nonlin)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeIntegrator.SSPRK33Stage(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'ssprk33semi':
            self.timeStepper = timeIntegrator.SSPRK33StageSemiImplicit(self.eq_sw,
                                                            self.dt, self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeIntegrator.ForwardEuler(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeIntegrator.CrankNicolson(self.eq_sw, self.dt,
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
        self.exporter = exporter.exportManager(self.outputDir, self.fieldsToExport,
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
        if self.checkVolConservation2d:
            eta = self.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))

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
                                       self.bathymetry2d)
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
