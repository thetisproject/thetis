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
import weakref
from cofs.fieldDefs import fieldMetadata
from cofs.options import modelOptions


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
    def __init__(self, mesh2d, bathymetry2d, n_layers, order=1, mimetic=False,
                 options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.mesh = extrudeMeshSigma(mesh2d, n_layers, bathymetry2d)

        # Time integrator setup
        self.dt = None
        self.dt_2d = None
        self.M_modesplit = None

        # override default options
        self.options = modelOptions()
        self.options.update(options)

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}

        self.visualizationSpaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.fields = fieldDict()
        """Holds all functions needed by the solver object."""
        self.fields.bathymetry2d = bathymetry2d

    def setTimeStep(self):
        if self.options.useModeSplit:
            self.dt = self.options.dt
            if self.dt is None:
                mesh_dt = self.eq_sw.getTimeStepAdvection(Umag=self.options.uAdvection)
                dt = self.options.cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
                dt = comm.allreduce(dt, op=MPI.MIN)
                if round(dt) > 0:
                    dt = round(dt)
                self.dt = dt
            self.dt_2d = self.options.dt_2d
            if self.dt_2d is None:
                mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.options.uAdvection)
                dt_2d = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
                dt_2d = comm.allreduce(dt_2d, op=MPI.MIN)
                self.dt_2d = dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.options.uAdvection)
            dt_2d = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
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
        self.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0, name='P0')
        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1')
        self.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1v')
        self.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DG')
        self.P1DGv = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DGv')

        # Construct HDiv OuterProductElements
        # for horizontal velocity component
        Uh_elt = FiniteElement('RT', triangle, self.options.order+1)
        Uv_elt = FiniteElement('DG', interval, self.options.order)
        U_elt = HDiv(OuterProductElement(Uh_elt, Uv_elt))
        # for vertical velocity component
        Wh_elt = FiniteElement('DG', triangle, self.options.order)
        Wv_elt = FiniteElement('CG', interval, self.options.order+1)
        W_elt = HDiv(OuterProductElement(Wh_elt, Wv_elt))
        # in deformed mesh horiz. velocity must actually live in U + W
        UW_elt = EnrichedElement(U_elt, W_elt)
        # final spaces
        if self.options.mimetic:
            #self.U = FunctionSpace(self.mesh, UW_elt)  # uv
            self.U = FunctionSpace(self.mesh, U_elt, name='U')  # uv
            self.W = FunctionSpace(self.mesh, W_elt, name='W')  # w
        else:
            self.U = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                         vfamily='DG', vdegree=self.options.order,
                                         name='U')
            self.W = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                         vfamily='CG', vdegree=self.options.order + 1,
                                         name='W')
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.Uproj = self.U

        self.Uint = self.U  # vertical integral of uv
        # tracers
        self.H = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=max(0, self.options.order), name='H')
        # vertical integral of tracers
        self.Hint = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='CG', vdegree=self.options.order+1, name='Hint')
        # for scalar fields to be used in momentum eq NOTE could be omitted ?
        self.U_scalar = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=self.options.order, name='U_scalar')
        # for turbulence
        self.turb_space = self.P0
        # spaces for visualization
        self.visualizationSpaces[self.U] = self.P1v
        self.visualizationSpaces[self.H] = self.P1
        self.visualizationSpaces[self.W] = self.P1v
        self.visualizationSpaces[self.P0] = self.P1

        # 2D spaces
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        # 2D velocity space
        if self.options.mimetic:
            # NOTE this is not compatible with enriched UW space used in 3D
            self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        else:
            self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
        self.Uproj_2d = self.U_2d
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_scalar_2d')
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='H_2d')
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d], name='V_2d')
        self.visualizationSpaces[self.U_2d] = self.P1v_2d
        self.visualizationSpaces[self.H_2d] = self.P1_2d

        # ----- fields
        self.fields.solution2d = Function(self.V_2d, name='solution2d')
        # correct treatment of the split 2d functions
        uv2d, eta2d = self.fields.solution2d.split()
        self.fields.uv2d = uv2d
        self.fields.elev2d = eta2d
        self.visualizationSpaces[uv2d.function_space()] = self.P1v_2d
        self.visualizationSpaces[eta2d.function_space()] = self.P1_2d
        if self.options.useBottomFriction:
            self.fields.uv_bottom2d = Function(self.P1v_2d, name='Bottom Velocity')
            self.fields.z_bottom2d = Function(self.P1_2d, name='Bot. Vel. z coord')
            self.fields.bottom_drag2d = Function(self.P1_2d, name='Bottom Drag')

        self.fields.elev3d = Function(self.H, name='Elevation')
        self.fields.elev3dCG = Function(self.P1, name='Elevation')
        self.fields.bathymetry3d = Function(self.P1, name='Bathymetry')
        self.fields.uv3d = Function(self.U, name='Velocity')
        if self.options.useBottomFriction:
            self.fields.uv_bottom3d = Function(self.P1v, name='Bottom Velocity')
            self.fields.z_bottom3d = Function(self.P1, name='Bot. Vel. z coord')
            self.fields.bottom_drag3d = Function(self.P1, name='Bottom Drag')
        # z coordinate in the strecthed mesh
        self.fields.z_coord3d = Function(self.P1, name='z coord')
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref3d = Function(self.P1, name='ref z coord')
        self.fields.uvDav3d = Function(self.Uproj, name='Depth Averaged Velocity 3d')
        self.fields.uvDav2d = Function(self.Uproj_2d, name='Depth Averaged Velocity 2d')
        #self.fields.uv3d_tmp = Function(self.U, name='Velocity')
        self.fields.uv3d_mag = Function(self.P0, name='Velocity magnitude')
        self.fields.uv3d_P1 = Function(self.P1v, name='Smoothed Velocity')
        self.fields.w3d = Function(self.W, name='Vertical Velocity')
        if self.options.useALEMovingMesh:
            self.fields.w_mesh3d = Function(self.H, name='Vertical Velocity')
            self.fields.dw_mesh_dz_3d = Function(self.H, name='Vertical Velocity dz')
            self.fields.w_mesh_surf3d = Function(self.H, name='Vertical Velocity Surf')
            self.fields.w_mesh_surf2d = Function(self.H_2d, name='Vertical Velocity Surf')
        if self.options.solveSalt:
            self.fields.salt3d = Function(self.H, name='Salinity')
        if self.options.solveVertDiffusion and self.fields.useParabolicViscosity:
            # FIXME useParabolicViscosity is OBSOLETE
            self.fields.parabViscosity_v = Function(self.P1, name='Eddy viscosity')
        if self.options.baroclinic:
            self.fields.baroHead3d = Function(self.Hint, name='Baroclinic head')
            self.fields.baroHeadInt3d = Function(self.Hint, name='V.int. baroclinic head')
            self.fields.baroHead2d = Function(self.H_2d, name='DAv baroclinic head')
        if self.options.coriolis is not None:
            if isinstance(self.fields.coriolis, Constant):
                self.fields.coriolis3d = self.optoins.coriolis
            else:
                self.fields.coriolis3d = Function(self.P1, name='Coriolis parameter')
                copy2dFieldTo3d(self.optoins.coriolis, self.fields.coriolis3d)
        if self.options.wind_stress is not None:
            self.fields.wind_stress3d = Function(self.U_visu, name='Wind stress')
            copy2dFieldTo3d(self.options.wind_stress, self.fields.wind_stress3d)
        self.fields.vElemSize3d = Function(self.P1DG, name='element height')
        self.fields.vElemSize2d = Function(self.P1DG_2d, name='element height')
        self.fields.hElemSize3d = getHorzontalElemSize(self.P1_2d, self.P1)
        self.fields.maxHDiffusivity = Function(self.P1, name='Maximum h. Diffusivity')
        if self.options.smagorinskyFactor is not None:
            self.fields.smag_viscosity = Function(self.P1, name='Smagorinsky viscosity')
        if self.options.saltJumpDiffFactor is not None:
            self.fields.saltJumpDiff = Function(self.P1, name='Salt Jump Diffusivity')
        if self.options.useLimiterForTracers:
            self.tracerLimiter = limiter.vertexBasedP1DGLimiter(self.H,
                                                                self.P1,
                                                                self.P0)
        else:
            self.tracerLimiter = None
        if self.options.useTurbulence:
            # NOTE tke and psi should be in H as tracers ??
            self.fields.tke3d = Function(self.turb_space, name='Turbulent kinetic energy')
            self.fields.psi3d = Function(self.turb_space, name='Turbulence psi variable')
            # NOTE other turb. quantities should share the same nodes ??
            self.fields.epsilon3d = Function(self.turb_space, name='TKE dissipation rate')
            self.fields.len3d = Function(self.turb_space, name='Turbulent lenght scale')
            self.fields.eddyVisc_v = Function(self.turb_space, name='Vertical eddy viscosity')
            self.fields.eddyDiff_v = Function(self.turb_space, name='Vertical eddy diffusivity')
            # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
            self.fields.shearFreq2_3d = Function(self.turb_space, name='Shear frequency squared')
            self.fields.buoyancyFreq2_3d = Function(self.turb_space, name='Buoyancy frequency squared')
            glsParameters = {}  # use default parameters for now
            self.glsModel = turbulence.genericLengthScaleModel(weakref.proxy(self),
                self.tke3d, self.psi3d, self.uv3d_P1, self.len3d, self.epsilon3d,
                self.eddyDiff_v, self.eddyVisc_v,
                self.buoyancyFreq2_3d, self.shearFreq2_3d,
                **glsParameters)
        else:
            self.glsModel = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = sumFunction()
        self.tot_h_visc.add(self.fields.get('hViscosity'))
        self.tot_h_visc.add(self.fields.get('smag_viscosity'))
        self.tot_v_visc = sumFunction()
        self.tot_v_visc.add(self.fields.get('vViscosity'))
        self.tot_v_visc.add(self.fields.get('eddyVisc_v'))
        self.tot_v_visc.add(self.fields.get('parabViscosity_v'))
        self.tot_salt_h_diff = sumFunction()
        self.tot_salt_h_diff.add(self.fields.get('hDiffusivity'))
        self.tot_salt_v_diff = sumFunction()
        self.tot_salt_v_diff.add(self.fields.get('vDiffusivity'))
        self.tot_salt_v_diff.add(self.fields.get('eddyDiff_v'))

        # set initial values
        copy2dFieldTo3d(self.fields.bathymetry2d, self.fields.bathymetry3d)
        getZCoordFromMesh(self.fields.z_coord_ref3d)
        self.fields.z_coord3d.assign(self.fields.z_coord_ref3d)
        computeElemHeight(self.fields.z_coord3d, self.fields.vElemSize3d)
        copy3dFieldTo2d(self.fields.vElemSize3d, self.fields.vElemSize2d)

        # ----- Equations
        if self.options.useModeSplit:
            # full 2D shallow water equations
            self.eq_sw = module_2d.shallowWaterEquations(
                self.fields.solution2d, self.fields.bathymetry2d,
                self.fields.get('uv_bottom2d'), self.fields.get('bottom_drag2d'),
                baro_head=self.fields.get('baroHead2d'),
                viscosity_h=self.fields.get('hViscosity'),  # FIXME add 2d smag
                uvLaxFriedrichs=self.options.uvLaxFriedrichs,
                coriolis=self.options.coriolis,
                wind_stress=self.options.wind_stress,
                lin_drag=self.options.lin_drag,
                nonlin=self.options.nonlin)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.fields.solution2d.split()
            self.eq_sw = module_2d.freeSurfaceEquation(
                eta, uv, self.fields.bathymetry2d,
                nonlin=self.options.nonlin)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = module_3d.momentumEquation(
            bnd_markers,
            bnd_len, self.fields.uv3d, self.fields.elev3d,
            self.fields.bathymetry3d, w=self.fields.w3d,
            baro_head=self.fields.get('baroHead3d'),
            w_mesh=self.fields.get('w_mesh3d'),
            dw_mesh_dz=self.fields.get('dw_mesh_dz_3d'),
            viscosity_v=self.tot_v_visc.getSum(),
            viscosity_h=self.tot_h_visc.getSum(),
            laxFriedrichsFactor=self.options.uvLaxFriedrichs,
            #uvMag=self.uv3d_mag,
            uvP1=self.fields.get('uv3d_P1'),
            coriolis=self.fields.get('coriolis3d'),
            lin_drag=self.options.lin_drag,
            nonlin=self.options.nonlin)
        if self.options.solveSalt:
            self.eq_salt = tracerEquation.tracerEquation(
                self.fields.salt3d, self.fields.elev3d, self.fields.uv3d,
                w=self.fields.w3d, w_mesh=self.fields.get('w_mesh3d'),
                dw_mesh_dz=self.fields.get('dw_mesh_dz_3d'),
                diffusivity_h=self.tot_salt_h_diff.getSum(),
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                #uvMag=self.uv3d_mag,
                uvP1=self.fields.get('uv3d_P1'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.options.solveVertDiffusion:
            self.eq_vertmomentum = module_3d.verticalMomentumEquation(
                self.fields.uv3d, w=None,
                viscosity_v=self.tot_v_visc.getSum(),
                uv_bottom=self.fields.get('uv_bottom3d'),
                bottom_drag=self.fields.get('bottom_drag3d'),
                wind_stress=self.fields.get('wind_stress3d'),
                vElemSize=self.fields.vElemSize3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.useTurbulence:
            # explicit advection equations
            self.eq_tke_adv = tracerEquation.tracerEquation(
                self.fields.tke3d, self.fields.elev3d, self.fields.uv3d,
                w=self.fields.w3d, w_mesh=self.fields.get('w_mesh3d'),
                dw_mesh_dz=self.fields.get('dw_mesh_dz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.fields.get('uv3d_P1'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            self.eq_psi_adv = tracerEquation.tracerEquation(
                self.fields.psi3d, self.fields.elev3d, self.fields.uv3d,
                w=self.fields.w3d, w_mesh=self.fields.get('w_mesh3d'),
                dw_mesh_dz=self.fields.get('dw_mesh_dz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.fields.get('uv3d_P1'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.tkeEquation(
                self.fields.tke3d,
                self.fields.elev3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.fields.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)
            self.eq_psi_diff = turbulence.psiEquation(
                self.fields.psi3d, self.fields.elev3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.fields.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)

        # ----- Time integrators
        self.setTimeStep()
        if self.options.useModeSplit:
            if self.options.useIMEX:
                self.timeStepper = coupledTimeIntegrator.coupledSSPIMEX(weakref.proxy(self))
            elif self.options.useSemiImplicit2D:
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSemiImplicit(weakref.proxy(self))
            else:
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSync(weakref.proxy(self))
        else:
            self.timeStepper = coupledTimeIntegrator.coupledSSPRKSingleMode(weakref.proxy(self))
        printInfo('using {:} time integrator'.format(self.timeStepper.__class__.__name__))

        # compute maximal diffusivity for explicit schemes
        maxDiffAlpha = 1.0/100.0  # FIXME depends on element type and order
        self.fields.maxHDiffusivity.assign(maxDiffAlpha/self.dt * self.fields.hElemSize3d**2)

        # ----- File exporters
        # create exportManagers and store in a list
        self.exporters = {}
        e = exporter.exportManager(self.options.outputDir,
                                   self.options.fieldsToExport,
                                   self.fields,
                                   self.visualizationSpaces,
                                   fieldMetadata,
                                   exportType='vtk',
                                   verbose=self.options.verbose > 0)
        self.exporters['vtk'] = e
        numpyDir = os.path.join(self.options.outputDir, 'numpy')
        e = exporter.exportManager(numpyDir,
                                   self.options.fieldsToExportNumpy,
                                   self.fields,
                                   self.visualizationSpaces,
                                   fieldMetadata,
                                   exportType='numpy',
                                   verbose=self.options.verbose > 0)
        self.exporters['numpy'] = e

        self.uvP1_projector = projector(self.fields.uv3d, self.fields.uv3d_P1)
        #self.uvDAV_to_tmp_projector = projector(self.uv3d_dav, self.uv3d_tmp)
        #self.uv2d_to_DAV_projector = projector(self.fields.solution2d.split()[0],
                                               #self.uv2d_dav)
        #self.uv2dDAV_to_uv2d_projector = projector(self.uv2d_dav,
                                                   #self.fields.solution2d.split()[0])
        self.elev3d_to_CG_projector = projector(self.fields.elev3d, self.fields.elev3dCG)

        self._initialized = True

    def assignInitialConditions(self, elev=None, salt=None, uv2d=None):
        if not self._initialized:
            self.mightyCreator()
        if elev is not None:
            elev2d = self.fields.solution2d.split()[1]
            elev2d.project(elev)
            copy2dFieldTo3d(elev2d, self.fields.elev3d)
            self.fields.elev3dCG.project(self.fields.elev3d)
            if self.options.useALEMovingMesh:
                updateCoordinates(self.mesh, self.fields.elev3dCG, self.fields.bathymetry3d,
                                  self.fields.z_coord3d, self.fields.z_coord_ref3d)
                computeElemHeight(self.fields.z_coord3d, self.fields.vElemSize3d)
                copy3dFieldTo2d(self.fields.vElemSize3d, self.fields.vElemSize2d)
        if uv2d is not None:
            uv2d_field = self.fields.solution2d.split()[0]
            uv2d_field.project(uv2d)
            copy2dFieldTo3d(uv2d_field, self.fields.uv3d,
                            elemHeight=self.vElemSize3d)

        if salt is not None and self.options.solveSalt:
            self.fields.salt3d.project(salt)
        computeVertVelocity(self.fields.w3d, self.fields.uv3d, self.fields.bathymetry3d,
                            self.eq_momentum.boundary_markers,
                            self.eq_momentum.bnd_functions)
        if self.options.useALEMovingMesh:
            computeMeshVelocity(self.fields.elev3d, self.fields.uv3d, self.fields.w3d, self.fields.w_mesh3d,
                                self.fields.w_mesh_surf3d, self.fields.w_mesh_surf2d,
                                self.fields.dw_mesh_dz_3d,
                                self.fields.bathymetry3d, self.fields.z_coord_ref3d)
        if self.options.baroclinic:
            computeBaroclinicHead(self.fields.salt3d, self.fields.baroHead3d,
                                  self.fields.baroHead2d, self.fields.baroHeadInt3d,
                                  self.fields.bathymetry3d)

        self.timeStepper.initialize()

        self.options.checkSaltConservation *= self.options.solveSalt
        self.options.checkSaltDeviation *= self.options.solveSalt
        self.options.checkVolConservation3d *= self.options.useALEMovingMesh

    def export(self):
        for key in self.exporters:
            self.exporters[key].export()

    def iterate(self, updateForcings=None, updateForcings3d=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.options.TExport

        # initialize conservation checks
        if self.options.checkVolConservation2d:
            eta = self.fields.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.fields.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))
        if self.options.checkVolConservation3d:
            Vol3d_0 = compVolume3d(self.mesh)
            printInfo('Initial volume 3d {0:f}'.format(Vol3d_0))
        if self.options.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.salt3d)
            printInfo('Initial salt mass {0:f}'.format(Mass3d_0))
        if self.options.checkSaltDeviation:
            saltSum = self.fields.salt3d.dat.data.sum()
            saltSum = op2.MPI.COMM.allreduce(saltSum, op=MPI.SUM)
            nbNodes = self.fields.salt3d.dat.data.shape[0]
            nbNodes = op2.MPI.COMM.allreduce(nbNodes, op=MPI.SUM)
            saltVal = saltSum/nbNodes
            printInfo('Initial mean salt value {0:f}'.format(saltVal))
        if self.options.checkSaltOvershoot:
            saltMin0 = self.fields.salt3d.dat.data.min()
            saltMax0 = self.fields.salt3d.dat.data.max()
            saltMin0 = op2.MPI.COMM.allreduce(saltMin0, op=MPI.MIN)
            saltMax0 = op2.MPI.COMM.allreduce(saltMax0, op=MPI.MAX)
            printInfo('Initial salt value range {0:.3f}-{1:.3f}'.format(saltMin0, saltMax0))

        # initial export
        self.export()
        if exportFunc is not None:
            exportFunc()
        self.exporters['vtk'].exportBathymetry(self.fields.bathymetry2d)

        while t <= self.options.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, updateForcings,
                                     updateForcings3d)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.fields.solution2d.split()[1])
                norm_u = norm(self.fields.solution2d.split()[0])

                if self.options.checkVolConservation2d:
                    Vol2d = compVolume2d(self.fields.solution2d.split()[1],
                                         self.fields.bathymetry2d)
                if self.options.checkVolConservation3d:
                    Vol3d = compVolume3d(self.mesh)
                if self.options.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.fields.salt3d)
                if self.options.checkSaltDeviation:
                    saltMin = self.fields.salt3d.dat.data.min()
                    saltMax = self.fields.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if self.options.checkSaltOvershoot:
                    saltMin = self.fields.salt3d.dat.data.min()
                    saltMax = self.fields.salt3d.dat.data.max()
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
                    if self.options.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    if self.options.checkVolConservation3d:
                        print(line.format('vol 3d', (Vol3d_0 - Vol3d)/Vol3d_0))
                    if self.options.checkSaltConservation:
                        print(line.format('mass ',
                                          (Mass3d_0 - Mass3d)/Mass3d_0))
                    if self.options.checkSaltDeviation:
                        print('salt deviation {:g} {:g}'.format(*saltDev))
                    if self.options.checkSaltOvershoot:
                        print('salt overshoots {:g} {:g}'.format(*saltOversh))
                    sys.stdout.flush()

                self.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.options.TExport
                iExp += 1

                if commrank == 0 and len(self.options.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.options.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.options.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()


class flowSolver2d(object):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry2d, order=1, options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        options.setdefault('timeStepperType', 'SSPRK33')
        options.setdefault('timerLabels', ['mode2d'])
        options.setdefault('fieldsToExport', ['elev2d', 'uv2d'])

        # override default options
        self.options = modelOptions()
        self.options.update(options)

        self.visualizationSpaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.fields = fieldDict()
        """Holds all functions needed by the solver object."""
        self.fields.bathymetry2d = bathymetry2d

        self.bnd_functions = {'shallow_water': {}}

    def setTimeStep(self):
        mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.options.uAdvection)
        dt = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
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
        self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', max(self.options.order,1))
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.H_visu_2d = self.P1_2d
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.fields.solution2d = Function(self.V_2d, name='solution2d')

        # ----- Equations
        self.eq_sw = module_2d.shallowWaterEquations(
            self.fields.solution2d,
            self.fields.bathymetry2d,
            lin_drag=self.options.lin_drag,
            viscosity_h=self.fields.get('hViscosity'),
            uvLaxFriedrichs=self.options.uvLaxFriedrichs,
            coriolis=self.options.coriolis,
            wind_stress=self.options.wind_stress,
            nonlin=self.options.nonlin)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.options.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeIntegrator.SSPRK33Stage(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'ssprk33semi':
            self.timeStepper = timeIntegrator.SSPRK33StageSemiImplicit(self.eq_sw,
                                                            self.dt, self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeIntegrator.ForwardEuler(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeIntegrator.CrankNicolson(self.eq_sw, self.dt,
                                                             self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'sspimex':
            # TODO meaningful solver params
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
            self.timeStepper = timeIntegrator.SSPIMEX(self.eq_sw, self.dt,
                                                      solver_parameters=sp_expl,
                                                      solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timeStepperType))

        # ----- File exporters
        uv2d, eta2d = self.fields.solution2d.split()
        self.exporter = exporter.exportManager(self.options.outputDir,
                                               self.options.fieldsToExport,
                                               self.fields,
                                               self.visualizationSpaces,
                                               fieldMetadata,
                                               verbose=self.options.verbose > 0)
        self._initialized = True

    def assignInitialConditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.mightyCreator()
        uv2d, eta2d = self.fields.solution2d.split()
        if elev is not None:
            eta2d.project(elev)
        if uv_init is not None:
            uv2d.project(uv_init)

        self.timeStepper.initialize(self.fields.solution2d)

    def iterate(self, updateForcings=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.options.TExport

        # initialize conservation checks
        if self.options.checkVolConservation2d:
            eta = self.fields.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.fields.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.fields.bathymetry2d)

        while t <= self.options.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, self.fields.solution2d,
                                     updateForcings)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.fields.solution2d.split()[1])
                norm_u = norm(self.fields.solution2d.split()[0])

                if self.options.checkVolConservation2d:
                    Vol2d = compVolume2d(self.fields.solution2d.split()[1],
                                         self.fields.bathymetry2d)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.options.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.options.TExport
                iExp += 1

                if commrank == 0 and len(self.options.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.options.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.options.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
