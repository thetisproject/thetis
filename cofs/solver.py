"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from utility import *
import shallowWaterEq
import momentumEquation
import tracerEquation
import turbulence
import coupledTimeIntegrator as coupledTimeIntegrator
import limiter
import time as timeMod
from mpi4py import MPI
import exporter
import weakref
from cofs.fieldDefs import fieldMetadata
from cofs.options import modelOptions


class flowSolver(FrozenClass):
    """Creates and solves coupled 2D-3D equations"""
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.mesh = extrudeMeshSigma(mesh2d, n_layers, bathymetry_2d)

        # Time integrator setup
        self.dt = None
        self.dt_2d = None
        self.M_modesplit = None

        # override default options
        self.options = modelOptions()
        self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.iExport = 1

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}

        self.visu_spaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.fields = fieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.fields.bathymetry_2d = bathymetry_2d
        self._isfrozen = True  # disallow creating new attributes

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

    def createFunctionSpaces(self):
        """Creates function spaces"""
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0, name='P0')
        self.function_spaces.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1')
        self.function_spaces.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1v')
        self.function_spaces.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DG')
        self.function_spaces.P1DGv = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DGv')

        # Construct HDiv OuterProductElements
        # for horizontal velocity component
        Uh_elt = FiniteElement('RT', triangle, self.options.order+1)
        Uv_elt = FiniteElement('DG', interval, self.options.order)
        U_elt = HDiv(OuterProductElement(Uh_elt, Uv_elt))
        # for vertical velocity component
        Wh_elt = FiniteElement('DG', triangle, self.options.order)
        Wv_elt = FiniteElement('CG', interval, self.options.order+1)
        W_elt = HDiv(OuterProductElement(Wh_elt, Wv_elt))
        # final spaces
        if self.options.mimetic:
            # self.U = FunctionSpace(self.mesh, UW_elt)  # uv
            self.function_spaces.U = FunctionSpace(self.mesh, U_elt, name='U')  # uv
            self.function_spaces.W = FunctionSpace(self.mesh, W_elt, name='W')  # w
        else:
            self.function_spaces.U = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='DG', vdegree=self.options.order,
                                                         name='U')
            # TODO should this be P(n-1)DG x P(n+1) ?
            self.function_spaces.W = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='CG', vdegree=self.options.order + 1,
                                                         name='W')
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.function_spaces.Uproj = self.function_spaces.U

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=max(0, self.options.order), name='H')
        # vertical integral of tracers
        self.function_spaces.Hint = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='CG', vdegree=self.options.order+1, name='Hint')
        # for scalar fields to be used in momentum eq NOTE could be omitted ?
        # self.function_spaces.U_scalar = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=self.options.order, name='U_scalar')
        # for turbulence
        self.function_spaces.turb_space = self.function_spaces.P0
        # spaces for visualization
        self.visu_spaces[self.function_spaces.U] = self.function_spaces.P1v
        self.visu_spaces[self.function_spaces.H] = self.function_spaces.P1
        self.visu_spaces[self.function_spaces.Hint] = self.function_spaces.P1
        self.visu_spaces[self.function_spaces.W] = self.function_spaces.P1v
        self.visu_spaces[self.function_spaces.P0] = self.function_spaces.P1
        self.visu_spaces[self.function_spaces.P1] = self.function_spaces.P1
        self.visu_spaces[self.function_spaces.P1DG] = self.function_spaces.P1

        # 2D spaces
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        # 2D velocity space
        if self.options.mimetic:
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        else:
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
        self.function_spaces.Uproj_2d = self.function_spaces.U_2d
        # TODO is this needed?
        # self.function_spaces.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_scalar_2d')
        self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='H_2d')
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d], name='V_2d')
        self.visu_spaces[self.function_spaces.U_2d] = self.function_spaces.P1v_2d
        self.visu_spaces[self.function_spaces.H_2d] = self.function_spaces.P1_2d
        self.visu_spaces[self.function_spaces.P1v_2d] = self.function_spaces.P1v_2d
        self._isfrozen = True

    def createEquations(self):
        """Creates function spaces, functions, equations and time steppers."""
        if not hasattr(self, 'U_2d'):
            self.createFunctionSpaces()
        self._isfrozen = False

        # ----- fields
        self.fields.solution2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        self.visu_spaces[uv_2d.function_space()] = self.function_spaces.P1v_2d
        self.visu_spaces[eta2d.function_space()] = self.function_spaces.P1_2d
        if self.options.useBottomFriction:
            self.fields.uv_bottom_2d = Function(self.function_spaces.P1v_2d)
            self.fields.z_bottom_2d = Function(self.function_spaces.P1_2d)
            self.fields.bottom_drag_2d = Function(self.function_spaces.P1_2d)

        self.fields.elev_3d = Function(self.function_spaces.H)
        self.fields.elev_cg_3d = Function(self.function_spaces.P1)
        self.fields.bathymetry_3d = Function(self.function_spaces.P1)
        self.fields.uv_3d = Function(self.function_spaces.U)
        if self.options.useBottomFriction:
            self.fields.uv_bottom_3d = Function(self.function_spaces.P1v)
            self.fields.bottom_drag_3d = Function(self.function_spaces.P1)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(self.function_spaces.P1)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(self.function_spaces.P1)
        self.fields.uv_dav_3d = Function(self.function_spaces.Uproj)
        self.fields.uv_dav_2d = Function(self.function_spaces.Uproj_2d)
        self.fields.uv_mag_3d = Function(self.function_spaces.P0)
        self.fields.uv_p1_3d = Function(self.function_spaces.P1v)
        self.fields.w_3d = Function(self.function_spaces.W)
        if self.options.useALEMovingMesh:
            self.fields.w_mesh_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_ddz_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_surf_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_surf_2d = Function(self.function_spaces.H_2d)
        if self.options.solveSalt:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solveVertDiffusion and self.options.useParabolicViscosity:
            # FIXME useParabolicViscosity is OBSOLETE
            self.fields.parab_visc_3d = Function(self.function_spaces.P1)
        if self.options.baroclinic:
            self.fields.baroc_head_3d = Function(self.function_spaces.Hint)
            self.fields.baroc_head_int_3d = Function(self.function_spaces.Hint)
            self.fields.baroc_head_2d = Function(self.function_spaces.H_2d)
        if self.options.coriolis is not None:
            if isinstance(self.options.coriolis, Constant):
                self.fields.coriolis_3d = self.options.coriolis
            else:
                self.fields.coriolis_3d = Function(self.function_spaces.P1)
                expandFunctionTo3d(self.options.coriolis, self.fields.coriolis_3d).solve()
        if self.options.wind_stress is not None:
            self.fields.wind_stress_3d = Function(self.function_spaces.P1)
            expandFunctionTo3d(self.options.wind_stress, self.fields.wind_stress_3d).solve()
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        getHorizontalElemSize(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.smagorinskyFactor is not None:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.salt_jump_diffFactor is not None:
            self.fields.salt_jump_diff = Function(self.function_spaces.P1)
        if self.options.useLimiterForTracers:
            self.tracerLimiter = limiter.vertexBasedP1DGLimiter(self.function_spaces.H,
                                                                self.function_spaces.P1,
                                                                self.function_spaces.P0)
        else:
            self.tracerLimiter = None
        if self.options.useTurbulence:
            # NOTE tke and psi should be in H as tracers ??
            self.fields.tke_3d = Function(self.function_spaces.turb_space)
            self.fields.psi_3d = Function(self.function_spaces.turb_space)
            # NOTE other turb. quantities should share the same nodes ??
            self.fields.eps_3d = Function(self.function_spaces.turb_space)
            self.fields.len_3d = Function(self.function_spaces.turb_space)
            self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
            self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
            # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
            self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
            self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
            glsParameters = {}  # use default parameters for now
            self.glsModel = turbulence.genericLengthScaleModel(weakref.proxy(self),
                                                               self.fields.tke_3d,
                                                               self.fields.psi_3d,
                                                               self.fields.uv_p1_3d,
                                                               self.fields.len_3d,
                                                               self.fields.eps_3d,
                                                               self.fields.eddy_diff_3d,
                                                               self.fields.eddy_visc_3d,
                                                               self.fields.buoy_freq_3d,
                                                               self.fields.shear_freq_3d,
                                                               **glsParameters)
        else:
            self.glsModel = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = SumFunction()
        self.tot_h_visc.add(self.options.get('hViscosity'))
        self.tot_h_visc.add(self.fields.get('smag_visc_3d'))
        self.tot_v_visc = SumFunction()
        self.tot_v_visc.add(self.options.get('vViscosity'))
        self.tot_v_visc.add(self.fields.get('eddy_visc_3d'))
        self.tot_v_visc.add(self.fields.get('parab_visc_3d'))
        self.tot_salt_h_diff = SumFunction()
        self.tot_salt_h_diff.add(self.options.get('hDiffusivity'))
        self.tot_salt_v_diff = SumFunction()
        self.tot_salt_v_diff.add(self.options.get('vDiffusivity'))
        self.tot_salt_v_diff.add(self.fields.get('eddy_diff_3d'))

        # ----- Equations
        if self.options.useModeSplit:
            # full 2D shallow water equations
            self.eq_sw = shallowWaterEq.ShallowWaterEquations(
                self.fields.solution2d, self.fields.bathymetry_2d,
                self.fields.get('uv_bottom_2d'), self.fields.get('bottom_drag_2d'),
                baroc_head=self.fields.get('baroc_head_2d'),
                viscosity_h=self.options.get('hViscosity'),  # FIXME add 2d smag
                uvLaxFriedrichs=self.options.uvLaxFriedrichs,
                coriolis=self.options.coriolis,
                wind_stress=self.options.wind_stress,
                uv_source=self.options.uv_source_2d,
                lin_drag=self.options.lin_drag,
                nonlin=self.options.nonlin)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.fields.solution2d.split()
            self.eq_sw = shallowWaterEq.FreeSurfaceEquation(
                eta, uv, self.fields.bathymetry_2d,
                nonlin=self.options.nonlin)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = momentumEquation.MomentumEquation(
            bnd_markers,
            bnd_len, self.fields.uv_3d, self.fields.elev_3d,
            self.fields.bathymetry_3d, w=self.fields.w_3d,
            baroc_head=self.fields.get('baroc_head_3d'),
            w_mesh=self.fields.get('w_mesh_3d'),
            dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
            viscosity_v=self.tot_v_visc.get_sum(),
            viscosity_h=self.tot_h_visc.get_sum(),
            laxFriedrichsFactor=self.options.uvLaxFriedrichs,
            # uvMag=self.uv_mag_3d,
            uvP1=self.fields.get('uv_p1_3d'),
            coriolis=self.fields.get('coriolis_3d'),
            source=self.options.uv_source_3d,
            lin_drag=self.options.lin_drag,
            nonlin=self.options.nonlin)
        if self.options.solveSalt:
            self.eq_salt = tracerEquation.TracerEquation(
                self.fields.salt_3d, self.fields.elev_3d, self.fields.uv_3d,
                w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
                dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
                diffusivity_h=self.tot_salt_h_diff.get_sum(),
                diffusivity_v=self.tot_salt_v_diff.get_sum(),
                source=self.options.salt_source_3d,
                # uvMag=self.uv_mag_3d,
                uvP1=self.fields.get('uv_p1_3d'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.v_elem_size_3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.options.solveVertDiffusion:
            self.eq_vertmomentum = momentumEquation.VerticalMomentumEquation(
                self.fields.uv_3d, w=None,
                viscosity_v=self.tot_v_visc.get_sum(),
                uv_bottom=self.fields.get('uv_bottom_3d'),
                bottom_drag=self.fields.get('bottom_drag_3d'),
                wind_stress=self.fields.get('wind_stress_3d'),
                vElemSize=self.fields.v_elem_size_3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.useTurbulence:
            # explicit advection equations
            self.eq_tke_adv = tracerEquation.TracerEquation(
                self.fields.tke_3d, self.fields.elev_3d, self.fields.uv_3d,
                w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
                dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.fields.get('uv_p1_3d'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.v_elem_size_3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            self.eq_psi_adv = tracerEquation.TracerEquation(
                self.fields.psi_3d, self.fields.elev_3d, self.fields.uv_3d,
                w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
                dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.fields.get('uv_p1_3d'),
                laxFriedrichsFactor=self.options.tracerLaxFriedrichs,
                vElemSize=self.fields.v_elem_size_3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.TKEEquation(
                self.fields.tke_3d,
                self.fields.elev_3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.get_sum(),
                viscosity_v=self.tot_v_visc.get_sum(),
                vElemSize=self.fields.v_elem_size_3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)
            self.eq_psi_diff = turbulence.PsiEquation(
                self.fields.psi_3d, self.fields.elev_3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.get_sum(),
                viscosity_v=self.tot_v_visc.get_sum(),
                vElemSize=self.fields.v_elem_size_3d,
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
        self.fields.max_h_diff.assign(maxDiffAlpha/self.dt * self.fields.h_elem_size_3d**2)

        # ----- File exporters
        # create exportManagers and store in a list
        self.exporters = {}
        e = exporter.exportManager(self.options.outputDir,
                                   self.options.fieldsToExport,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   exportType='vtk',
                                   verbose=self.options.verbose > 0)
        self.exporters['vtk'] = e
        numpyDir = os.path.join(self.options.outputDir, 'numpy')
        e = exporter.exportManager(numpyDir,
                                   self.options.fieldsToExportNumpy,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   exportType='numpy',
                                   verbose=self.options.verbose > 0)
        self.exporters['numpy'] = e
        hdf5Dir = os.path.join(self.options.outputDir, 'hdf5')
        e = exporter.exportManager(hdf5Dir,
                                   self.options.fieldsToExportHDF5,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   exportType='hdf5',
                                   verbose=self.options.verbose > 0)
        self.exporters['hdf5'] = e

        # ----- Operators
        self.wSolver = verticalVelocitySolver(self.fields.w_3d,
                                              self.fields.uv_3d,
                                              self.fields.bathymetry_3d,
                                              self.eq_momentum.boundary_markers,
                                              self.eq_momentum.bnd_functions)
        # NOTE averager is a word. now.
        self.uvAverager = verticalIntegrator(self.fields.uv_3d,
                                             self.fields.uv_dav_3d,
                                             bottomToTop=True,
                                             bndValue=Constant((0.0, 0.0, 0.0)),
                                             average=True,
                                             bathymetry=self.fields.bathymetry_3d)
        if self.options.baroclinic:
            self.rhoIntegrator = verticalIntegrator(self.fields.salt_3d,
                                                    self.fields.baroc_head_3d,
                                                    bottomToTop=False)
            self.baroHeadAverager = verticalIntegrator(self.fields.baroc_head_3d,
                                                       self.fields.baroc_head_int_3d,
                                                       bottomToTop=True,
                                                       average=True,
                                                       bathymetry=self.fields.bathymetry_3d)
            self.extractSurfBaroHead = subFunctionExtractor(self.fields.baroc_head_int_3d,
                                                            self.fields.baroc_head_2d,
                                                            useBottomValue=False)

        self.extractSurfDavUV = subFunctionExtractor(self.fields.uv_dav_3d,
                                                     self.fields.uv_dav_2d,
                                                     useBottomValue=False,
                                                     elemHeight=self.fields.v_elem_size_2d)
        self.copyVElemSizeTo2d = subFunctionExtractor(self.fields.v_elem_size_3d,
                                                      self.fields.v_elem_size_2d)
        self.copyElevTo3d = expandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
        self.copyUVDavToUVDav3d = expandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                                                     elemHeight=self.fields.v_elem_size_3d)
        self.copyUVToUVDav3d = expandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                  elemHeight=self.fields.v_elem_size_3d)
        self.uvMagSolver = velocityMagnitudeSolver(self.fields.uv_mag_3d, u=self.fields.uv_3d)
        if self.options.useBottomFriction:
            self.extractUVBottom = subFunctionExtractor(self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                                                        useBottomValue=True, elemBottomNodes=False,
                                                        elemHeight=self.fields.v_elem_size_2d)
            self.extractZBottom = subFunctionExtractor(self.fields.z_coord_3d, self.fields.z_bottom_2d,
                                                       useBottomValue=True, elemBottomNodes=False,
                                                       elemHeight=self.fields.v_elem_size_2d)
            self.copyUVBottomTo3d = expandFunctionTo3d(self.fields.uv_bottom_2d,
                                                       self.fields.uv_bottom_3d,
                                                       elemHeight=self.fields.v_elem_size_3d)
            self.copyBottomDragTo3d = expandFunctionTo3d(self.fields.bottom_drag_2d,
                                                         self.fields.bottom_drag_3d,
                                                         elemHeight=self.fields.v_elem_size_3d)
        if self.options.useALEMovingMesh:
            self.meshCoordUpdater = ALEMeshCoordinateUpdater(self.mesh,
                                                             self.fields.elev_3d,
                                                             self.fields.bathymetry_3d,
                                                             self.fields.z_coord_3d,
                                                             self.fields.z_coord_ref_3d)
            self.extractSurfW = subFunctionExtractor(self.fields.w_mesh_surf_3d,
                                                     self.fields.w_mesh_surf_2d,
                                                     useBottomValue=False)
            self.copySurfWMeshTo3d = expandFunctionTo3d(self.fields.w_mesh_surf_2d,
                                                        self.fields.w_mesh_surf_3d)
            self.wMeshSolver = meshVelocitySolver(self, self.fields.elev_3d,
                                                  self.fields.uv_3d,
                                                  self.fields.w_3d,
                                                  self.fields.w_mesh_3d,
                                                  self.fields.w_mesh_surf_3d,
                                                  self.fields.w_mesh_surf_2d,
                                                  self.fields.w_mesh_ddz_3d,
                                                  self.fields.bathymetry_3d,
                                                  self.fields.z_coord_ref_3d)

        if self.options.salt_jump_diffFactor is not None:
            self.horizJumpDiffSolver = horizontalJumpDiffusivity(self.options.salt_jump_diffFactor, self.fields.salt_3d,
                                                                 self.fields.salt_jump_diff, self.fields.h_elem_size_3d,
                                                                 self.fields.uv_mag_3d, self.options.saltRange,
                                                                 self.fields.max_h_diff)
        if self.options.smagorinskyFactor is not None:
            self.smagorinskyDiffSolver = smagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                              self.options.smagorinskyFactor, self.fields.h_elem_size_3d)
        if self.options.useParabolicViscosity:
            self.parabolicViscositySolver = parabolicViscosity(self.fields.uv_bottom_3d,
                                                               self.fields.bottom_drag_3d,
                                                               self.fields.bathymetry_3d,
                                                               self.fields.parab_visc_3d)
        self.uvP1_projector = projector(self.fields.uv_3d, self.fields.uv_p1_3d)
        # self.uvDAV_to_tmp_projector = projector(self.uv_dav_3d, self.uv_3d_tmp)
        # self.uv_2d_to_DAV_projector = projector(self.fields.solution2d.split()[0],
        #                                         self.uv_dav_2d)
        # self.uv_2dDAV_to_uv_2d_projector = projector(self.uv_dav_2d,
        #                                              self.fields.solution2d.split()[0])
        self.elev_3d_to_CG_projector = projector(self.fields.elev_3d, self.fields.elev_cg_3d)

        # ----- set initial values
        expandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d).solve()
        getZCoordFromMesh(self.fields.z_coord_ref_3d)
        self.fields.z_coord_3d.assign(self.fields.z_coord_ref_3d)
        computeElemHeight(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
        self.copyVElemSizeTo2d.solve()

        self._initialized = True
        self._isfrozen = True

    def assignInitialConditions(self, elev=None, salt=None, uv_2d=None):
        if not self._initialized:
            self.createEquations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
            self.copyElevTo3d.solve()
            self.fields.elev_cg_3d.project(self.fields.elev_3d)
            if self.options.useALEMovingMesh:
                self.meshCoordUpdater.solve()
                computeElemHeight(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
                self.copyVElemSizeTo2d.solve()
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            expandFunctionTo3d(self.fields.uv_2d, self.fields.uv_3d,
                               elemHeight=self.fields.v_elem_size_3d).solve()
        if salt is not None and self.options.solveSalt:
            self.fields.salt_3d.project(salt)
        self.wSolver.solve()
        if self.options.useALEMovingMesh:
            self.wMeshSolver.solve()
        if self.options.baroclinic:
            computeBaroclinicHead(self, self.fields.salt_3d, self.fields.baroc_head_3d,
                                  self.fields.baroc_head_2d, self.fields.baroc_head_int_3d,
                                  self.fields.bathymetry_3d)

        self.timeStepper.initialize()

        self.options.checkSaltConservation *= self.options.solveSalt
        self.options.checkSaltDeviation *= self.options.solveSalt
        self.options.checkVolConservation3d *= self.options.useALEMovingMesh

    def export(self):
        for key in self.exporters:
            self.exporters[key].export()

    def loadState(self, iExport, t, iteration):
        """Loads simulation state from hdf5 outputs."""
        # TODO use options to figure out which functions need to be loaded
        raise NotImplementedError('state loading is not yet implemented for 3d solver')

    def printState(self, cputime):
        norm_h = norm(self.fields.solution2d.split()[1])
        norm_u = norm(self.fields.solution2d.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(bold(line.format(iexp=self.iExport, i=self.iteration, t=self.simulation_time, e=norm_h,
                                   u=norm_u, cpu=cputime)))
            sys.stdout.flush()

    def iterate(self, updateForcings=None, updateForcings3d=None,
                exportFunc=None):
        if not self._initialized:
            self.createEquations()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        self.simulation_time = 0
        self.iteration = 0
        self.iExport = 1
        next_export_t = self.simulation_time + self.options.TExport

        # initialize conservation checks
        if self.options.checkVolConservation2d:
            eta = self.fields.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.fields.bathymetry_2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))
        if self.options.checkVolConservation3d:
            Vol3d_0 = compVolume3d(self.mesh)
            printInfo('Initial volume 3d {0:f}'.format(Vol3d_0))
        if self.options.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.fields.salt_3d)
            printInfo('Initial salt mass {0:f}'.format(Mass3d_0))
        if self.options.checkSaltDeviation:
            saltSum = self.fields.salt_3d.dat.data.sum()
            saltSum = op2.MPI.COMM.allreduce(saltSum, op=MPI.SUM)
            nbNodes = self.fields.salt_3d.dat.data.shape[0]
            nbNodes = op2.MPI.COMM.allreduce(nbNodes, op=MPI.SUM)
            saltVal = saltSum/nbNodes
            printInfo('Initial mean salt value {0:f}'.format(saltVal))
        if self.options.checkSaltOvershoot:
            saltMin0 = self.fields.salt_3d.dat.data.min()
            saltMax0 = self.fields.salt_3d.dat.data.max()
            saltMin0 = op2.MPI.COMM.allreduce(saltMin0, op=MPI.MIN)
            saltMax0 = op2.MPI.COMM.allreduce(saltMax0, op=MPI.MAX)
            printInfo('Initial salt value range {0:.3f}-{1:.3f}'.format(saltMin0, saltMax0))

        # initial export
        self.export()
        if exportFunc is not None:
            exportFunc()
        self.exporters['vtk'].exportBathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.T + T_epsilon:

            self.timeStepper.advance(self.simulation_time, self.dt,
                                     updateForcings, updateForcings3d)

            # Move to next time step
            self.simulation_time += self.dt
            self.iteration += 1

            # Write the solution to file
            if self.simulation_time >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                self.printState(cputime)

                if self.options.checkVolConservation2d:
                    Vol2d = compVolume2d(self.fields.solution2d.split()[1],
                                         self.fields.bathymetry_2d)
                if self.options.checkVolConservation3d:
                    Vol3d = compVolume3d(self.mesh)
                if self.options.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.fields.salt_3d)
                if self.options.checkSaltDeviation:
                    saltMin = self.fields.salt_3d.dat.data.min()
                    saltMax = self.fields.salt_3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if self.options.checkSaltOvershoot:
                    saltMin = self.fields.salt_3d.dat.data.min()
                    saltMax = self.fields.salt_3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    overshoot = max(saltMax-saltMax0, 0.0)
                    undershoot = min(saltMin-saltMin0, 0.0)
                    saltOversh = (undershoot, overshoot)
                if commrank == 0:
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
                self.iExport += 1

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
