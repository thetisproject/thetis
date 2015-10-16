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


class modelOptions(object):
    """
    Stores all model options

    """
    def __init__(self):
        """
        Initialize with default options
        """
        self.cfl_2d = 1.0
        """float: Factor to scale the 2d time step"""
        self.cfl_3d = 1.0
        """float: Factor to scale the 2d time step"""
        self.order = 1
        """int: Polynomial degree of elements"""
        self.mimetic = True
        """bool: Use mimetic elements for uvw instead of DG"""
        self.nonlin = True
        """bool: Use nonlinear shallow water equations"""
        self.solveSalt = True
        """bool: Solve salt transport"""
        self.solveVertDiffusion = True
        """bool: Solve implicit vert diffusion"""
        self.useBottomFriction = True
        """bool: Apply log layer bottom stress"""
        self.useParabolicViscosity = False
        """bool: Compute parabolic eddy viscosity"""
        self.useALEMovingMesh = True
        """bool: 3D mesh tracks free surface"""
        self.useModeSplit = True
        """bool: Solve 2D/3D modes with different dt"""
        self.useSemiImplicit2D = True
        """bool: Implicit 2D waves (only w. mode split)"""
        self.useIMEX = False
        """bool: Use IMEX time integrator (only with mode split)"""
        self.useTurbulence = False
        """bool: GLS turbulence model"""
        self.useTurbulenceAdvection = False
        """bool: Advect tke,psi with velocity"""
        self.lin_drag = None
        """Coefficient or None: 2D linear drag parameter tau/H/rho_0 = -drag*u"""
        self.hDiffusivity = None
        """Coefficient or None: Background diffusivity"""
        self.vDiffusivity = None
        """Coefficient or None: background diffusivity"""
        self.hViscosity = None
        """Coefficient or None: background viscosity"""
        self.vViscosity = None
        """Coefficient or None: background viscosity"""
        self.coriolis = None
        """2D Coefficient or None: Coriolis parameter"""
        self.wind_stress = None
        """Coefficient or None: Stress at free surface (2D vector function)"""
        self.baroclinic = False  #: NOTE implies that salt3d field is density [kg/m3]
        """bool: Compute internal pressure gradient in momentum equation"""
        self.smagorinskyFactor = None
        """Constant or None: Smagorinsky viscosity factor C_S"""
        self.saltJumpDiffFactor = None
        """Constant or None: Non-linear jump diffusion factor"""
        self.saltRange = Constant(30.0)
        """Constant or None: Salt max-min range for jump diffusion"""
        self.useLimiterForTracers = False
        """bool: Apply P1DG limiter for tracer fields"""
        self.uvLaxFriedrichs = Constant(1.0)
        """Coefficient or None: Scaling factor for uv L-F stability term."""
        self.tracerLaxFriedrichs = Constant(1.0)
        """Coefficient or None: Scaling factor for tracer L-F stability term."""
        self.checkVolConservation2d = False
        """bool: Print deviation from initial volume for 2D mode (eta)"""
        self.checkVolConservation3d = False
        """bool: Print deviation from initial volume for 3D mode (domain volume)"""
        self.checkSaltConservation = False
        """bool: Print deviation from initial salt mass"""
        self.checkSaltDeviation = False
        """bool: Print deviation from mean of initial value"""
        self.checkSaltOvershoot = False
        """bool: Print overshoots that exceed initial range"""
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_eta3d',
                            'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                            'aux_mom_coupling',
                            'func_copy2dTo3d', 'func_copy3dTo2d',
                            'func_vert_int']
        """list of str: Labels of timer sections to print out"""
        self.outputDir = 'outputs'
        """str: Directory where model output files are stored"""
        self.fieldsToExport = ['elev2d', 'uv2d', 'uv3d', 'w3d']
        """list of str: Fields to export in VTK format"""
        self.fieldsToExportNumpy = []
        """list of str: Fields to export in numpy format"""
        self.verbose = 0
        """int: Verbosity level"""
        self.TExport = 100.0  
        """float: Export interval in seconds. All fields in fieldsToExport list will be stored to disk and diagnostics will be computed."""
        self.T = 1000.0
        """float: Simulation duration in seconds"""
        self.uAdvection = Constant(0.0)
        """Constant: Max. horizontal velocity magnitude for computing max stable advection time step."""

    @classmethod
    def fromDict(cls, d):
        """
        Creates a new object overriding all devault values from the given dict
        """
        o = cls()
        o.__dict__.update(d)
        return o

    def getDict(self):
        """
        Returns all options in a dict
        """
        return self.__dict__


# TODO move to fieldDefs.py
# TODO come up with consistent field naming scheme
# name_3d_version?
fieldMetadata = {}
"""
Holds description, units and output file information for each field.

name      - human readable description
fieldname - description used in visualization etc
filename  - filename for output files
unit      - SI unit of the field
"""

fieldMetadata['bathymetry2d'] = {
    'name': 'Bathymetry',
    'fieldname': 'Bathymetry',
    'filename': 'bathymetry2d',
    'unit': 'm',
    }
fieldMetadata['bathymetry3d'] = {
    'name': 'Bathymetry',
    'fieldname': 'Bathymetry',
    'filename': 'bathymetry3d',
    'unit': 'm',
    }
fieldMetadata['z_coord3d'] = {
    'name': 'Mesh z coordinates',
    'fieldname': 'Z coordinates',
    'filename': 'ZCoord3d',
    'unit': 'm',
    }
fieldMetadata['z_coord_ref3d'] = {
    'name': 'Static mesh z coordinates',
    'fieldname': 'Z coordinates',
    'filename': 'ZCoordRef3d',
    'unit': 'm',
    }
fieldMetadata['uv2d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'Velocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['uvDav2d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'DAVelocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['uvDav3d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'DAVelocity3d',
    'unit': 'm s-1',
    }
fieldMetadata['uv3d_mag'] = {
    'name': 'Magnitude of horizontal velocity',
    'fieldname': 'Velocity magnitude',
    'filename': 'VeloMag3d',
    'unit': 'm s-1',
    }
fieldMetadata['uv3d_P1'] = {
    'name': 'P1 projection of horizontal velocity',
    'fieldname': 'P1 Velocity',
    'filename': 'VeloCG3d',
    'unit': 'm s-1',
    }
fieldMetadata['uvBot2d'] = {
    'name': 'Bottom velocity',
    'fieldname': 'Bottom velocity',
    'filename': 'BotVelocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['elev2d'] = {
    'name': 'Water elevation',
    'fieldname': 'Elevation',
    'filename': 'Elevation2d',
    'unit': 'm',
    }
fieldMetadata['elev3d'] = {
    'name': 'Water elevation',
    'fieldname': 'Elevation',
    'filename': 'Elevation3d',
    'unit': 'm',
    }
fieldMetadata['elev3dCG'] = {
    'name': 'Water elevation CG',
    'fieldname': 'Elevation',
    'filename': 'ElevationCG3d',
    'unit': 'm',
    }
fieldMetadata['uv3d'] = {
    'name': 'Horizontal velocity',
    'fieldname': 'Horizontal velocity',
    'filename': 'Velocity3d',
    'unit': 'm s-1',
    }
fieldMetadata['w3d'] = {
    'name': 'Vertical velocity',
    'fieldname': 'Vertical velocity',
    'filename': 'VertVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['wMesh3d'] = {
    'name': 'Mesh velocity',
    'fieldname': 'Mesh velocity',
    'filename': 'MeshVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['salt3d'] = {
    'name': 'Water salinity',
    'fieldname': 'Salinity',
    'filename': 'Salinity3d',
    'unit': 'psu',
    }
fieldMetadata['parabNuv3d'] = {
    'name': 'Parabolic Viscosity',
    'fieldname': 'Parabolic Viscosity',
    'filename': 'ParabVisc3d',
    'unit': 'm2 s-1',
    }

fieldMetadata['eddyNuv3d'] = {
    'name': 'Eddy Viscosity',
    'fieldname': 'Eddy Viscosity',
    'file': 'EddyVisc3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['shearFreq3d'] = {
    'name': 'Vertical shear frequency squared',
    'fieldname': 'Vertical shear frequency squared',
    'file': 'ShearFreq3d',
    'unit': 's-2',
    }
fieldMetadata['tke3d'] = {
    'name': 'Turbulent Kinetic Energy',
    'fieldname': 'Turbulent Kinetic Energy',
    'file': 'TurbKEnergy3d',
    'unit': 'm2 s-2',
    }
fieldMetadata['psi3d'] = {
    'name': 'Turbulence psi variable',
    'fieldname': 'Turbulence psi variable',
    'file': 'TurbPsi3d',
    'unit': '',
    }
fieldMetadata['eps3d'] = {
    'name': 'TKE dissipation rate',
    'fieldname': 'TKE dissipation rate',
    'file': 'TurbEps3d',
    'unit': 'm2 s-2',
    }
fieldMetadata['len3d'] = {
    'name': 'Turbulent lenght scale',
    'fieldname': 'Turbulent lenght scale',
    'file': 'TurbLen3d',
    'unit': 'm',
    }
fieldMetadata['barohead3d'] = {
    'name': 'Baroclinic head',
    'fieldname': 'Baroclinic head',
    'file': 'Barohead3d',
    'unit': 'm',
    }
fieldMetadata['barohead2d'] = {
    'name': 'Dav baroclinic head',
    'fieldname': 'Dav baroclinic head',
    'file': 'Barohead2d',
    'unit': 'm',
    }
fieldMetadata['gjvAlphaH3d'] = {
    'name': 'GJV Parameter h',
    'fieldname': 'GJV Parameter h',
    'file': 'GJVParamH',
    'unit': '',
    }
fieldMetadata['gjvAlphaV3d'] = {
    'name': 'GJV Parameter v',
    'fieldname': 'GJV Parameter v',
    'file': 'GJVParamV',
    'unit': '',
    }
fieldMetadata['smagViscosity'] = {
    'name': 'Smagorinsky viscosity',
    'fieldname': 'Smagorinsky viscosity',
    'file': 'SmagViscosity3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['saltJumpDiff'] = {
    'name': 'Salt Jump Diffusivity',
    'fieldname': 'Salt Jump Diffusivity',
    'file': 'SaltJumpDiff3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['maxHDiffusivity'] = {
    'name': 'Maximum stable horizontal diffusivity',
    'fieldname': 'Maximum horizontal diffusivity',
    'file': 'MaxHDiffusivity3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['vElemSize3d'] = {
    'name': 'Element size in vertical direction',
    'fieldname': 'Vertical element size',
    'file': 'VElemSize3d',
    'unit': 'm',
    }
fieldMetadata['vElemSize2d'] = {
    'name': 'Element size in vertical direction',
    'fieldname': 'Vertical element size',
    'file': 'VElemSize2d',
    'unit': 'm',
    }
fieldMetadata['hElemSize3d'] = {
    'name': 'Element size in horizontal direction',
    'fieldname': 'Horizontal element size',
    'file': 'hElemSize3d',
    'unit': 'm',
    }
fieldMetadata['hElemSize2d'] = {
    'name': 'Element size in horizontal direction',
    'fieldname': 'Horizontal element size',
    'file': 'hElemSize2d',
    'unit': 'm',
    }


# TODO pass functions object to equations to simplify
# TODO make sure all function names match with fieldMetadata
# TODO store options are solver.options for clarity
# TODO remove function space from the equation constructors - use solution.function_space

class AttrDict(dict):
    """
    Dictionary that provides both self['key'] and self.key access to members.

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """
    def __init__(self, *args, **kwargs):
        if sys.version_info < (2, 7, 4):
            raise Exception('AttrDict requires python >= 2.7.4 to avoid memory leaks')
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class functionDict(AttrDict):
    """
    AttrDict that checks that all added functions have proper meta data
    """
    def _checkInputs(self, key, value):
        if key != '__dict__':
            if not isinstance(value, Function):
                raise Exception('Wrong type: only Function objects can be added')
            fs = value.function_space()
            if not isinstance(fs, MixedFunctionSpace) and key not in fieldMetadata:
                raise Exception('Trying to add a field "{:}" that has no fieldMetadata'.format(key))

    def __setitem__(self, key, value):
        self._checkInputs(key, value)
        super(functionDict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self._checkInputs(key, value)
        super(functionDict, self).__setattr__(key, value)


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
        opt = modelOptions().fromDict(options)
        # add as attributes to this class
        self.__dict__.update(opt.getDict())

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}

        self.visualizationSpaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.functions = functionDict()
        """Holds all functions needed by the solver object."""
        self.functions.bathymetry2d = bathymetry2d

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
        self.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0, name='P0')
        self.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1')
        self.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1v')
        self.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DG')
        self.P1DGv = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DGv')

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
        if self.mimetic:
            #self.U = FunctionSpace(self.mesh, UW_elt)  # uv
            self.U = FunctionSpace(self.mesh, U_elt, name='U')  # uv
            self.W = FunctionSpace(self.mesh, W_elt, name='W')  # w
        else:
            self.U = VectorFunctionSpace(self.mesh, 'DG', self.order,
                                         vfamily='DG', vdegree=self.order,
                                         name='U')
            self.W = VectorFunctionSpace(self.mesh, 'DG', self.order,
                                         vfamily='CG', vdegree=self.order + 1,
                                         name='W')
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.Uproj = self.U

        self.Uint = self.U  # vertical integral of uv
        # tracers
        self.H = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=max(0, self.order), name='H')
        # vertical integral of tracers
        self.Hint = FunctionSpace(self.mesh, 'DG', self.order, vfamily='CG', vdegree=self.order+1, name='Hint')
        # for scalar fields to be used in momentum eq NOTE could be omitted ?
        self.U_scalar = FunctionSpace(self.mesh, 'DG', self.order, vfamily='DG', vdegree=self.order, name='U_scalar')
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
        if self.mimetic:
            # NOTE this is not compatible with enriched UW space used in 3D
            self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.order+1)
        else:
            self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.order, name='U_2d')
        self.Uproj_2d = self.U_2d
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.order, name='U_scalar_2d')
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.order, name='H_2d')
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d], name='V_2d')
        self.visualizationSpaces[self.U_2d] = self.P1v_2d
        self.visualizationSpaces[self.H_2d] = self.P1_2d

        # ----- fields
        self.functions.solution2d = Function(self.V_2d, name='solution2d')
        # correct treatment of the split 2d functions
        uv2d, eta2d = self.functions.solution2d.split()
        self.functions.uv2d = uv2d
        self.functions.elev2d = eta2d
        self.visualizationSpaces[uv2d.function_space()] = self.P1v_2d
        self.visualizationSpaces[eta2d.function_space()] = self.P1_2d
        if self.useBottomFriction:
            self.functions.uv_bottom2d = Function(self.P1v_2d, name='Bottom Velocity')
            self.functions.z_bottom2d = Function(self.P1_2d, name='Bot. Vel. z coord')
            self.functions.bottom_drag2d = Function(self.P1_2d, name='Bottom Drag')

        self.functions.elev3d = Function(self.H, name='Elevation')
        self.functions.elev3dCG = Function(self.P1, name='Elevation')
        self.functions.bathymetry3d = Function(self.P1, name='Bathymetry')
        self.functions.uv3d = Function(self.U, name='Velocity')
        if self.useBottomFriction:
            self.functions.uv_bottom3d = Function(self.P1v, name='Bottom Velocity')
            self.functions.z_bottom3d = Function(self.P1, name='Bot. Vel. z coord')
            self.functions.bottom_drag3d = Function(self.P1, name='Bottom Drag')
        # z coordinate in the strecthed mesh
        self.functions.z_coord3d = Function(self.P1, name='z coord')
        # z coordinate in the reference mesh (eta=0)
        self.functions.z_coord_ref3d = Function(self.P1, name='ref z coord')
        self.functions.uvDav3d = Function(self.Uproj, name='Depth Averaged Velocity 3d')
        self.functions.uvDav2d = Function(self.Uproj_2d, name='Depth Averaged Velocity 2d')
        #self.functions.uv3d_tmp = Function(self.U, name='Velocity')
        self.functions.uv3d_mag = Function(self.P0, name='Velocity magnitude')
        self.functions.uv3d_P1 = Function(self.P1v, name='Smoothed Velocity')
        self.functions.w3d = Function(self.W, name='Vertical Velocity')
        if self.useALEMovingMesh:
            self.functions.w_mesh3d = Function(self.H, name='Vertical Velocity')
            self.functions.dw_mesh_dz_3d = Function(self.H, name='Vertical Velocity dz')
            self.functions.w_mesh_surf3d = Function(self.H, name='Vertical Velocity Surf')
            self.functions.w_mesh_surf2d = Function(self.H_2d, name='Vertical Velocity Surf')
        if self.solveSalt:
            self.functions.salt3d = Function(self.H, name='Salinity')
        if self.solveVertDiffusion and self.functions.useParabolicViscosity:
            # FIXME useParabolicViscosity is OBSOLETE
            self.functions.parabViscosity_v = Function(self.P1, name='Eddy viscosity')
        if self.baroclinic:
            self.functions.baroHead3d = Function(self.Hint, name='Baroclinic head')
            self.functions.baroHeadInt3d = Function(self.Hint, name='V.int. baroclinic head')
            self.functions.baroHead2d = Function(self.H_2d, name='DAv baroclinic head')
        if self.coriolis is not None:
            if isinstance(self.functions.coriolis, Constant):
                self.functions.coriolis3d = self.coriolis
            else:
                self.functions.coriolis3d = Function(self.P1, name='Coriolis parameter')
                copy2dFieldTo3d(self.coriolis, self.coriolis3d)
        if self.wind_stress is not None:
            self.functions.wind_stress3d = Function(self.U_visu, name='Wind stress')
            copy2dFieldTo3d(self.wind_stress, self.functions.wind_stress3d)
        self.functions.vElemSize3d = Function(self.P1DG, name='element height')
        self.functions.vElemSize2d = Function(self.P1DG_2d, name='element height')
        self.functions.hElemSize3d = getHorzontalElemSize(self.P1_2d, self.P1)
        self.functions.maxHDiffusivity = Function(self.P1, name='Maximum h. Diffusivity')
        if self.smagorinskyFactor is not None:
            self.functions.smag_viscosity = Function(self.P1, name='Smagorinsky viscosity')
        if self.saltJumpDiffFactor is not None:
            self.functions.saltJumpDiff = Function(self.P1, name='Salt Jump Diffusivity')
        if self.useLimiterForTracers:
            self.tracerLimiter = limiter.vertexBasedP1DGLimiter(self.H,
                                                                self.P1,
                                                                self.P0)
        else:
            self.tracerLimiter = None
        if self.useTurbulence:
            # NOTE tke and psi should be in H as tracers ??
            self.functions.tke3d = Function(self.turb_space, name='Turbulent kinetic energy')
            self.functions.psi3d = Function(self.turb_space, name='Turbulence psi variable')
            # NOTE other turb. quantities should share the same nodes ??
            self.functions.epsilon3d = Function(self.turb_space, name='TKE dissipation rate')
            self.functions.len3d = Function(self.turb_space, name='Turbulent lenght scale')
            self.functions.eddyVisc_v = Function(self.turb_space, name='Vertical eddy viscosity')
            self.functions.eddyDiff_v = Function(self.turb_space, name='Vertical eddy diffusivity')
            # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
            self.functions.shearFreq2_3d = Function(self.turb_space, name='Shear frequency squared')
            self.functions.buoyancyFreq2_3d = Function(self.turb_space, name='Buoyancy frequency squared')
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
        self.tot_h_visc.add(self.functions.get('hViscosity'))
        self.tot_h_visc.add(self.functions.get('smag_viscosity'))
        self.tot_v_visc = sumFunction()
        self.tot_v_visc.add(self.functions.get('vViscosity'))
        self.tot_v_visc.add(self.functions.get('eddyVisc_v'))
        self.tot_v_visc.add(self.functions.get('parabViscosity_v'))
        self.tot_salt_h_diff = sumFunction()
        self.tot_salt_h_diff.add(self.functions.get('hDiffusivity'))
        self.tot_salt_v_diff = sumFunction()
        self.tot_salt_v_diff.add(self.functions.get('vDiffusivity'))
        self.tot_salt_v_diff.add(self.functions.get('eddyDiff_v'))

        # set initial values
        copy2dFieldTo3d(self.functions.bathymetry2d, self.functions.bathymetry3d)
        getZCoordFromMesh(self.functions.z_coord_ref3d)
        self.functions.z_coord3d.assign(self.functions.z_coord_ref3d)
        computeElemHeight(self.functions.z_coord3d, self.functions.vElemSize3d)
        copy3dFieldTo2d(self.functions.vElemSize3d, self.functions.vElemSize2d)

        # ----- Equations
        if self.useModeSplit:
            # full 2D shallow water equations
            self.eq_sw = module_2d.shallowWaterEquations(
                self.mesh2d, self.V_2d, self.functions.solution2d, self.functions.bathymetry2d,
                self.functions.get('uv_bottom2d'), self.functions.get('bottom_drag2d'),
                baro_head=self.functions.get('baroHead2d'),
                viscosity_h=self.functions.get('hViscosity'),  # FIXME add 2d smag
                uvLaxFriedrichs=self.uvLaxFriedrichs,
                coriolis=self.coriolis,
                wind_stress=self.wind_stress,
                lin_drag=self.lin_drag,
                nonlin=self.nonlin)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.functions.solution2d.split()
            self.eq_sw = module_2d.freeSurfaceEquation(
                self.mesh2d, self.H_2d, eta, uv, self.functions.bathymetry2d,
                nonlin=self.nonlin)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = module_3d.momentumEquation(
            self.mesh, self.U, self.U_scalar, bnd_markers,
            bnd_len, self.functions.uv3d, self.functions.elev3d,
            self.functions.bathymetry3d, w=self.functions.w3d,
            baro_head=self.functions.get('baroHead3d'),
            w_mesh=self.functions.get('w_mesh3d'),
            dw_mesh_dz=self.functions.get('dw_mesh_dz_3d'),
            viscosity_v=self.tot_v_visc.getSum(),
            viscosity_h=self.tot_h_visc.getSum(),
            laxFriedrichsFactor=self.uvLaxFriedrichs,
            #uvMag=self.uv3d_mag,
            uvP1=self.functions.get('uv3d_P1'),
            coriolis=self.functions.get('coriolis3d'),
            lin_drag=self.lin_drag,
            nonlin=self.nonlin)
        if self.solveSalt:
            self.eq_salt = tracerEquation.tracerEquation(
                self.mesh, self.H, self.functions.salt3d, self.functions.elev3d, self.functions.uv3d,
                w=self.functions.w3d, w_mesh=self.functions.get('w_mesh3d'),
                dw_mesh_dz=self.functions.get('dw_mesh_dz_3d'),
                diffusivity_h=self.tot_salt_h_diff.getSum(),
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                #uvMag=self.uv3d_mag,
                uvP1=self.functions.get('uv3d_P1'),
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.functions.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
        if self.solveVertDiffusion:
            self.eq_vertmomentum = module_3d.verticalMomentumEquation(
                self.mesh, self.U, self.U_scalar, self.functions.uv3d, w=None,
                viscosity_v=self.tot_v_visc.getSum(),
                uv_bottom=self.functions.get('uv_bottom3d'),
                bottom_drag=self.functions.get('bottom_drag3d'),
                wind_stress=self.functions.get('wind_stress3d'),
                vElemSize=self.functions.vElemSize3d)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.solveSalt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.useTurbulence:
            # explicit advection equations
            self.eq_tke_adv = tracerEquation.tracerEquation(
                self.mesh, self.H, self.functions.tke3d, self.functions.elev3d, self.functions.uv3d,
                w=self.functions.w3d, w_mesh=self.functions.get('w_mesh3d'),
                dw_mesh_dz=self.functions.get('dw_mesh_dz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.functions.get('uv3d_P1'),
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.functions.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            self.eq_psi_adv = tracerEquation.tracerEquation(
                self.mesh, self.H, self.functions.psi3d, self.functions.elev3d, self.functions.uv3d,
                w=self.functions.w3d, w_mesh=self.functions.get('w_mesh3d'),
                dw_mesh_dz=self.functions.get('dw_mesh_dz_3d'),
                diffusivity_h=None,  # TODO add horiz. diffusivity?
                diffusivity_v=None,
                uvP1=self.functions.get('uv3d_P1'),
                laxFriedrichsFactor=self.tracerLaxFriedrichs,
                vElemSize=self.functions.vElemSize3d,
                bnd_markers=bnd_markers,
                bnd_len=bnd_len)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.tkeEquation(
                self.mesh, self.functions.tke3d.function_space(), self.functions.tke3d,
                self.functions.elev3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.functions.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)
            self.eq_psi_diff = turbulence.psiEquation(
                self.mesh, self.functions.psi3d.function_space(), self.functions.psi3d, self.functions.elev3d, uv=None,
                w=None, w_mesh=None,
                dw_mesh_dz=None,
                diffusivity_h=None,
                diffusivity_v=self.tot_salt_v_diff.getSum(),
                viscosity_v=self.tot_v_visc.getSum(),
                vElemSize=self.functions.vElemSize3d,
                uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                bnd_markers=bnd_markers, bnd_len=bnd_len,
                glsModel=self.glsModel)

        # ----- Time integrators
        self.setTimeStep()
        if self.useModeSplit:
            if self.useIMEX:
                self.timeStepper = coupledTimeIntegrator.coupledSSPIMEX(weakref.proxy(self))
            elif self.useSemiImplicit2D:
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSemiImplicit(weakref.proxy(self))
            else:
                self.timeStepper = coupledTimeIntegrator.coupledSSPRKSync(weakref.proxy(self))
        else:
            self.timeStepper = coupledTimeIntegrator.coupledSSPRKSingleMode(weakref.proxy(self))
        printInfo('using {:} time integrator'.format(self.timeStepper.__class__.__name__))

        # compute maximal diffusivity for explicit schemes
        maxDiffAlpha = 1.0/100.0  # FIXME depends on element type and order
        self.functions.maxHDiffusivity.assign(maxDiffAlpha/self.dt * self.functions.hElemSize3d**2)

        # ----- File exporters
        # create exportManagers and store in a list
        self.exporters = {}
        e = exporter.exportManager(self.outputDir,
                                   self.fieldsToExport,
                                   self.functions,
                                   self.visualizationSpaces,
                                   fieldMetadata,
                                   exportType='vtk',
                                   verbose=self.verbose > 0)
        self.exporters['vtk'] = e
        numpyDir = os.path.join(self.outputDir, 'numpy')
        e = exporter.exportManager(numpyDir,
                                   self.fieldsToExportNumpy,
                                   self.functions,
                                   self.visualizationSpaces,
                                   fieldMetadata,
                                   exportType='numpy',
                                   verbose=self.verbose > 0)
        self.exporters['numpy'] = e

        self.uvP1_projector = projector(self.functions.uv3d, self.functions.uv3d_P1)
        #self.uvDAV_to_tmp_projector = projector(self.uv3d_dav, self.uv3d_tmp)
        #self.uv2d_to_DAV_projector = projector(self.solution2d.split()[0],
                                               #self.uv2d_dav)
        #self.uv2dDAV_to_uv2d_projector = projector(self.uv2d_dav,
                                                   #self.solution2d.split()[0])
        self.eta3d_to_CG_projector = projector(self.functions.elev3d, self.functions.elev3dCG)

        self._initialized = True

    def assignInitialConditions(self, elev=None, salt=None, uv2d=None):
        if not self._initialized:
            self.mightyCreator()
        if elev is not None:
            elev2d = self.functions.solution2d.split()[1]
            elev2d.project(elev)
            copy2dFieldTo3d(elev2d, self.functions.elev3d)
            self.functions.elev3dCG.project(self.functions.elev3d)
            if self.useALEMovingMesh:
                updateCoordinates(self.mesh, self.functions.elev3dCG, self.functions.bathymetry3d,
                                  self.functions.z_coord3d, self.functions.z_coord_ref3d)
                computeElemHeight(self.functions.z_coord3d, self.functions.vElemSize3d)
                copy3dFieldTo2d(self.functions.vElemSize3d, self.functions.vElemSize2d)
        if uv2d is not None:
            uv2d_field = self.functions.solution2d.split()[0]
            uv2d_field.project(uv2d)
            copy2dFieldTo3d(uv2d_field, self.functions.uv3d,
                            elemHeight=self.vElemSize3d)

        if salt is not None and self.solveSalt:
            self.functions.salt3d.project(salt)
        computeVertVelocity(self.functions.w3d, self.functions.uv3d, self.functions.bathymetry3d,
                            self.eq_momentum.boundary_markers,
                            self.eq_momentum.bnd_functions)
        if self.useALEMovingMesh:
            computeMeshVelocity(self.functions.elev3d, self.functions.uv3d, self.functions.w3d, self.functions.w_mesh3d,
                                self.functions.w_mesh_surf3d, self.functions.w_mesh_surf2d,
                                self.functions.dw_mesh_dz_3d,
                                self.functions.bathymetry3d, self.functions.z_coord_ref3d)
        if self.baroclinic:
            computeBaroclinicHead(self.functions.salt3d, self.functions.baroHead3d,
                                  self.functions.baroHead2d, self.functions.baroHeadInt3d,
                                  self.functions.bathymetry3d)

        self.timeStepper.initialize()

        self.checkSaltConservation *= self.solveSalt
        self.checkSaltDeviation *= self.solveSalt
        self.checkVolConservation3d *= self.useALEMovingMesh

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
        next_export_t = t + self.TExport

        # initialize conservation checks
        if self.checkVolConservation2d:
            eta = self.functions.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.functions.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))
        if self.checkVolConservation3d:
            Vol3d_0 = compVolume3d(self.mesh)
            printInfo('Initial volume 3d {0:f}'.format(Vol3d_0))
        if self.checkSaltConservation:
            Mass3d_0 = compTracerMass3d(self.salt3d)
            printInfo('Initial salt mass {0:f}'.format(Mass3d_0))
        if self.checkSaltDeviation:
            saltSum = self.functions.salt3d.dat.data.sum()
            saltSum = op2.MPI.COMM.allreduce(saltSum, op=MPI.SUM)
            nbNodes = self.functions.salt3d.dat.data.shape[0]
            nbNodes = op2.MPI.COMM.allreduce(nbNodes, op=MPI.SUM)
            saltVal = saltSum/nbNodes
            printInfo('Initial mean salt value {0:f}'.format(saltVal))
        if self.checkSaltOvershoot:
            saltMin0 = self.functions.salt3d.dat.data.min()
            saltMax0 = self.functions.salt3d.dat.data.max()
            saltMin0 = op2.MPI.COMM.allreduce(saltMin0, op=MPI.MIN)
            saltMax0 = op2.MPI.COMM.allreduce(saltMax0, op=MPI.MAX)
            printInfo('Initial salt value range {0:.3f}-{1:.3f}'.format(saltMin0, saltMax0))

        # initial export
        self.export()
        if exportFunc is not None:
            exportFunc()
        self.exporters['vtk'].exportBathymetry(self.functions.bathymetry2d)

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
                norm_h = norm(self.functions.solution2d.split()[1])
                norm_u = norm(self.functions.solution2d.split()[0])

                if self.checkVolConservation2d:
                    Vol2d = compVolume2d(self.functions.solution2d.split()[1],
                                         self.functions.bathymetry2d)
                if self.checkVolConservation3d:
                    Vol3d = compVolume3d(self.mesh)
                if self.checkSaltConservation:
                    Mass3d = compTracerMass3d(self.functions.salt3d)
                if self.checkSaltDeviation:
                    saltMin = self.functions.salt3d.dat.data.min()
                    saltMax = self.functions.salt3d.dat.data.max()
                    saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
                    saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
                    saltDev = ((saltMin-saltVal)/saltVal,
                               (saltMax-saltVal)/saltVal)
                if self.checkSaltOvershoot:
                    saltMin = self.functions.salt3d.dat.data.min()
                    saltMax = self.functions.salt3d.dat.data.max()
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

                self.export()
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
    def __init__(self, mesh2d, bathymetry2d, order=1, options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.bathymetry2d = bathymetry2d

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        options.setdefault('timeStepperType', 'SSPRK33')
        options.setdefault('timerLabels', ['mode2d'])
        options.setdefault('fieldsToExport', ['elev2d', 'uv2d'])

        # override default options
        opt = modelOptions().fromDict(options)
        # add as attributes to this class
        self.__dict__.update(opt.getDict())

        self.bnd_functions = {'shallow_water': {}}

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
        elif self.timeStepperType.lower() == 'sspimex':
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
