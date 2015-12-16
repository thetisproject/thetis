"""
This file defines all options of the 2D/3D models excluding field values.

Tuomas Karna 2015-10-17
"""
from cofs.utility import *


class modelOptions(AttrDict):
    """
    Stores all model options
    """
    def __init__(self):
        """
        Initialize with default options
        """
        super(modelOptions, self).__init__()
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
        self.baroclinic = False  #: NOTE implies that salt_3d field is density [kg/m3]
        """bool: Compute internal pressure gradient in momentum equation"""
        self.smagorinskyFactor = None
        """Constant or None: Smagorinsky viscosity factor C_S"""
        self.salt_jump_diffFactor = None
        """Constant or None: Non-linear jump diffusion factor"""
        self.saltRange = Constant(30.0)
        """Constant or None: Salt max-min range for jump diffusion"""
        self.useLimiterForTracers = False
        """bool: Apply P1DG limiter for tracer fields"""
        self.uvLaxFriedrichs = Constant(1.0)
        """Constant or None: Scaling factor for uv L-F stability term."""
        self.tracerLaxFriedrichs = Constant(1.0)
        """Constant or None: Scaling factor for tracer L-F stability term."""
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
        self.dt = None
        """float: Time step. If set overrides automatically computed stable dt"""
        self.dt_2d = None
        """float: Time step for 2d mode. If set overrides automatically computed stable dt"""
        self.cfl_2d = 1.0
        """float: Factor to scale the 2d time step"""
        self.cfl_3d = 1.0
        """float: Factor to scale the 2d time step"""
        self.TExport = 100.0
        """float: Export interval in seconds. All fields in fieldsToExport list will be stored to disk and diagnostics will be computed."""
        self.T = 1000.0
        """float: Simulation duration in seconds"""
        self.uAdvection = Constant(0.0)
        """Constant: Max. horizontal velocity magnitude for computing max stable advection time step."""
        self.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion',
                            'continuityEq', 'saltEq', 'aux_eta3d',
                            'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                            'aux_mom_coupling',
                            'func_copy2dTo3d', 'func_copy3dTo2d',
                            'func_vert_int']
        """list of str: Labels of timer sections to print out"""
        self.outputDir = 'outputs'
        """str: Directory where model output files are stored"""
        self.fieldsToExport = ['elev_2d', 'uv_2d', 'uv_3d', 'w_3d']
        """list of str: Fields to export in VTK format"""
        self.fieldsToExportNumpy = []
        """list of str: Fields to export in HDF5 format"""
        self.fieldsToExportHDF5 = []
        """list of str: Fields to export in numpy format"""
        self.verbose = 0
        """int: Verbosity level"""
        # NOTE these are fields, potentially Functions: move out of this class?
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
        self.uv_source_2d = None
        """Coefficient or None: Stress at free surface (3D vector function)"""
        self.uv_source_3d = None
        """Coefficient or None: source term for 2d momentum equation"""
        self.elev_source_2d = None
        """Coefficient or None: source term for 2d continuity equation"""
        self.salt_source_3d = None
        """Coefficient or None: source term for salinity equation"""
