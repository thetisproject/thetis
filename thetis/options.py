"""
This file defines all options of the 2D/3D models excluding field values.

Tuomas Karna 2015-10-17
"""
from thetis.utility import *
from thetis.turbulence import GLSModelOptions


class ModelOptions(AttrDict):
    """
    Stores all circulation model options
    """
    def __init__(self):
        """
        Initialize with default options
        """
        super(ModelOptions, self).__init__()
        self.order = 1
        """int: Polynomial degree of elements"""
        self.mimetic = True
        """bool: Use mimetic elements for uvw instead of DG"""
        self.continuous_pressure = False
        """bool: Use PnDG-P(n+1) velocity, pressure pair"""
        self.nonlin = True
        """bool: Use nonlinear shallow water equations"""
        self.solve_salt = True
        """bool: Solve salt transport"""
        self.solve_temp = True
        """bool: Solve temperature transport"""
        self.solve_vert_diffusion = True
        """bool: Solve implicit vert diffusion"""
        self.use_bottom_friction = True
        """bool: Apply log layer bottom stress"""
        self.use_parabolic_viscosity = False
        """bool: Compute parabolic eddy viscosity"""
        self.include_grad_div_viscosity_term = False
        """bool: Include grad(nu div(u)) term in the depth-averaged viscosity"""
        self.include_grad_depth_viscosity_term = True
        """bool: Include grad(H) term in the depth-averaged viscosity"""
        self.use_ale_moving_mesh = True
        """bool: 3D mesh tracks free surface"""
        self.use_mode_split = True
        """bool: Solve 2D/3D modes with different dt"""
        self.use_semi_implicit_2d = True
        """bool: Implicit 2D waves (only w. mode split)"""
        self.use_imex = False
        """bool: Use IMEX time integrator (only with mode split)"""
        self.use_linearized_semi_implicit_2d = False
        """bool: Use linearized semi-implicit time integration for the horizontal mode"""
        self.use_turbulence = False
        """bool: GLS turbulence model"""
        self.use_smooth_eddy_viscosity = False
        """bool: Cast eddy viscosity to p1 space instead of p0"""
        self.turbulence_model = 'gls'
        """str: Defines the type of vertical turbulence model. Currently only 'gls'"""
        self.gls_options = GLSModelOptions()
        """GLSModelOptions: Dictionary of default GLS model options"""
        self.use_turbulence_advection = False
        """bool: Advect tke,psi with velocity"""
        self.baroclinic = False
        """bool: Compute internal pressure gradient in momentum equation"""
        self.smagorinsky_factor = None
        """Constant or None: Smagorinsky viscosity factor C_S"""
        self.salt_jump_diff_factor = None
        """Constant or None: Non-linear jump diffusion factor"""
        self.salt_range = Constant(30.0)
        """Constant or None: Salt max-min range for jump diffusion"""
        self.use_limiter_for_tracers = False
        """bool: Apply P1DG limiter for tracer fields"""
        self.uv_lax_friedrichs = Constant(1.0)
        """Constant or None: Scaling factor for uv L-F stability term."""
        self.tracer_lax_friedrichs = Constant(1.0)
        """Constant or None: Scaling factor for tracer L-F stability term."""
        self.check_vol_conservation_2d = False
        """bool: Print deviation from initial volume for 2D mode (eta)"""
        self.check_vol_conservation_3d = False
        """bool: Print deviation from initial volume for 3D mode (domain volume)"""
        self.check_salt_conservation = False
        """bool: Print deviation from initial salt mass"""
        self.check_salt_overshoot = False
        """bool: Print overshoots that exceed initial range"""
        self.check_temp_conservation = False
        """bool: Print deviation from initial temp mass"""
        self.check_temp_overshoot = False
        """bool: Print overshoots that exceed initial range"""
        self.dt = None
        """float: Time step. If set overrides automatically computed stable dt"""
        self.dt_2d = None
        """float: Time step for 2d mode. If set overrides automatically computed stable dt"""
        self.cfl_2d = 1.0
        """float: Factor to scale the 2d time step"""
        self.cfl_3d = 1.0
        """float: Factor to scale the 2d time step"""
        self.t_export = 100.0
        """float: Export interval in seconds. All fields in fields_to_export list will be stored to disk and diagnostics will be computed."""
        self.t_end = 1000.0
        """float: Simulation duration in seconds"""
        self.u_advection = Constant(0.0)
        """Constant: Max. horizontal velocity magnitude for computing max stable advection time step."""
        self.timer_labels = ['mode2d', 'momentum_eq', 'vert_diffusion',
                             'continuity_eq', 'salt_eq', 'temp_eq', 'aux_eta3d',
                             'aux_mesh_ale', 'aux_friction', 'aux_barolinicity',
                             'aux_mom_coupling',
                             'func_copy_2d_to_3d', 'func_copy_3d_to_2d',
                             'func_vert_int']
        """list of str: Labels of timer sections to print out"""
        self.outputdir = 'outputs'
        """str: Directory where model output files are stored"""
        self.no_exports = False
        """bool: Do not store any outputs to disk, used in CI test suite"""
        self.fields_to_export = ['elev_2d', 'uv_2d', 'uv_3d', 'w_3d']
        """list of str: Fields to export in VTK format"""
        self.fields_to_export_numpy = []
        """list of str: Fields to export in HDF5 format"""
        self.fields_to_export_hdf5 = []
        """list of str: Fields to export in numpy format"""
        self.verbose = 0
        """int: Verbosity level"""
        # NOTE these are fields, potentially Functions: move out of this class?
        self.lin_drag = None
        """Coefficient or None: 2D linear drag parameter tau/H/rho_0 = -drag*u"""
        self.h_diffusivity = None
        """Coefficient or None: Background diffusivity"""
        self.v_diffusivity = None
        """Coefficient or None: background diffusivity"""
        self.h_viscosity = None
        """Coefficient or None: background viscosity"""
        self.v_viscosity = None
        """Coefficient or None: background viscosity"""
        self.coriolis = None
        """2D Coefficient or None: Coriolis parameter"""
        self.wind_stress = None
        """Coefficient or None: Stress at free surface (2D vector function)"""
        self.uv_source_2d = None
        """Coefficient or None: source term for 2d momentum equation"""
        self.uv_source_3d = None
        """Coefficient or None: source term for 3d momentum equation"""
        self.elev_source_2d = None
        """Coefficient or None: source term for 2d continuity equation"""
        self.salt_source_3d = None
        """Coefficient or None: source term for salinity equation"""
        self.temp_source_3d = None
        """Coefficient or None: source term for temperature equation"""
        self.constant_temp = Constant(10.0)
        """Coefficient: Constant temperature if temperature is not solved"""
        self.constant_salt = Constant(0.0)
        """Coefficient: Constant salinity if salinity is not solved"""
