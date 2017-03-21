"""
This file defines all options of the 2D/3D models excluding field values.

"""
from __future__ import absolute_import
from .utility import *
from .turbulence import GLSModelOptions


class ModelOptions(AttrDict, FrozenClass):
    """
    Stores all circulation model options
    """
    def __init__(self):
        """
        Every instance is initialized with default values
        """
        super(ModelOptions, self).__init__()
        self.order = 1
        """int: Polynomial degree of elements"""
        self.element_family = 'rt-dg'
        """str: Finite element family.

        2D solver supports 'dg-dg', 'rt-dg', or 'dg-cg' velocity-pressure pairs.
        3D solver supports 'dg-dg', or 'rt-dg' velocity-pressure pairs.
        """
        self.nonlin = True
        """bool: Use nonlinear shallow water equations"""
        self.solve_salt = True
        """bool: Solve salt transport"""
        self.solve_temp = True
        """bool: Solve temperature transport"""
        self.solve_vert_diffusion = True
        """bool: Solve implicit vertical diffusion"""
        self.use_bottom_friction = True
        """bool: Apply log layer bottom stress in the 3D model"""
        self.use_parabolic_viscosity = False
        """
        bool: Use idealized parabolic eddy viscosity

        See :class:`.ParabolicViscosity`
        """
        self.include_grad_div_viscosity_term = False
        r"""
        bool: Include :math:`\nabla (\nu_h \nabla \cdot \bar{\textbf{u}})` term in the depth-averaged viscosity

        See :class:`.shallowwater_eq.HorizontalViscosityTerm` for details.
        """
        self.include_grad_depth_viscosity_term = True
        r"""
        bool: Include :math:`\nabla H` term in the depth-averaged viscosity

        See :class:`.shallowwater_eq.HorizontalViscosityTerm` for details.
        """
        self.use_ale_moving_mesh = True
        """bool: Use ALE formulation where 3D mesh tracks free surface"""
        self.timestepper_type = 'ssprk33'
        """
        str: time integrator option.

        Valid options for the 2D solver: 'forwardeuler'|'backwardeuler'|'ssprk33'|'ssprk33semi'|'cranknicolson'|'sspimex'|'pressureprojectionpicard'|'steadystate'

        Valid options for the 2D solver: 'ssprk33'|'erkale'|'leapfrog'|'imexale'|'ssprk22'
        """
        self.use_linearized_semi_implicit_2d = False
        """bool: Use linearized semi-implicit time integration for the horizontal mode"""
        self.shallow_water_theta = 0.5
        """float: theta parameter for shallow water semi-implicit scheme"""
        self.use_turbulence = False
        """bool: Activate GLS turbulence model"""
        self.use_smooth_eddy_viscosity = False
        """bool: Cast eddy viscosity to p1 space instead of p0"""
        self.turbulence_model = 'gls'
        """
        str: Defines the type of vertical turbulence model.

        Currently only 'gls' is supported
        """
        self.gls_options = GLSModelOptions()
        """:class:`.GLSModelOptions`: Dictionary of default GLS model options"""
        self.use_turbulence_advection = False
        """bool: Advect TKE and Psi in the GLS turbulence model"""
        self.baroclinic = False
        """bool: Compute internal pressure gradient in momentum equation"""
        self.smagorinsky_factor = None
        """
        :class:`Constant` or None: Smagorinsky viscosity factor :math:`C_S`

        See :class:`.SmagorinskyViscosity`.
        """
        self.use_limiter_for_tracers = False
        """bool: Apply P1DG limiter for tracer fields"""
        self.uv_lax_friedrichs = Constant(1.0)
        """:class:`Constant` or None: Scaling factor for uv L-F stability term."""
        self.tracer_lax_friedrichs = Constant(1.0)
        """:class:`Constant` or None: Scaling factor for tracer L-F stability term."""
        self.check_vol_conservation_2d = False
        """
        bool: Compute volume of the 2D mode at every export

        2D volume is defined as the integral of the water elevation field.
        Prints deviation from the initial volume to stdout.
        """
        self.check_vol_conservation_3d = False
        """
        bool: Compute volume of the 2D domain at every export

        Prints deviation from the initial volume to stdout.
        """
        self.check_salt_conservation = False
        """
        bool: Compute total salinity mass at every export

        Prints deviation from the initial mass to stdout.
        """
        self.check_salt_overshoot = False
        """
        bool: Compute salinity overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """
        self.check_temp_conservation = False
        """
        bool: Compute total temperature mass at every export

        Prints deviation from the initial mass to stdout.
        """
        self.check_temp_overshoot = False
        """
        bool: Compute temperature overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """
        self.log_output = True
        """bool: Redirect all output to log file in output directory"""
        self.dt = None
        """
        float: Time step.

        If set, overrides automatically computed stable dt
        """
        self.dt_2d = None
        """
        float: Time step of the 2d mode

        If set overrides automatically computed stable dt
        """
        self.cfl_2d = 1.0
        """float: Factor to scale the 2d time step"""
        # TODO OBSOLETE
        self.cfl_3d = 1.0
        """float: Factor to scale the 2d time step"""
        # TODO OBSOLETE
        self.t_export = 100.0
        """
        float: Export interval in seconds

        All fields in fields_to_export list will be stored to disk and
        diagnostics will be computed
        """
        self.t_end = 1000.0
        """float: Simulation duration in seconds"""
        self.u_advection = Constant(0.1)
        """
        :class:`Constant`: Maximum horizontal velocity magnitude

        Used to compute max stable advection time step.
        """
        self.w_advection = Constant(1e-4)
        """
        :class:`Constant`: Maximum vertical velocity magnitude

        Used to compute max stable advection time step.
        """
        self.nu_viscosity = Constant(1.0)
        """
        :class:`Constant`: Maximum horizontal viscosity

        Used to compute max stable diffusion time step.
        """
        self.outputdir = 'outputs'
        """str: Directory where model output files are stored"""
        self.no_exports = False
        """
        bool: Do not store any outputs to disk

        Disables VTK and HDF5 field outputs. and HDF5 diagnostic outputs.
        Used in CI test suite.
        """
        self.export_diagnostics = True
        """bool: Store diagnostic variables to disk in HDF5 format"""
        self.equation_of_state = 'full'
        """str: type of equation of state that defines water density

        Either 'full' or 'linear'. See :class:`.JackettEquationOfState` and
        :class:`.LinearEquationOfState`.
        """
        self.lin_equation_of_state_params = {
            'rho_ref': 1000.0,
            's_ref': 35.0,
            'th_ref': 15.0,
            'alpha': 0.2,
            'beta': 0.77,
        }
        """dict: Parameters for linear equation of state"""
        self.use_quadratic_pressure = False
        """
        bool: use P2DGxP2 space for baroclinic head.

        If element_family='dg-dg', P2DGxP1DG space is also used for the internal
        pressure gradient.

        This is useful to alleviate bathymetry-induced pressure gradient errors.
        If False, the baroclinic head is in the tracer space, and internal
        pressure gradient is in the velocity space.
        """
        self.use_quadratic_density = False
        """
        bool: water density is projected to P2DGxP2 space.

        This reduces pressure gradient errors associated with nonlinear
        equation of state.
        If False, density is computed point-wise in the tracer space.
        """
        self.fields_to_export = ['elev_2d', 'uv_2d', 'uv_3d', 'w_3d']
        """list of str: Fields to export in VTK format"""
        self.fields_to_export_hdf5 = []
        """list of str: Fields to export in HDF5 format"""
        self.verbose = 0
        """int: Verbosity level"""
        # NOTE these are fields, potentially Functions: move out of this class?
        self.linear_drag = None
        r"""
        Coefficient or None: 2D linear drag parameter :math:`L`

        Bottom stress is :math:`\tau_b/\rho_0 = -L \mathbf{u} H`
        """
        self.quadratic_drag = None
        r"""Coefficient or None: dimensionless 2D quadratic drag parameter :math:`C_D`

        Bottom stress is :math:`\tau_b/\rho_0 = -C_D |\mathbf{u}|\mathbf{u}`
        """
        self.mu_manning = None
        r"""Coefficient or None: Manning-Strickler 2D quadratic drag parameter :math:`\mu`

        Bottom stress is :math:`\tau_b/\rho_0 = -g \mu^2 |\mathbf{u}|\mathbf{u}/H^{1/3}`
        """
        self.wd_alpha = None
        r"""Coefficient or None: Wetting-drying parameter :math:`\alpha`

        Used in bathymetry displacement function that ensures positive water depths. Unit is meters.
        Default is None, which disables wetting and drying.
        """
        self.h_diffusivity = None
        """Coefficient or None: Background horizontal diffusivity for tracers"""
        self.v_diffusivity = None
        """Coefficient or None: background vertical diffusivity for tracers"""
        self.h_viscosity = None
        """Coefficient or None: background horizontal viscosity"""
        self.v_viscosity = None
        """Coefficient or None: background vertical viscosity"""
        self.coriolis = None
        """2D Coefficient or None: Coriolis parameter"""
        self.wind_stress = None
        """Coefficient or None: Stress at free surface (2D vector function)"""
        self.atmospheric_pressure = None
        """Coefficient or None: Atmospheric pressure at free surface, in pascals"""
        self.uv_source_2d = None
        """Coefficient or None: source term for 2D momentum equation"""
        self.uv_source_3d = None
        """Coefficient or None: source term for 3D momentum equation"""
        self.elev_source_2d = None
        """Coefficient or None: source term for 2D continuity equation"""
        self.salt_source_3d = None
        """Coefficient or None: source term for salinity equation"""
        self.temp_source_3d = None
        """Coefficient or None: source term for temperature equation"""
        self.constant_temp = Constant(10.0)
        """Coefficient: Constant temperature if temperature is not solved"""
        self.constant_salt = Constant(0.0)
        """Coefficient: Constant salinity if salinity is not solved"""
        self.solver_parameters_sw = {
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
        }
        """PETSc solver parameters for 2D shallow water equations"""
        self.solver_parameters_sw_momentum = {
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'sor',
        }
        """PETSc solver parameters for 2D depth averaged momentum equation"""
        self.solver_parameters_momentum_explicit = {
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        }
        """PETSc solver parameters for explicit 3D momentum equation"""
        self.solver_parameters_momentum_implicit = {
            'snes_monitor': False,
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        }
        """PETSc solver parameters for implicit 3D momentum equation"""
        self.solver_parameters_tracer_explicit = {
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        }
        """PETSc solver parameters for explicit 3D tracer equations"""
        self.solver_parameters_tracer_implicit = {
            'snes_monitor': False,
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        }
        """PETSc solver parameters for implicit 3D tracer equations"""

        self._isfrozen = True
