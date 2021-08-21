"""
Thetis options for the 2D and 3D model

All options are type-checked and they are stored in traitlets Configurable
objects.
"""
from .configuration import *
from firedrake import Constant
from .sediment_model import SedimentModel
from collections import OrderedDict


class TimeStepperOptions(FrozenHasTraits):
    """Base class for all time stepper options"""
    name = 'Time stepper'
    solver_parameters = PETScSolverParameters({}).tag(config=True)


class ExplicitTimeStepperOptions(TimeStepperOptions):
    """Options for explicit time integrator"""
    use_automatic_timestep = Bool(True, help='Set time step automatically based on local CFL conditions.').tag(config=True)


class ExplicitTimeStepperOptions2d(ExplicitTimeStepperOptions):
    """Options for 2d explicit time integrator"""


class SemiImplicitTimeStepperOptions2d(TimeStepperOptions):
    """Options for 2d semi-implicit time integrator"""
    use_semi_implicit_linearization = Bool(
        False, help="Use linearized semi-implicit time integration").tag(config=True)


class SemiImplicitSWETimeStepperOptions2d(SemiImplicitTimeStepperOptions2d):
    """
    Options for 2d semi-implicit time integrator
    applied to shallow water equations
    """
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
    }).tag(config=True)


class SemiImplicitTracerTimeStepperOptions2d(SemiImplicitTimeStepperOptions2d):
    """
    Options for 2d semi-implicit time integrator
    applied to tracer equations
    """
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'sor',
    }).tag(config=True)


class SteadyStateTimeStepperOptions2d(TimeStepperOptions):
    """Options for 2d steady state solver"""
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'mat_type': 'aij'
    }).tag(config=True)


class CrankNicolsonSWETimeStepperOptions2d(SemiImplicitSWETimeStepperOptions2d):
    """
    Options for 2d Crank-Nicolson time integrator
    applied to shallow water equations
    """
    implicitness_theta = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta. Value 0.5 implies Crank-Nicolson scheme, 1.0 implies fully implicit formulation.').tag(config=True)


class CrankNicolsonTracerTimeStepperOptions2d(SemiImplicitTracerTimeStepperOptions2d):
    """
    Options for 2d Crank-Nicolson time integrator
    applied to tracer equations
    """
    implicitness_theta = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta. Value 0.5 implies Crank-Nicolson scheme, 1.0 implies fully implicit formulation.').tag(config=True)


class PressureProjectionSWETimeStepperOptions2d(TimeStepperOptions):
    """
    Options for 2d pressure-projection time integrator
    applied to shallow water equations
    """
    solver_parameters_pressure = PETScSolverParameters({
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_U_2d': {
            'ksp_type': 'gmres',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_ksp_type': 'preonly',
            'assembled_pc_type': 'bjacobi',
            'assembled_sub_pc_type': 'ilu',
        },
        # schur system: explicitly assemble the schur system
        # this only works with pressureprojectionicard if the velocity block is just the mass matrix
        # and if the velocity is DG so that this mass matrix can be inverted explicitly
        'fieldsplit_H_2d': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'thetis.AssembledSchurPC',
            'schur_ksp_type': 'gmres',
            'schur_ksp_max_it': 100,
            'schur_pc_type': 'gamg',
        },
    }).tag(config=True)
    solver_parameters_momentum = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'sor',
    }).tag(config=True)
    implicitness_theta = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta. Value 0.5 implies Crank-Nicolson scheme, 1.0 implies fully implicit formulation.').tag(config=True)
    use_semi_implicit_linearization = Bool(
        True, help="Use linearized semi-implicit time integration").tag(config=True)
    picard_iterations = PositiveInteger(
        default_value=2,
        help='number of Picard iterations to converge the nonlinearity in the equations.')


class ExplicitSWETimeStepperOptions2d(ExplicitTimeStepperOptions2d):
    """
    Options for 2d explicit time integrator
    applied to shallow water equations
    """
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
        'mat_type': 'aij',
    }).tag(config=True)


class ExplicitTracerTimeStepperOptions2d(ExplicitTimeStepperOptions2d):
    """
    Options for 2d explicit time integrator
    applied to tracer equations
    """
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'sor',
    }).tag(config=True)


class IMEXSWETimeStepperOptions2d(SemiImplicitTimeStepperOptions2d):
    """
    Options for 2d implicit-explicit time integrators
    applied to shallow water equations
    """
    # TODO: Meaningful solver parameters for DIRK
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
    }).tag(config=True)


class TimeStepperOptions3d(FrozenHasTraits):
    """Options for 3d explicit time integrator"""


class SWETimeStepperOptions3d(TimeStepperOptions3d):
    """
    Options for 3d time integrator
    shallow water component
    """
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
    }).tag(config=True)


class ImplicitSWETimeStepperOptions3d(SWETimeStepperOptions3d):
    """
    Options for 3d implicit time integrator
    shallow water component
    """
    implicitness_theta = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta. Value 0.5 implies Crank-Nicolson scheme, 1.0 implies fully implicit formulation.').tag(config=True)


class ExplicitMomentumTimeStepperOptions3d(TimeStepperOptions3d):
    """
    Options for 3d explicit time integrator
    momentum component
    """
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)


class ImplicitMomentumTimeStepperOptions3d(TimeStepperOptions3d):
    """
    Options for 3d implicit time integrator
    momentum component
    """
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)


class ExplicitTracerTimeStepperOptions3d(TimeStepperOptions3d):
    """
    Options for 3d explicit time integrator
    applied to tracer equations
    """
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)


class ImplicitTracerTimeStepperOptions3d(TimeStepperOptions3d):
    """
    Options for 3d implicit time integrator
    applied to tracer equations
    """
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)


class SSPRKTimeStepperOptions3d(TimeStepperOptions3d):
    """Options for 3d SSPRK coupled time integrator"""
    use_automatic_timestep = Bool(True, help='Set time step automatically based on local CFL conditions.').tag(config=True)
    swe_options = SWETimeStepperOptions3d()
    implicit_momentum_options = ImplicitMomentumTimeStepperOptions3d()
    explicit_momentum_options = ExplicitMomentumTimeStepperOptions3d()
    implicit_tracer_options = ImplicitTracerTimeStepperOptions3d()
    explicit_tracer_options = ExplicitTracerTimeStepperOptions3d()


class LeapFrogTimeStepperOptions3d(SSPRKTimeStepperOptions3d):
    """Options for 3d LeapFrog coupled time integrator"""
    swe_options = ImplicitSWETimeStepperOptions3d()


class TurbulenceModelOptions(FrozenHasTraits):
    """Abstract base class for all turbulence model options"""
    name = 'Turbulence closure model'


class PacanowskiPhilanderModelOptions(TurbulenceModelOptions):
    """Options for Pacanowski-Philander turbulence model"""
    name = 'Pacanowski-Philander turbulence closure model'
    max_viscosity = PositiveFloat(5e-2, help=r"float: Constant maximum viscosity :math:`\nu_{max}`").tag(config=True)
    alpha = PositiveFloat(10.0, help="float: Richardson number multiplier").tag(config=True)
    exponent = PositiveFloat(2.0, help=r"float: Exponent of viscosity numerator :math:`n`").tag(config=True)


class GLSModelOptions(TurbulenceModelOptions):
    """Options for Generic Length Scale turbulence model"""
    name = 'Generic Length Scale turbulence closure model'
    closure_name = Enum(
        ['k-epsilon', 'k-omega', 'Generic Length Scale'],
        default_value='k-epsilon',
        help='Name of two-equation closure').tag(config=True)
    stability_function_name = Enum(
        ['Canuto A', 'Canuto B', 'Kantha-Clayson', 'Cheng'],
        default_value='Canuto A',
        help='Name of stability function family').tag(config=True)
    p = Float(3.0,
              help='float: parameter p for the definition of psi').tag(config=True)
    m = Float(1.5,
              help='float: parameter m for the definition of psi').tag(config=True)
    n = Float(-1.0,
              help='float: parameter n for the definition of psi').tag(config=True)
    schmidt_nb_tke = PositiveFloat(1.0,
                                   help='float: turbulent kinetic energy Schmidt number').tag(config=True)
    schmidt_nb_psi = PositiveFloat(1.3,
                                   help='float: psi Schmidt number').tag(config=True)
    cmu0 = PositiveFloat(0.5477,
                         help='float: cmu0 parameter').tag(config=True)
    compute_cmu0 = Bool(
        True, help="""bool: compute cmu0 from stability function parameters

        If :attr:`compute_cmu0` is True, this value will be overriden""").tag(config=True)
    c1 = Float(1.44, help='float: c1 parameter for Psi equations').tag(config=True)
    c2 = Float(1.92, help='float: c2 parameter for Psi equations').tag(config=True)
    c3_minus = Float(
        -0.52, help="""float: c3 parameter for Psi equations, stable stratification

        If :attr:`compute_c3_minus` is True this value will be overriden""").tag(config=True)
    c3_plus = Float(1.0,
                    help='float: c3 parameter for Psi equations, unstable stratification').tag(config=True)
    compute_c3_minus = Bool(True,
                            help='bool: compute :attr:`c3_minus` from :attr:`ri_st`').tag(config=True)
    f_wall = Float(1.0, help='float: wall function parameter').tag(config=True)
    ri_st = Float(0.25, help='steady state gradient Richardson number').tag(config=True)
    # FIXME should default to physical_constants['von_karman'] ??
    # FIXME remove the dualism of von Karman constant
    kappa = Float(
        0.4, help="""float: von Karman constant

        If :attr:`compute_kappa` is True this value will be overriden""").tag(config=True)
    compute_kappa = Bool(False,
                         help='bool: compute von Karman constant from :attr:`schmidt_nb_psi`').tag(config=True)
    compute_schmidt_nb_psi = Bool(True,
                                  help='bool: compute psi Schmidt number').tag(config=True)
    k_min = PositiveFloat(1.0e-6,
                          help='float: minimum value for turbulent kinetic energy').tag(config=True)
    psi_min = PositiveFloat(1.0e-14, help='float: minimum value for psi').tag(config=True)
    eps_min = PositiveFloat(1.0e-14, help='float: minimum value for epsilon').tag(config=True)
    len_min = PositiveFloat(1.0e-12,
                            help='float: minimum value for turbulent length scale').tag(config=True)
    compute_galperin_clim = Bool(True,
                                 help='bool: compute c_lim length scale limiting factor').tag(config=True)
    compute_len_min = Bool(False,
                           help='bool: compute min_len from k_min and psi_min').tag(config=True)
    compute_psi_min = Bool(False,
                           help='bool: compute psi_len from k_min and eps_min').tag(config=True)
    visc_min = PositiveFloat(1.0e-8,
                             help='float: minimum value for eddy viscosity').tag(config=True)
    diff_min = PositiveFloat(1.0e-8,
                             help='float: minimum value for eddy diffusivity').tag(config=True)
    galperin_clim = PositiveFloat(0.30,
                                  help='float: Galperin length scale limitation parameter').tag(config=True)

    limit_len = Bool(False, help='bool: apply Galperin length scale limit').tag(config=True)
    limit_psi = Bool(True,
                     help='bool: apply Galperin length scale limit on psi').tag(config=True)
    limit_eps = Bool(False,
                     help='bool: apply Galperin length scale limit on epsilon').tag(config=True)
    limit_len_min = Bool(True,
                         help='bool: limit minimum turbulent length scale to len_min').tag(config=True)

    def apply_defaults(self, closure_name):
        """
        Applies default parameters for given closure name

        :arg closure_name: name of the turbulence closure model
        :type closure_name: string

        Sets default values for parameters p, m, n, schmidt_nb_tke,
        schmidt_nb_psi, c1, c2, c3_plus, c3_minus,
        f_wall, k_min, psi_min
        """

        kepsilon = {'p': 3,
                    'm': 1.5,
                    'n': -1.0,
                    'cmu0': 0.5477,
                    'schmidt_nb_tke': 1.0,
                    'schmidt_nb_psi': 1.3,
                    'c1': 1.44,
                    'c2': 1.92,
                    'c3_plus': 1.0,
                    'c3_minus': -0.52,
                    'f_wall': 1.0,
                    'k_min': 1.0e-6,
                    'eps_min': 1.0e-14,
                    'psi_min': 1.0e-14,
                    'closure_name': 'k-epsilon',
                    }
        # k-epsilon defaults, from tables 1 and 2 in [3]
        komega = {'p': -1.0,
                  'm': 0.5,
                  'n': -1.0,
                  'cmu0': 0.5477,
                  'schmidt_nb_tke': 2.0,
                  'schmidt_nb_psi': 2.0,
                  'c1': 0.555,
                  'c2': 0.833,
                  'c3_plus': 1.0,
                  'c3_minus': -0.52,
                  'f_wall': 1.0,
                  'k_min': 7.6e-6,
                  'eps_min': 1.0e-14,
                  'psi_min': 1.0e-14,
                  'closure_name': 'k-omega',
                  }
        # k-omega defaults, from tables 1 and 2 in [3]
        gen = {'p': 2.0,
               'm': 1.0,
               'n': -0.67,
               'cmu0': 0.5477,
               'schmidt_nb_tke': 0.8,
               'schmidt_nb_psi': 1.07,
               'c1': 1.0,
               'c2': 1.22,
               'c3_plus': 1.0,
               'c3_minus': 0.05,
               'f_wall': 1.0,
               'k_min': 1.0e-6,
               'eps_min': 1.0e-14,
               'psi_min': 1.0e-14,
               'closure_name': 'Generic Length Scale',
               }
        # GLS model A defaults, from tables 1 and 2 in [3]

        if closure_name == 'k-epsilon':
            self.update(kepsilon)
        elif closure_name == 'k-omega':
            self.update(komega)
        elif closure_name == 'Generic Length Scale':
            self.update(gen)
        else:
            raise ValueError('Unknown closure name "{:}"'.format(closure_name))


class EquationOfStateOptions(FrozenHasTraits):
    """Base class of equation of state options"""
    name = 'Equation of State'


class LinearEquationOfStateOptions(EquationOfStateOptions):
    """Linear equation of state options"""
    # TODO more human readable parameter names
    # TODO document the actual equation somewhere
    name = 'Linear Equation of State'
    rho_ref = NonNegativeFloat(1000.0, help='Reference water density').tag(config=True)
    s_ref = NonNegativeFloat(35.0, help='Reference water salinity').tag(config=True)
    th_ref = Float(15.0, help='Reference water temperature').tag(config=True)
    alpha = Float(0.2, help='Thermal expansion coefficient of ocean water').tag(config=True)
    beta = Float(0.77, help='Saline contraction coefficient of ocean water').tag(config=True)


class TidalTurbineOptions(FrozenHasTraits):
    """Tidal turbine parameters"""
    thrust_coefficient = PositiveFloat(
        0.8, help='Thrust coefficient C_T').tag(config=True)
    diameter = PositiveFloat(
        18., help='Turbine diameter').tag(config=True)


class TidalTurbineFarmOptions(FrozenHasTraits, TraitType):
    """Tidal turbine farm options"""
    name = 'Farm options'
    turbine_options = TidalTurbineOptions()
    turbine_density = FiredrakeScalarExpression(
        Constant(0.0), help='Density of turbines within the farm')
    break_even_wattage = NonNegativeFloat(
        0.0, help='Average power production per turbine required to break even')


class TracerFieldOptions(FrozenHasTraits):
    """Tracer field options"""
    name = 'Tracer options'
    function = FiredrakeScalarExpression(
        None, allow_none=True, help='Firedrake Function representing the tracer')
    source = FiredrakeScalarExpression(
        None, allow_none=True, help='Source term for the tracer equation')
    diffusivity = FiredrakeScalarExpression(
        None, allow_none=True, help='Diffusion coefficient for the tracer equation')
    use_conservative_form = Bool(
        False, help='Should the tracer equation be solved in conservative form?')
    metadata = Dict({
        'label': 'tracer_2d',
        'name': 'Depth averaged tracer',
        'filename': 'Tracer2d',
        'shortname': 'Tracer',
        'unit': '-',
    }, help='Dictionary of metadata for the tracer field')


@attach_paired_options("free_surface_timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitTracerTimeStepperOptions2d),
                                   ('ForwardEuler', ExplicitTracerTimeStepperOptions2d),
                                   ('BackwardEuler', SemiImplicitTracerTimeStepperOptions2d),
                                   ('CrankNicolson', CrankNicolsonTracerTimeStepperOptions2d),
                                   ('DIRK22', SemiImplicitTracerTimeStepperOptions2d),
                                   ('DIRK33', SemiImplicitTracerTimeStepperOptions2d),
                                   ],
                                  "free_surface_timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the free surface time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
class NonhydrostaticModelOptions(FrozenHasTraits):
    """Options for non-hydrostatic models"""
    name = 'Non-hydrostatic 2D/3D models'
    solve_nonhydrostatic_pressure = Bool(False, help='Solve equations with the non-hydrostatic pressure').tag(config=True)
    q_degree = NonNegativeInteger(
        None, allow_none=True, help="Polynomial degree of the non-hydrostatic pressure space").tag(config=True)
    update_free_surface = Bool(True, help='Update free surface elevation after pressure projection/correction step').tag(config=True)
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'mat_type': 'aij',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_mumps_icntl_14': 200,
    }).tag(config=True)


class CommonModelOptions(FrozenConfigurable):
    """Options that are common for both 2d and 3d models"""
    name = 'Model options'
    nh_model_options = Instance(NonhydrostaticModelOptions, args=()).tag(config=True)
    polynomial_degree = NonNegativeInteger(1, help='Polynomial degree of elements').tag(config=True)
    element_family = Enum(
        ['dg-dg', 'rt-dg', 'bdm-dg', 'dg-cg'],
        default_value='dg-dg',
        help="""Finite element family

        2D solver supports 'dg-dg', 'rt-dg', 'bdm-dg', or 'dg-cg' velocity-pressure pairs.
        3D solver supports 'dg-dg', 'rt-dg', or 'bdm-dg' velocity-pressure pairs.""").tag(config=True)

    use_nonlinear_equations = Bool(True, help='Use nonlinear shallow water equations').tag(config=True)
    use_grad_div_viscosity_term = Bool(
        False,
        help=r"""Include :math:`\nabla (\nu_h \nabla \cdot \bar{\textbf{u}})` term in the depth-averaged viscosity

        See :class:`.shallowwater_eq.HorizontalViscosityTerm` for details.""").tag(config=True)
    use_grad_depth_viscosity_term = Bool(
        True,
        help=r"""Include :math:`\nabla H` term in the depth-averaged viscosity

        See :class:`.shallowwater_eq.HorizontalViscosityTerm` for details.""").tag(config=True)

    use_lax_friedrichs_velocity = Bool(
        True, help="use Lax Friedrichs stabilisation in horizontal momentum advection.").tag(config=True)
    lax_friedrichs_velocity_scaling_factor = FiredrakeConstantTraitlet(
        Constant(1.0), help="Scaling factor for Lax Friedrichs stabilisation term in horizontal momentum advection.").tag(config=True)
    use_lax_friedrichs_tracer = Bool(
        False, help="Use Lax Friedrichs stabilisation in tracer advection.").tag(config=True)
    lax_friedrichs_tracer_scaling_factor = FiredrakeConstantTraitlet(
        Constant(1.0), help="Scaling factor for tracer Lax Friedrichs stability term.").tag(config=True)
    use_limiter_for_tracers = Bool(
        True, help="Apply P1DG limiter for tracer fields").tag(config=True)

    check_volume_conservation_2d = Bool(
        False, help="""
        Compute volume of the 2D mode at every export

        2D volume is defined as the integral of the water elevation field.
        Prints deviation from the initial volume to stdout.
        """).tag(config=True)
    log_output = Bool(
        True, help="Redirect all output to log file in output directory").tag(config=True)
    timestep = PositiveFloat(
        10.0, help="Time step").tag(config=True)
    cfl_2d = PositiveFloat(
        1.0, help="Factor to scale the 2d time step OBSOLETE").tag(config=True)  # TODO OBSOLETE
    cfl_3d = PositiveFloat(
        1.0, help="Factor to scale the 2d time step OBSOLETE").tag(config=True)  # TODO OBSOLETE
    simulation_export_time = PositiveFloat(
        100.0, help="""
        Export interval in seconds

        All fields in fields_to_export list will be stored to disk and
        diagnostics will be computed
        """).tag(config=True)
    simulation_end_time = PositiveFloat(
        1000.0, help="Simulation duration in seconds").tag(config=True)
    horizontal_velocity_scale = FiredrakeConstantTraitlet(
        Constant(0.1), help="""
        Maximum horizontal velocity magnitude

        Used to compute max stable advection time step.
        """).tag(config=True)
    horizontal_viscosity_scale = FiredrakeConstantTraitlet(
        Constant(1.0), help="""
        Maximum horizontal viscosity

        Used to compute max stable diffusion time step.
        """).tag(config=True)
    horizontal_diffusivity_scale = FiredrakeConstantTraitlet(
        Constant(1.0), help="""
        Maximum horizontal diffusivity

        Used to compute the mesh Peclet number in
        the 2D tracer SUPG stabilization scheme.
        """).tag(config=True)
    output_directory = Unicode(
        'outputs', help="Directory where model output files are stored").tag(config=True)
    no_exports = Bool(
        False, help="""
        Do not store any outputs to disk

        Disables VTK and HDF5 field outputs. and HDF5 diagnostic outputs.
        Used in CI test suite.
        """).tag(config=True)
    export_diagnostics = Bool(
        True, help="Store diagnostic variables to disk in HDF5 format").tag(config=True)
    fields_to_export = List(
        trait=Unicode(),
        default_value=['elev_2d', 'uv_2d', 'uv_3d', 'w_3d'],
        help="Fields to export in VTK format").tag(config=True)
    fields_to_export_hdf5 = List(
        trait=Unicode(),
        default_value=[],
        help="Fields to export in HDF5 format").tag(config=True)
    verbose = Integer(0, help="Verbosity level").tag(config=True)
    linear_drag_coefficient = FiredrakeScalarExpression(
        None, allow_none=True, help=r"""
        2D linear drag parameter :math:`L`

        Bottom stress is :math:`\tau_b/\rho_0 = -L \mathbf{u} H`
        """).tag(config=True)
    quadratic_drag_coefficient = FiredrakeScalarExpression(
        None, allow_none=True, help=r"""
        Dimensionless 2D quadratic drag parameter :math:`C_D`

        Bottom stress is :math:`\tau_b/\rho_0 = -C_D |\mathbf{u}|\mathbf{u}`
        """).tag(config=True)
    manning_drag_coefficient = FiredrakeScalarExpression(
        None, allow_none=True, help=r"""
        Manning-Strickler 2D quadratic drag parameter :math:`\mu`

        Bottom stress is :math:`\tau_b/\rho_0 = -g \mu^2 |\mathbf{u}|\mathbf{u}/H^{1/3}`
        """).tag(config=True)
    nikuradse_bed_roughness = FiredrakeScalarExpression(
        None, allow_none=True, help=r"""
        Nikuradse bed roughness length used to construct the 2D quadratic drag parameter :math:`C_D`.

        In sediment transport this term is usually three times the average sediment diameter size.
        """).tag(config=True)
    norm_smoother = FiredrakeConstantTraitlet(
        Constant(0.0), help=r"""
        Coefficient used to avoid non-differentiable functions in the continuous formulation of the velocity norm in
        the quadratic bottom drag term in the momentum equation. This replaces the velocity norm in the quadratic
        bottom drag term with :math:`\|u\| \approx \sqrt{\|u\|^2 + \alpha^2}`
        """).tag(config=True)
    horizontal_viscosity = FiredrakeScalarExpression(
        None, allow_none=True, help="Horizontal viscosity").tag(config=True)
    coriolis_frequency = FiredrakeScalarExpression(
        None, allow_none=True, help="2D Coriolis parameter").tag(config=True)
    wind_stress = FiredrakeVectorExpression(
        None, allow_none=True, help="Stress at free surface (2D vector function)").tag(config=True)
    atmospheric_pressure = FiredrakeScalarExpression(
        None, allow_none=True, help="Atmospheric pressure at free surface, in pascals").tag(config=True)
    momentum_source_2d = FiredrakeVectorExpression(
        None, allow_none=True, help="Source term for 2D momentum equation").tag(config=True)
    volume_source_2d = FiredrakeScalarExpression(
        None, allow_none=True, help="Source term for 2D continuity equation").tag(config=True)
    sipg_factor = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for horizontal viscosity terms.").tag(config=True)
    sipg_factor_tracer = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for horizontal diffusivity terms.").tag(config=True)


@attach_paired_options("sediment_timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitTracerTimeStepperOptions2d),
                                   ('ForwardEuler', ExplicitTracerTimeStepperOptions2d),
                                   ('BackwardEuler', SemiImplicitTracerTimeStepperOptions2d),
                                   ('CrankNicolson', CrankNicolsonTracerTimeStepperOptions2d),
                                   ('DIRK22', SemiImplicitTracerTimeStepperOptions2d),
                                   ('DIRK33', SemiImplicitTracerTimeStepperOptions2d),
                                   ],
                                  "sediment_timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the sediment time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
@attach_paired_options("exner_timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitTracerTimeStepperOptions2d),
                                   ('ForwardEuler', ExplicitTracerTimeStepperOptions2d),
                                   ('BackwardEuler', SemiImplicitTracerTimeStepperOptions2d),
                                   ('CrankNicolson', CrankNicolsonTracerTimeStepperOptions2d),
                                   ('DIRK22', SemiImplicitTracerTimeStepperOptions2d),
                                   ('DIRK33', SemiImplicitTracerTimeStepperOptions2d),
                                   ],
                                  "exner_timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the exner time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
class SedimentModelOptions(FrozenHasTraits):
    solve_exner = Bool(False, help='Solve exner equation for bed morphology').tag(config=True)
    solve_suspended_sediment = Bool(False, help='Solve suspended sediment transport equation').tag(config=True)
    use_sediment_conservative_form = Bool(False, help='Solve 2D sediment transport in the conservative form').tag(config=True)
    use_bedload = Bool(False, help='Use bedload transport in sediment model').tag(config=True)
    use_sediment_slide = Bool(False, help='Use sediment slide mechanism in sediment model').tag(config=True)
    horizontal_diffusivity = FiredrakeCoefficient(
        None, allow_none=True, help="Horizontal diffusivity for sediment").tag(config=True)
    use_angle_correction = Bool(True, help='Switch to use slope effect angle correction').tag(config=True)
    use_slope_mag_correction = Bool(True, help='Switch to use slope effect magnitude correction').tag(config=True)
    use_secondary_current = Bool(False, help='Switch to use secondary current for helical flow effect').tag(config=True)
    average_sediment_size = FiredrakeScalarExpression(None, allow_none=True, help='Average sediment size').tag(config=True)
    slide_region = FiredrakeScalarExpression(None, allow_none=True, help="""Region where sediment slide occurs.

                                             If None then sediment slide is applied over whole domain.
                                             """).tag(config=True)
    bed_reference_height = FiredrakeScalarExpression(None, allow_none=True, help='Bottom bed reference height').tag(config=True)
    use_advective_velocity_correction = Bool(True, help="""
        Switch to apply correction to advective velocity used in sediment equation

        Accounts for mismatch between depth-averaged product of velocity with sediment
        and product of depth-averaged velocity with depth-averaged sediment
        """).tag(config=True)
    porosity = FiredrakeCoefficient(
        Constant(0.4), help="Bed porosity for exner equation").tag(config=True)
    max_angle = FiredrakeConstantTraitlet(
        Constant(32), help="Angle of repose for sediment slide mechanism in degrees").tag(config=True)
    sed_slide_length_scale = FiredrakeConstantTraitlet(
        Constant(0), help="""Length scale for sediment slide mechanism.

                           This should normally be the average meshgrid size.
                           """).tag(config=True)
    morphological_acceleration_factor = FiredrakeConstantTraitlet(
        Constant(1), help="""Rate at which timestep in exner equation is accelerated compared to timestep for model

        timestep in exner = morphological_acceleration_factor * timestep
        """).tag(config=True)
    morphological_viscosity = FiredrakeScalarExpression(
        None, allow_none=True, help="""Viscosity used to derive morphology terms.

        Usually equal to horizontal viscosity but can be set to have a different value""").tag(config=True)
    sediment_density = FiredrakeScalarExpression(
        Constant(2650), help='Density of sediment').tag(config=True)
    secondary_current_parameter = FiredrakeConstantTraitlet(
        Constant(0.75), help='Parameter controlling secondary current').tag(config=True)
    slope_effect_parameter = FiredrakeConstantTraitlet(
        Constant(1.3), help='Parameter controlling magnitude of slope effect').tag(config=True)
    slope_effect_angle_parameter = FiredrakeConstantTraitlet(
        Constant(2/3), help='Parameter controlling angle of slope effect').tag(config=True)
    check_sediment_conservation = Bool(
        False, help="""
        Compute total sediment mass at every export

        Prints deviation from the initial mass to stdout.
        """).tag(config=True)
    check_sediment_overshoot = Bool(
        False, help="""
        Compute sediment overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """).tag(config=True)
    sediment_model_class = Type(SedimentModel, help="""Class used to define the sediment model

    This option can be used to provide a user-defined sediment model class that should
    be a subclass of SedimentModel. For example:

    .. code-block:: python

        class UserSedimentModel(SedimentModel):
           def __init__(options, mesh2d, uv, elev, depth, extra_term):
              super().__init__(options, mesh2d, uv, elev, depth)
              self.extra_term = extra_term

           def get_bedloadterm(self, bathymetry):
              return super().get_bedloadterm(bathymetry) + self.term
    """)


@attach_paired_options("swe_timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitSWETimeStepperOptions2d),
                                   ('ForwardEuler', ExplicitSWETimeStepperOptions2d),
                                   ('BackwardEuler', SemiImplicitSWETimeStepperOptions2d),
                                   ('CrankNicolson', CrankNicolsonSWETimeStepperOptions2d),
                                   ('DIRK22', SemiImplicitSWETimeStepperOptions2d),
                                   ('DIRK33', SemiImplicitSWETimeStepperOptions2d),
                                   ('SteadyState', SteadyStateTimeStepperOptions2d),
                                   ('PressureProjectionPicard', PressureProjectionSWETimeStepperOptions2d),
                                   ('SSPIMEX', IMEXSWETimeStepperOptions2d),
                                   ],
                                  "swe_timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
@attach_paired_options("tracer_timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitTracerTimeStepperOptions2d),
                                   ('ForwardEuler', ExplicitTracerTimeStepperOptions2d),
                                   ('BackwardEuler', SemiImplicitTracerTimeStepperOptions2d),
                                   ('CrankNicolson', CrankNicolsonTracerTimeStepperOptions2d),
                                   ('DIRK22', SemiImplicitTracerTimeStepperOptions2d),
                                   ('DIRK33', SemiImplicitTracerTimeStepperOptions2d),
                                   ('SteadyState', SteadyStateTimeStepperOptions2d),
                                   ],
                                  "tracer_timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the tracer time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
class ModelOptions2d(CommonModelOptions):
    """Options for 2D depth-averaged shallow water model"""
    name = 'Depth-averaged 2D model'
    sediment_model_options = Instance(SedimentModelOptions, args=()).tag(config=True)
    use_tracer_conservative_form = Bool(False, help='Solve 2D tracer transport in the conservative form').tag(config=True)
    use_wetting_and_drying = Bool(
        False, help=r"""bool: Turn on wetting and drying

        Uses the wetting and drying scheme from Karna et al (2011).
        If ``True``, one should also set :attr:`wetting_and_drying_alpha` to control the bathymetry displacement.
        """).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(
        Constant(0.5), help=r"""
        Coefficient: Wetting and drying parameter :math:`\alpha`.

        Used in bathymetry displacement function that ensures positive water depths. Unit is meters.
        """).tag(config=True)
    use_automatic_wetting_and_drying_alpha = Bool(False, help=r"""
        Toggle automatic computation of the alpha parameter used in wetting and drying schemes.

        By default, this parameter is set to 0.5.

        For problems whose bathymetry varies wildly in coastal regions, it is advisable to use the
        automatic wetting and drying parameter, rather than the default.
        """).tag(config=True)
    wetting_and_drying_alpha_min = FiredrakeScalarExpression(
        None, allow_none=True, help=r"""
        Minimum value to be taken by wetting and drying parameter :math:`\alpha`.

        Note this is only relevant if `use_automatic_wetting_and_drying_alpha` is set to ``True``.
        """).tag(config=True)
    wetting_and_drying_alpha_max = FiredrakeScalarExpression(
        Constant(2.0), allow_none=True, help=r"""
        Maximum value to be taken by wetting and drying parameter :math:`\alpha`.

        Note this is only relevant if `use_automatic_wetting_and_drying_alpha` is set to ``True``.
        """).tag(config=True)
    tidal_turbine_farms = Dict(trait=TidalTurbineFarmOptions(),
                               default_value={}, help='Dictionary mapping subdomain ids to the options of the corresponding farm')
    check_tracer_conservation = Bool(
        False, help="""
        Compute total tracer mass at every export

        Prints deviation from the initial mass to stdout.
        """).tag(config=True)
    tracer_advective_velocity_factor = FiredrakeScalarExpression(
        Constant(1.0), help="""
        Custom factor multiplied to the velocity variable in tracer advection equation.

        Used to account for mismatch between depth-averaged product of velocity with tracer
        and product of depth-averaged velocity with depth-averaged tracer
        """).tag(config=True)
    check_tracer_overshoot = Bool(
        False, help="""
        Compute tracer overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """).tag(config=True)
    tracer_only = Bool(
        False, help="""Hold shallow water variables in initial state

        Advects tracer in the associated (constant) velocity field.
        """).tag(config=True)
    tracer_element_family = Enum(
        ['dg', 'cg'],
        default_value='dg',
        help="""Finite element family for tracer transport

        2D solver supports 'dg' or 'cg'.""").tag(config=True)
    use_supg_tracer = Bool(
        False, help="Use SUPG stabilisation in tracer advection").tag(config=True)
    tracer_picard_iterations = PositiveInteger(
        1, help="Number of Picard iterations taken for tracer equations.").tag(config=True)

    def __init__(self, *args, **kwargs):
        self.tracer = OrderedDict()
        super().__init__(*args, **kwargs)

    def add_tracer_2d(self, label, name, filename, shortname=None, unit='-', **kwargs):
        """
        Add a 2D tracer field to :attr:`tracer`.

        Note that the tracer equations will be solved in the order in which they are added.

        :arg label: field label used internally by Thetis, e.g. 'tracer_2d'
        :arg name: human readable name for the tracer field, e.g. 'Depth averaged tracer'
        :arg filename: file name for outputs, e.g. 'Tracer2d'
        :kwarg shortname: short version of name, e.g. 'Tracer'
        :kwarg unit: units for field, e.g. '-'
        :kwarg function: :class:`Function` to use for the tracer
        :kwarg source: associated source term
        :kwarg diffusivity: associated diffusivity coefficient
        :kwarg use_conservative_form: should the tracer equation be solved in conservative form?
        """
        assert isinstance(label, str)
        assert isinstance(name, str)
        assert isinstance(filename, str)
        assert shortname is None or isinstance(shortname, str)
        assert isinstance(unit, str)
        assert label not in self.tracer, f"Field '{label}' already exists."
        assert ' ' not in label, "Labels cannot contain spaces"
        assert ' ' not in filename, "Filenames cannot contain spaces"
        self.tracer[label] = TracerFieldOptions()
        self.tracer[label].metadata = {
            'name': name,
            'shortname': shortname or name,
            'unit': unit,
            'filename': filename,
        }
        self.tracer[label].function = kwargs.get('function')
        self.tracer[label].source = kwargs.get('source')
        self.tracer[label].diffusivity = kwargs.get('diffusivity')
        self.tracer[label].use_conservative_form = kwargs.get('use_conservative_form', False)

    def set_timestepper_type(self, timestepper_type, **kwargs):
        """
        Set the same timestepper type for all components.

        Any keyword arguments are passed through to the
        associated :class:`TimeStepperOptions` object.
        """
        self.swe_timestepper_type = timestepper_type
        self.tracer_timestepper_type = timestepper_type
        options = (self.swe_timestepper_options, self.tracer_timestepper_options)
        if self.sediment_model_options.solve_suspended_sediment:
            self.sediment_model_options.sediment_timestepper_type = timestepper_type
            options += (self.sediment_model_options.sediment_timestepper_options,)
        if self.sediment_model_options.solve_exner:
            self.sediment_model_options.exner_timestepper_type = timestepper_type
            options += (self.sediment_model_options.exner_timestepper_options,)
        if self.nh_model_options.solve_nonhydrostatic_pressure:
            self.nh_model_options.free_surface_timestepper_type = timestepper_type
            options += (self.nh_model_options.free_surface_timestepper_options,)
        for key, value in kwargs.items():
            for option in options:
                setattr(option, key, value)


@attach_paired_options("timestepper_type",
                       PairedEnum([('LeapFrog', LeapFrogTimeStepperOptions3d),
                                   ('SSPRK22', SSPRKTimeStepperOptions3d),
                                   ],
                                  "timestepper_options",
                                  default_value='SSPRK22',
                                  help='Name of the 3D time integrator').tag(config=True),
                       Instance(TimeStepperOptions3d, args=()).tag(config=True))
@attach_paired_options("turbulence_model_type",
                       PairedEnum([('gls', GLSModelOptions),
                                   ('pacanowski', PacanowskiPhilanderModelOptions)
                                   ],
                                  "turbulence_model_options",
                                  default_value='gls',
                                  help='Type of vertical turbulence model').tag(config=True),
                       Instance(TurbulenceModelOptions, args=()).tag(config=True))
@attach_paired_options("equation_of_state_type",
                       PairedEnum([('full', EquationOfStateOptions),
                                   ('linear', LinearEquationOfStateOptions)],
                                  "equation_of_state_options",
                                  default_value='full',
                                  help='Type of equation of state').tag(config=True),
                       Instance(EquationOfStateOptions, args=()).tag(config=True))
class ModelOptions3d(CommonModelOptions):
    """Options for 3D hydrostatic model"""
    name = '3D hydrostatic model'
    solve_salinity = Bool(True, help='Solve salinity transport').tag(config=True)
    solve_temperature = Bool(True, help='Solve temperature transport').tag(config=True)
    use_implicit_vertical_diffusion = Bool(True, help='Solve vertical diffusion and viscosity implicitly').tag(config=True)
    use_bottom_friction = Bool(True, help='Apply log layer bottom stress in the 3D model').tag(config=True)
    use_ale_moving_mesh = Bool(
        True, help="Use ALE formulation where 3D mesh tracks free surface").tag(config=True)
    use_baroclinic_formulation = Bool(
        False, help="Compute internal pressure gradient in momentum equation").tag(config=True)
    use_turbulence = Bool(
        False, help="Activate turbulence model in the 3D model").tag(config=True)
    use_turbulence_advection = Bool(
        False, help="Advect TKE and Psi in the GLS turbulence model").tag(config=True)
    use_smagorinsky_viscosity = Bool(
        False, help="Use Smagorinsky horisontal viscosity parametrization").tag(config=True)
    smagorinsky_coefficient = FiredrakeConstantTraitlet(
        Constant(0.1),
        help="""Smagorinsky viscosity coefficient :math:`C_S`

        See :class:`.SmagorinskyViscosity`.""").tag(config=True)

    use_limiter_for_velocity = Bool(
        True, help="Apply P1DG limiter for 3D horizontal velocity field").tag(config=True)
    check_volume_conservation_3d = Bool(
        False, help="""
        Compute volume of the 3D domain at every export

        Prints deviation from the initial volume to stdout.
        """).tag(config=True)
    check_salinity_conservation = Bool(
        False, help="""
        Compute total salinity mass at every export

        Prints deviation from the initial mass to stdout.
        """).tag(config=True)
    check_salinity_overshoot = Bool(
        False, help="""
        Compute salinity overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """).tag(config=True)
    check_temperature_conservation = Bool(
        False, help="""
        Compute total temperature mass at every export

        Prints deviation from the initial mass to stdout.
        """).tag(config=True)
    check_temperature_overshoot = Bool(
        False, help="""
        Compute temperature overshoots at every export

        Prints overshoot values that exceed the initial range to stdout.
        """).tag(config=True)
    timestep_2d = PositiveFloat(
        10.0, help="""
        Time step of the 2d mode

        This option is only used in the 3d solver, if 2d mode is solved
        explicitly.
        """).tag(config=True)
    vertical_velocity_scale = FiredrakeConstantTraitlet(
        Constant(1e-4), help="""
        Maximum vertical velocity magnitude

        Used to compute max stable advection time step.
        """).tag(config=True)
    use_quadratic_pressure = Bool(
        False, help="""
        Use P2DGxP2 space for baroclinic head.

        If element_family='dg-dg', P2DGxP1DG space is also used for the internal
        pressure gradient.

        This is useful to alleviate bathymetry-induced pressure gradient errors.
        If False, the baroclinic head is in the tracer space, and internal
        pressure gradient is in the velocity space.
        """).tag(config=True)
    use_quadratic_density = Bool(
        False, help="""
        Water density is projected to P2DGxP2 space.

        This reduces pressure gradient errors associated with nonlinear
        equation of state.
        If False, density is computed point-wise in the tracer space.
        """).tag(config=True)
    bottom_roughness = FiredrakeScalarExpression(
        None, allow_none=True, help="Bottom roughness length in meters.").tag(config=True)
    horizontal_diffusivity = FiredrakeScalarExpression(
        None, allow_none=True, help="Horizontal diffusivity for tracers").tag(config=True)
    vertical_diffusivity = FiredrakeScalarExpression(
        None, allow_none=True, help="Vertical diffusivity for tracers").tag(config=True)
    vertical_viscosity = FiredrakeScalarExpression(
        None, allow_none=True, help="Vertical viscosity").tag(config=True)
    momentum_source_3d = FiredrakeVectorExpression(
        None, allow_none=True, help="Source term for 3D momentum equation").tag(config=True)
    salinity_source_3d = FiredrakeScalarExpression(
        None, allow_none=True, help="Source term for salinity equation").tag(config=True)
    temperature_source_3d = FiredrakeScalarExpression(
        None, allow_none=True, help="Source term for temperature equation").tag(config=True)
    constant_temperature = FiredrakeConstantTraitlet(
        Constant(10.0), help="Constant temperature if temperature is not solved").tag(config=True)
    constant_salinity = FiredrakeConstantTraitlet(
        Constant(0.0), help="Constant salinity if salinity is not solved").tag(config=True)
    sipg_factor_vertical = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for vertical viscosity terms.").tag(config=True)
    sipg_factor_vertical_tracer = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for vertical diffusivity terms.").tag(config=True)
    sipg_factor_turb = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for horizontal diffusivity terms of the turbulence model.").tag(config=True)
    sipg_factor_vertical_turb = FiredrakeScalarExpression(
        Constant(1.0), help="Penalty parameter scaling factor for vertical diffusivity terms of the turbulence model.").tag(config=True)
    internal_pg_scalar = FiredrakeConstantTraitlet(
        None, allow_none=True, help="A constant to scale the internal pressure gradient. Used to ramp up the model.").tag(config=True)
