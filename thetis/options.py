"""
Thetis options for the 2D and 3D model

All options are type-checked and they are stored in traitlets Configurable
objects.
"""
from ipython_genutils.text import indent, dedent
from traitlets.config.configurable import Configurable
from traitlets import *

from thetis import FiredrakeConstant as Constant
from thetis import FiredrakeFunction as Function
from thetis import print_output


def rst_all_options(cls, nspace=0, prefix=None):
    """Recursively generate rST for a provided Configurable class.

    :arg cls: The Configurable class.
    :arg nspace: Indentation level.
    :arg prefix: Prefix to use for new traits."""
    lines = []
    classname = cls.__name__

    # Slaved options don't appear directly, but underneath their controlling enum.
    slaved_options = set()
    for trait in cls.class_own_traits().values():
        if isinstance(trait, PairedEnum):
            slaved_options.add(trait.paired_name)
    for k, trait in sorted(cls.class_own_traits(config=True).items()):
        typ = trait.__class__.__name__
        if trait.name in slaved_options:
            continue
        if prefix is not None:
            termline = prefix + "." + trait.name
        else:
            termline = classname + "." + trait.name

        if 'Enum' in typ:
            termline += ' : ' + '|'.join(repr(x) for x in trait.values)
        else:
            termline += ' : ' + typ
        lines.append(indent(termline, nspace))

        if isinstance(trait, PairedEnum):
            # Find the slaved option and recurse to fill in the subtree.
            dvr = trait.default_value_repr()
            extra = ["",
                     "Setting value implies configuration of sub-tree %s.%s:" % (classname, trait.paired_name),
                     ""]
            for opt, val in trait.paired_defaults.items():
                extra.append("'%s':" % opt)
                extra.append("")
                extra.append(rst_all_options(val.__class__, 4 + nspace, prefix=classname + "." + trait.paired_name))
            extra = "\n".join(extra)
        else:
            extra = None
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None
        help = trait.help or 'No description'
        lines.append(indent(dedent(help), 4 + nspace))
        lines.append('')
        lines.append(indent("Default:\n", 4 + nspace))
        lines.append(indent(dvr.replace("\\n", "\\\\n"), 4 + nspace))
        if extra is not None:
            lines.append(indent(extra, 4 + nspace))
        lines.append('')
    return "\n".join(lines)


class PositiveInteger(Integer):
    def info(self):
        return u'a positive integer'

    def validate(self, obj, proposal):
        super(PositiveInteger, self).validate(obj, proposal)
        assert proposal > 0, self.error(obj, proposal)
        return proposal


class PositiveFloat(Float):
    def info(self):
        return u'a positive float'

    def validate(self, obj, proposal):
        super(PositiveFloat, self).validate(obj, proposal)
        assert proposal > 0.0, self.error(obj, proposal)
        return proposal


class NonNegativeInteger(Integer):
    def info(self):
        return u'a non-negative integer'

    def validate(self, obj, proposal):
        super(NonNegativeInteger, self).validate(obj, proposal)
        assert proposal >= 0, self.error(obj, proposal)
        return proposal


class NonNegativeFloat(Float):
    def info(self):
        return u'a non-negative float'

    def validate(self, obj, proposal):
        super(NonNegativeFloat, self).validate(obj, proposal)
        assert proposal >= 0.0, self.error(obj, proposal)
        return proposal


class BoundedInteger(Integer):
    def __init__(self, default_value=Undefined, bounds=None, **kwargs):
        super(BoundedInteger, self).__init__(default_value, **kwargs)
        self.minval = bounds[0]
        self.maxval = bounds[1]

    def info(self):
        return u'an integer between {:} and {:}'.format(self.minval, self.maxval)

    def validate(self, obj, proposal):
        super(BoundedInteger, self).validate(obj, proposal)
        assert proposal >= self.minval, self.error(obj, proposal)
        assert proposal <= self.maxval, self.error(obj, proposal)
        return proposal


class BoundedFloat(Float):
    def __init__(self, default_value=Undefined, bounds=None, **kwargs):
        self.minval = bounds[0]
        self.maxval = bounds[1]
        super(BoundedFloat, self).__init__(default_value, **kwargs)

    def info(self):
        return u'a float between {:} and {:}'.format(self.minval, self.maxval)

    def validate(self, obj, proposal):
        super(BoundedFloat, self).validate(obj, proposal)
        assert proposal >= self.minval, self.error(obj, proposal)
        assert proposal <= self.maxval, self.error(obj, proposal)
        return proposal


class FiredrakeConstant(TraitType):
    default_value = None
    info_text = 'a Firedrake Constant'

    def validate(self, obj, value):
        if isinstance(value, Constant):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        return 'Constant({:})'.format(self.default_value.dat.data[0])


class FiredrakeCoefficient(TraitType):
    default_value = None
    info_text = 'a Firedrake Constant or Function'

    def validate(self, obj, value):
        if isinstance(value, (Constant, Function)):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        if isinstance(self.default_value, Constant):
            return 'Constant({:})'.format(self.default_value.dat.data[0])
        return 'Function'


class PETScSolverParameters(Dict):
    """PETSc solver options dictionary"""
    default_value = None
    info_text = 'a PETSc solver options dictionary'

    def validate(self, obj, value):
        if isinstance(value, dict):
            return value
        self.error(obj, value)


class PairedEnum(Enum):
    """A enum whose value must be in a given sequence.

    This enum controls a slaved option, with default values provided here.

    :arg values: iterable of (value, HasTraits) pairs
    :arg paired_name: trait name this enum is paired with.
    :arg default_value: default value.
    """
    def __init__(self, values, paired_name, default_value=Undefined, **kwargs):
        self.paired_defaults = dict(values)
        self.paired_name = paired_name
        values, _ = zip(*values)
        super(PairedEnum, self).__init__(values, default_value, **kwargs)

    def info(self):
        result = "This option also requires configuration of %s\n" % (self.paired_name)
        return result + super(PairedEnum, self).info()


class FrozenHasTraits(HasTraits):
    """
    A HasTraits class that only allows adding new attributes in the class
    definition or when  self._isfrozen is False.
    """
    _isfrozen = False

    def __init__(self, *args, **kwargs):
        super(FrozenHasTraits, self).__init__(*args, **kwargs)
        self._isfrozen = True

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenHasTraits, self).__setattr__(key, value)


class FrozenConfigurable(Configurable):
    """
    A Configurable class that only allows adding new attributes in the class
    definition or when  self._isfrozen is False.
    """
    _isfrozen = False

    def __init__(self, *args, **kwargs):
        super(FrozenConfigurable, self).__init__(*args, **kwargs)
        self._isfrozen = True

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenConfigurable, self).__setattr__(key, value)

    def update(self, source):
        if isinstance(source, dict):
            for key in source:
                self.__setattr__(key, source[key])
        else:
            self.add_traits(**source.traits())


class TimeStepperOptions(FrozenHasTraits):
    """Base class for all time stepper options"""
    pass


class ExplicitTimestepperOptions(TimeStepperOptions):
    """Options for explicit time integrator"""
    use_automatic_timestep = Bool(True, help='Set time step automatically based on local CFL conditions.').tag(config=True)


class SemiImplicitTimestepperOptions2d(TimeStepperOptions):
    """Options for 2d explicit time integrator"""
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
    }).tag(config=True)


class SteadyStateTimestepperOptions2d(TimeStepperOptions):
    """Options for 2d steady state solver"""
    solver_parameters = PETScSolverParameters({
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'mat_type': 'aij'
    }).tag(config=True)


class CrankNicolsonTimestepperOptions2d(SemiImplicitTimestepperOptions2d):
    """Options for 2d Crank-Nicolson time integrator"""
    implicitness_theta = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta. Value 0.5 implies Crank-Nicolson scheme, 1.0 implies fully implicit formulation.').tag(config=True)


class PressureProjectionTimestepperOptions2d(TimeStepperOptions):
    """Options for 2d pressure-projection time integrator"""
    solver_parameters_pressure = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
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


class ExplicitTimestepperOptions2d(ExplicitTimestepperOptions):
    """Options for 2d explicit time integrator"""
    solver_parameters = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
        'mat_type': 'aij',
    }).tag(config=True)


class ExplicitTimestepperOptions3d(ExplicitTimestepperOptions):
    """Base class for all 3d time stepper options"""
    solver_parameters_2d_swe = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
    }).tag(config=True)
    solver_parameters_momentum_explicit = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)
    solver_parameters_momentum_implicit = PETScSolverParameters({
        'snes_monitor': False,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)
    solver_parameters_tracer_explicit = PETScSolverParameters({
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)
    solver_parameters_tracer_implicit = PETScSolverParameters({
        'snes_monitor': False,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'ilu',
    }).tag(config=True)


class SemiImplicitTimestepperOptions3d(ExplicitTimestepperOptions3d):
    """Class for all 3d time steppers that have a configurable semi-implicit 2D solver"""
    implicitness_theta_2d = BoundedFloat(
        default_value=0.5, bounds=[0.5, 1.0],
        help='implicitness parameter theta for 2D solver. Value 1.0 implies fully implicit formulation.').tag(config=True)


class GLSModelOptions(FrozenHasTraits):
    """Options for generic length scale turbulence model"""
    closure_name = Enum(
        ['k-epsilon', 'k-omega', 'Generic Lenght Scale'],
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
    compute_kappa = Bool(True,
                         help='bool: compute von Karman constant from :attr:`schmidt_nb_psi`').tag(config=True)
    k_min = PositiveFloat(3.7e-8,
                          help='float: minimum value for turbulent kinetic energy').tag(config=True)
    psi_min = PositiveFloat(1.0e-10, help='float: minimum value for psi').tag(config=True)
    eps_min = PositiveFloat(1.0e-10, help='float: minimum value for epsilon').tag(config=True)
    len_min = PositiveFloat(1.0e-10,
                            help='float: minimum value for turbulent lenght scale').tag(config=True)
    compute_len_min = Bool(True,
                           help='bool: compute min_len from k_min and psi_min').tag(config=True)
    compute_psi_min = Bool(True,
                           help='bool: compute psi_len from k_min and eps_min').tag(config=True)
    visc_min = PositiveFloat(1.0e-8,
                             help='float: minimum value for eddy viscosity').tag(config=True)
    diff_min = PositiveFloat(1.0e-8,
                             help='float: minimum value for eddy diffusivity').tag(config=True)
    galperin_lim = PositiveFloat(0.56,
                                 help='float: Galperin lenght scale limitation parameter').tag(config=True)

    limit_len = Bool(False, help='bool: apply Galperin lenght scale limit').tag(config=True)
    limit_psi = Bool(False,
                     help='bool: apply Galperin lenght scale limit on psi').tag(config=True)
    limit_eps = Bool(False,
                     help='bool: apply Galperin lenght scale limit on epsilon').tag(config=True)
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
                    'k_min': 3.7e-8,
                    'psi_min': 1.0e-10,
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
                  'k_min': 3.7e-8,
                  'eps_min': 1.0e-10,
                  'psi_min': 1.0e-10,
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
               'k_min': 3.7e-8,
               'eps_min': 1.0e-10,
               'psi_min': 2.0e-7,
               'closure_name': 'gen',
               }
        # GLS model A defaults, from tables 1 and 2 in [3]

        if closure_name == 'k-epsilon':
            self.update(kepsilon)
        elif closure_name == 'k-omega':
            self.update(komega)
        elif closure_name == 'gen':
            self.update(gen)

    def update(self, params_dict):
            for key in params_dict:
                self.__setattr__(key, params_dict[key])

    def print_summary(self):
        """Prints all defined parameters and their values."""
        print_output('GLS Turbulence model parameters')
        params_dict = self._trait_values
        for k in sorted(params_dict.keys()):
            print_output('  {:16s} : {:}'.format(k, params_dict[k]))


class EquationOfStateOptions(FrozenHasTraits):
    """Base class of equation of state options"""
    pass


class LinearEquationOfStateOptions(EquationOfStateOptions):
    """Linear equation of state options"""
    # TODO more human readable parameter names
    # TODO document the actual equation somewhere
    rho_ref = NonNegativeFloat(1000.0, help='Reference water density').tag(config=True)
    s_ref = NonNegativeFloat(35.0, help='Reference water salinity').tag(config=True)
    th_ref = Float(15.0, help='Reference water temperature').tag(config=True)
    alpha = Float(0.2, help='Thermal expansion coefficient of ocean water').tag(config=True)
    beta = Float(0.77, help='Saline contraction coefficient of ocean water').tag(config=True)


def attach_paired_options(name, name_trait, value_trait):
    """Attach paired options to a Configurable object.

    :arg name: the name of the enum trait
    :arg name_trait: the enum trait (a PairedEnum)
    :arg value_trait: the slaved value trait."""

    def _observer(self, change):
        "Observer called when the choice option is updated."
        setattr(self, name_trait.paired_name, name_trait.paired_defaults[change["new"]])

    def _default(self):
        "Dynamic default value setter"
        if hasattr(name_trait, 'default_value') and name_trait.default_value is not None:
            return name_trait.paired_defaults[name_trait.default_value]

    obs_handler = ObserveHandler(name, type="change")
    def_handler = DefaultHandler(name_trait.paired_name)

    def update_class(cls):
        "Programmatically update the class"
        # Set the new class attributes
        setattr(cls, name, name_trait)
        setattr(cls, name_trait.paired_name, value_trait)
        setattr(cls, "_%s_observer" % name, obs_handler(_observer))
        setattr(cls, "_%s_default" % name_trait.paired_name, def_handler(_default))
        # Mimic the magic metaclass voodoo.
        name_trait.class_init(cls, name)
        value_trait.class_init(cls, name_trait.paired_name)
        obs_handler.class_init(cls, "_%s_observer" % name)
        def_handler.class_init(cls, "_%s_default" % name_trait.paired_name)

        return cls
    return update_class


class CommonModelOptions(FrozenConfigurable):
    """Options that are common for both 2d and 3d models"""
    polynomial_degree = NonNegativeInteger(1, help='Polynomial degree of elements').tag(config=True)
    element_family = Enum(
        ['dg-dg', 'rt-dg', 'dg-cg'],
        default_value='dg-dg',
        help="""Finite element family

        2D solver supports 'dg-dg', 'rt-dg', or 'dg-cg' velocity-pressure pairs.
        3D solver supports 'dg-dg', or 'rt-dg' velocity-pressure pairs.""").tag(config=True)

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
    lax_friedrichs_velocity_scaling_factor = FiredrakeConstant(
        Constant(1.0), help="Scaling factor for Lax Friedrichs stabilisation term in horiozonal momentum advection.").tag(config=True)
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
    horizontal_velocity_scale = FiredrakeConstant(
        Constant(0.1), help="""
        Maximum horizontal velocity magnitude

        Used to compute max stable advection time step.
        """).tag(config=True)
    horizontal_viscosity_scale = FiredrakeConstant(
        Constant(1.0), help="""
        Maximum horizontal viscosity

        Used to compute max stable diffusion time step.
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
        trait=Unicode,
        default_value=['elev_2d', 'uv_2d', 'uv_3d', 'w_3d'],
        help="Fields to export in VTK format").tag(config=True)
    fields_to_export_hdf5 = List(
        trait=Unicode,
        default_value=[],
        help="Fields to export in HDF5 format").tag(config=True)
    verbose = Integer(0, help="Verbosity level").tag(config=True)
    linear_drag_coefficient = FiredrakeCoefficient(
        None, allow_none=True, help=r"""
        2D linear drag parameter :math:`L`

        Bottom stress is :math:`\tau_b/\rho_0 = -L \mathbf{u} H`
        """).tag(config=True)
    quadratic_drag_coefficient = FiredrakeCoefficient(
        None, allow_none=True, help=r"""
        Dimensionless 2D quadratic drag parameter :math:`C_D`

        Bottom stress is :math:`\tau_b/\rho_0 = -C_D |\mathbf{u}|\mathbf{u}`
        """).tag(config=True)
    manning_drag_coefficient = FiredrakeCoefficient(
        None, allow_none=True, help=r"""
        Manning-Strickler 2D quadratic drag parameter :math:`\mu`

        Bottom stress is :math:`\tau_b/\rho_0 = -g \mu^2 |\mathbf{u}|\mathbf{u}/H^{1/3}`
        """).tag(config=True)
    horizontal_viscosity = FiredrakeCoefficient(
        None, allow_none=True, help="Horizontal viscosity").tag(config=True)
    coriolis_frequency = FiredrakeCoefficient(
        None, allow_none=True, help="2D Coriolis parameter").tag(config=True)
    wind_stress = FiredrakeCoefficient(
        None, allow_none=True, help="Stress at free surface (2D vector function)").tag(config=True)
    atmospheric_pressure = FiredrakeCoefficient(
        None, allow_none=True, help="Atmospheric pressure at free surface, in pascals").tag(config=True)
    momentum_source_2d = FiredrakeCoefficient(
        None, allow_none=True, help="Source term for 2D momentum equation").tag(config=True)
    volume_source_2d = FiredrakeCoefficient(
        None, allow_none=True, help="Source term for 2D continuity equation").tag(config=True)


# NOTE all parameters are now case sensitive
# TODO rename time stepper types? Allow capitals and spaces?
@attach_paired_options("timestepper_type",
                       PairedEnum([('SSPRK33', ExplicitTimestepperOptions2d()),
                                   ('SSPRK33Semi', CrankNicolsonTimestepperOptions2d()),
                                   ('ForwardEuler', ExplicitTimestepperOptions2d()),
                                   ('BackwardEuler', SemiImplicitTimestepperOptions2d()),
                                   ('CrankNicolson', CrankNicolsonTimestepperOptions2d()),
                                   ('DIRK22', SemiImplicitTimestepperOptions2d()),
                                   ('DIRK33', SemiImplicitTimestepperOptions2d()),
                                   ('SteadyState', SteadyStateTimestepperOptions2d()),
                                   ('PressureProjectionPicard', PressureProjectionTimestepperOptions2d()),
                                   ('SSPIMEX', SemiImplicitTimestepperOptions2d()),
                                   ],
                                  "timestepper_options",
                                  default_value='CrankNicolson',
                                  help='Name of the time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
class ModelOptions2d(CommonModelOptions):
    """Options for 2D depth-averaged shallow water model"""
    use_linearized_semi_implicit_2d = Bool(
        False, help="Use linearized semi-implicit time integration for the horizontal mode").tag(config=True)
    use_wetting_and_drying = Bool(
        False, help=r"""bool: Turn on wetting and drying

        Uses the wetting and drying scheme from Karna et al (2011).
        If ``True``, one should also set :attr:`wetting_and_drying_alpha` to control the bathymetry displacement.
        """).tag(config=True)
    wetting_and_drying_alpha = FiredrakeConstant(
        Constant(0.5), help=r"""
        Coefficient: Wetting and drying parameter :math:`alpha`.

        Used in bathymetry displacement function that ensures positive water depths. Unit is meters.
        """).tag(config=True)


@attach_paired_options("timestepper_type",
                       PairedEnum([('SSPRK33', SemiImplicitTimestepperOptions3d()),
                                   ('LeapFrog', ExplicitTimestepperOptions3d()),
                                   ('SSPRK22', ExplicitTimestepperOptions3d()),
                                   ('IMEXALE', ExplicitTimestepperOptions3d()),
                                   ('ERKALE', ExplicitTimestepperOptions3d()),
                                   ],
                                  "timestepper_options",
                                  default_value='SSPRK22',
                                  help='Name of the time integrator').tag(config=True),
                       Instance(TimeStepperOptions, args=()).tag(config=True))
@attach_paired_options("turbulence_model_type",
                       PairedEnum([('gls', GLSModelOptions())],
                                  "gls_options",
                                  default_value='gls',
                                  help='Type of vertical turbulence model').tag(config=True),
                       Instance(GLSModelOptions, args=()).tag(config=True))
@attach_paired_options("equation_of_state_type",
                       PairedEnum([('full', EquationOfStateOptions()),
                                   ('linear', LinearEquationOfStateOptions())],
                                  "equation_of_state_options",
                                  default_value='full',
                                  help='Type of equation of state').tag(config=True),
                       Instance(EquationOfStateOptions, args=()).tag(config=True))
class ModelOptions3d(CommonModelOptions):
    """Options for 3D hydrostatic model"""
    solve_salinity = Bool(True, help='Solve salinity transport').tag(config=True)
    solve_temperature = Bool(True, help='Solve temperature transport').tag(config=True)
    use_implicit_vertical_diffusion = Bool(True, help='Solve vertical diffusion and viscosity implicitly').tag(config=True)
    use_bottom_friction = Bool(True, help='Apply log layer bottom stress in the 3D model').tag(config=True)
    use_parabolic_viscosity = Bool(
        False,
        help="""Use idealized parabolic eddy viscosity

        See :class:`.ParabolicViscosity`""").tag(config=True)
    use_ale_moving_mesh = Bool(
        True, help="Use ALE formulation where 3D mesh tracks free surface").tag(config=True)
    use_baroclinic_formulation = Bool(
        False, help="Compute internal pressure gradient in momentum equation").tag(config=True)
    use_turbulence = Bool(
        False, help="Activate turbulence model in the 3D model").tag(config=True)

    use_turbulence_advection = Bool(
        False, help="Advect TKE and Psi in the GLS turbulence model").tag(config=True)
    use_smooth_eddy_viscosity = Bool(
        False, help="Cast eddy viscosity to p1 space instead of p0").tag(config=True)
    use_smagorinsky_viscosity = Bool(
        False, help="Use Smagorinsky horisontal viscosity parametrization").tag(config=True)
    smagorinsky_coefficient = FiredrakeConstant(
        Constant(0.1),
        help="""Smagorinsky viscosity coefficient :math:`C_S`

        See :class:`.SmagorinskyViscosity`.""").tag(config=True)

    use_limiter_for_tracers = Bool(
        False, help="Apply P1DG limiter for tracer fields").tag(config=True)
    use_lax_friedrichs_tracer = Bool(
        True, help="Use Lax Friedrichs stabilisation in tracer advection.").tag(config=True)
    lax_friedrichs_tracer_scaling_factor = FiredrakeConstant(
        Constant(1.0), help="Scaling factor for tracer Lax Friedrichs stability term.").tag(config=True)
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
    vertical_velocity_scale = FiredrakeConstant(
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
    horizontal_diffusivity = FiredrakeCoefficient(
        None, allow_none=True, help="Horizontal diffusivity for tracers").tag(config=True)
    vertical_diffusivity = FiredrakeCoefficient(
        None, allow_none=True, help="Vertical diffusivity for tracers").tag(config=True)
    vertical_viscosity = FiredrakeCoefficient(
        None, allow_none=True, help="Vertical viscosity").tag(config=True)
    momentum_source_3d = FiredrakeCoefficient(
        None, allow_none=True, help="Source term for 3D momentum equation").tag(config=True)
    salinity_source_3d = FiredrakeCoefficient(
        None, allow_none=True, help="Source term for salinity equation").tag(config=True)
    temperature_source_3d = FiredrakeCoefficient(
        None, allow_none=True, help="Source term for temperature equation").tag(config=True)
    constant_temperature = FiredrakeConstant(
        Constant(10.0), help="Constant temperature if temperature is not solved").tag(config=True)
    constant_salinity = FiredrakeConstant(
        Constant(0.0), help="Constant salinity if salinity is not solved").tag(config=True)
