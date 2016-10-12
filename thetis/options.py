"""
This file defines all options of the 2D/3D models excluding field values.

Tuomas Karna 2015-10-17
"""
from __future__ import absolute_import
from .utility import *
from .turbulence import GLSModelOptions
from firedrake.parameter_types import KeyType, IntType, FloatType, StrType


class NoneKeyType(KeyType):
    """KeyType that only allows None as value"""
    def validate(self, value):
        return value is None

    def parse(self, value):
        return "None"


class CombinedType(KeyType):
    """KeyType that allows the value to be of any of the specified types"""
    def __init__(self, *types):
        self.types = types

    def validate(self, value):
        return any(t.validate(value) for t in self.types)

    def parse(self, value):
        for t in self.types:
            if t.validate(value):
                return t.parse(value)
        return None


class IterableType(KeyType):
    """KeyType where the value should be an iterable of one specified type"""
    def __init__(self, item_type):
        self.item_type = item_type

    def validate(self, values):
        try:
            return all(self.item_type.validate(v) for v in values)
        except TypeError:
            return False

    def parse(self, values):
        if self.validate(values):
            return (self.item_type.parse(v) for v in values)
        else:
            return None


def is_ufl(expr):
    from ufl.core.expr import Expr
    return isinstance(expr, Expr)


class UFLScalarType(KeyType):
    """KeyType for any scalar ufl expression"""
    def validate(self, value):
        return is_ufl(value) and not value.ufl_shape

    def parse(self, value):
        if self.validate(value):
            return repr(value)
        else:
            return None


class UFLVectorType(KeyType):
    """KeyType for any vector ufl expression"""
    def validate(self, value):
        return is_ufl(value) and len(value.ufl_shape) == 1

    def parse(self, value):
        if self.validate(value):
            return repr(value)
        else:
            return None


class ModelOptions(Parameters):
    """
    Stores all circulation model options
    """
    _allow_new_attributes = True
    _isfrozen = False

    def __init__(self):
        """
        Initialize with default options
        """
        super(ModelOptions, self).__init__()
        self._allow_new_attributes = False

        self.add_option("order", 1, "Polynomial degree of elements",
                        val_type=IntType(lower_bound=0))
        self.add_option("element_family", 'rt-dg',
                        "Finite element family. Currently 'dg-dg', 'rt-dg', or 'dg-cg' velocity-pressure pairs are supported.",
                        val_type=StrType("rt-dg", "dg-dg", "dg-cg"))
        # TODO: make this a choice list - and see what happens in 3D
        self.add_option("timestepper_type", "ssprk33", "Timestepper for 2D mode""")
        self.add_option("nonlin", True,
                        "bool: Use nonlinear shallow water equations")
        self.add_option("solve_salt", True, "Solve salt transport")
        self.add_option("solve_temp", True, "Solve temperature transport")
        self.add_option("solve_vert_diffusion", True, "Solve implicit vert diffusion")
        self.add_option("use_bottom_friction", True, "Apply log layer bottom stress")
        self.add_option("use_parabolic_viscosity", False, "Compute parabolic eddy viscosity")
        self.add_option("include_grad_div_viscosity_term", False, "Include grad(nu div(u)) term in the depth-averaged viscosity")
        self.add_option("include_grad_depth_viscosity_term", True, "Include grad(H) term in the depth-averaged viscosity")
        self.add_option("use_ale_moving_mesh", True, "3D mesh tracks free surface")
        self.add_option("use_mode_split", True, "Solve 2D/3D modes with different dt")
        self.add_option("use_semi_implicit_2d", True, "Implicit 2D waves (only w. mode split)")
        self.add_option("use_imex", False, "Use IMEX time integrator (only with mode split)")
        self.add_option("use_linearized_semi_implicit_2d", False, "Use linearized semi-implicit time integration for the horizontal mode")
        self.add_option("shallow_water_theta", 0.5, "theta parameter for shallow water semi-implicit scheme")
        self.add_option("use_turbulence", False, "GLS turbulence model")
        self.add_option("use_smooth_eddy_viscosity", False, "Cast eddy viscosity to p1 space instead of p0")
        self.add_option("turbulence_model", 'gls', "Defines the type of vertical turbulence model. Currently only 'gls'")
        self.add_option("gls_options", GLSModelOptions(), "Dictionary of default GLS model options")
        self.add_option("use_turbulence_advection", False, "Advect tke,psi with velocity")
        self.add_option("baroclinic", False, "Compute internal pressure gradient in momentum equation")
        none_type = NoneKeyType()
        float_type = FloatType()
        none_or_float = CombinedType(none_type, float_type)
        self.add_option("smagorinsky_factor", None, "Smagorinsky viscosity factor C_S",
                        val_type=none_or_float)
        self.add_option("salt_jump_diff_factor", None, "Non-linear jump diffusion factor",
                        val_type=none_or_float)
        self.add_option("salt_range", 30.0, "Salt max-min range for jump diffusion")
        self.add_option("use_limiter_for_tracers", False, "Apply P1DG limiter for tracer fields")
        self.add_option("uv_lax_friedrichs", 1.0, "Scaling factor for uv L-F stability term.",
                        val_type=none_or_float)
        self.add_option("tracer_lax_friedrichs", 1.0, "Scaling factor for tracer L-F stability term.",
                        val_type=none_or_float)
        self.add_option("check_vol_conservation_2d", False, "Print deviation from initial volume for 2D mode (eta)")
        self.add_option("check_vol_conservation_3d", False, "Print deviation from initial volume for 3D mode (domain volume)")
        self.add_option("check_salt_conservation", False, "Print deviation from initial salt mass")
        self.add_option("check_salt_overshoot", False, "Print overshoots that exceed initial range")
        self.add_option("check_temp_conservation", False, "Print deviation from initial temp mass")
        self.add_option("check_temp_overshoot", False, "Print overshoots that exceed initial range")
        self.add_option("dt", None, "Time step. If set overrides automatically computed stable dt",
                        val_type=none_or_float)
        self.add_option("dt_2d", None, "Time step for 2d mode. If set overrides automatically computed stable dt",
                        val_type=none_or_float)
        self.add_option("cfl_2d", 1.0, "Factor to scale the 2d time step")
        self.add_option("cfl_3d", 1.0, "Factor to scale the 2d time step")
        self.add_option("t_export", 100.0, "Export interval in seconds. All fields in fields_to_export list will be stored to disk and diagnostics will be computed.")
        self.add_option("t_end", 1000.0, "Simulation duration in seconds")
        self.add_option("u_advection", 0.0, "Max. horizontal velocity magnitude for computing max stable advection time step.")
        self.add_option("outputdir", 'outputs', "Directory where model output files are stored")
        self.add_option("no_exports", False, "Do not store any outputs to disk, used in CI test suite. Disables vtk and hdf5 field outputs and hdf5 diagnostic outputs.")
        self.add_option("export_diagnostics", True, "Store diagnostic variables to disk in hdf5 format")
        eos_type = StrType('full', 'linear')
        self.add_option("equation_of_state", 'full', "type of equation of state, either 'full' or 'linear'",
                        val_type=eos_type)
        self.add_option("lin_equation_of_state_params", {
            'rho_ref': 1000.0,
            's_ref': 35.0,
            'th_ref': 15.0,
            'alpha': 0.2,
            'beta': 0.77,
        }, "definition of linear equation of state")
        self.add_option("fields_to_export", ['elev_2d', 'uv_2d', 'uv_3d', 'w_3d'], "of str: Fields to export in VTK format")
        self.add_option("fields_to_export_numpy", [], "of str: Fields to export in HDF5 format")
        self.add_option("fields_to_export_hdf5", [], "of str: Fields to export in numpy format")
        self.add_option("verbose", 0, "Verbosity level")
        # NOTE these are fields, potentially Functions: move out of this class?
        ufl_scalar_type = UFLScalarType()
        ufl_vector_type = UFLVectorType()
        none_or_scalar = CombinedType(none_type, ufl_scalar_type, float_type)
        none_or_vector = CombinedType(none_type, ufl_vector_type, IterableType(float_type))
        self.add_option("linear_drag", None, "2D linear drag parameter tau/rho_0 = -drag*u*H",
                        val_type=none_or_scalar)
        self.add_option("quadratic_drag", None, "dimensionless 2D quadratic drag parameter tau/rho_0 = -drag*|u|*u",
                        val_type=none_or_scalar)
        self.add_option("mu_manning", None, "Manning-Strickler 2D quadratic drag parameter tau/rho_0 = -g*mu**2*|u|*u/H^(1/3)",
                        val_type=none_or_scalar)
        self.add_option("h_diffusivity", None, "Background diffusivity",
                        val_type=none_or_scalar)
        self.add_option("v_diffusivity", None, "background diffusivity",
                        val_type=none_or_scalar)
        self.add_option("h_viscosity", None, "background viscosity",
                        val_type=none_or_scalar)
        self.add_option("v_viscosity", None, "background viscosity",
                        val_type=none_or_scalar)
        self.add_option("coriolis", None, "Coefficient or None: Coriolis parameter",
                        val_type=none_or_scalar)
        self.add_option("wind_stress", None, "Stress at free surface (2D vector function)",
                        val_type=none_or_vector)
        self.add_option("uv_source_2d", None, "source term for 2d momentum equation",
                        val_type=none_or_vector)
        self.add_option("uv_source_3d", None, "source term for 3d momentum equation",
                        val_type=none_or_vector)
        self.add_option("elev_source_2d", None, "source term for 2d continuity equation",
                        val_type=none_or_scalar)
        self.add_option("salt_source_3d", None, "source term for salinity equation",
                        val_type=none_or_scalar)
        self.add_option("temp_source_3d", None, "source term for temperature equation",
                        val_type=none_or_scalar)
        self.add_option("constant_temp", 10.0, "Constant temperature if temperature is not solved",
                        val_type=none_or_scalar)
        self.add_option("constant_salt", 0.0, "Constant salinity if salinity is not solved",
                        val_type=none_or_scalar)
        self.add_option("solver_parameters_sw", {
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
        },
            "PETSc solver parameters for 2D shallow water equations")
        self.add_option("solver_parameters_sw_momentum", {
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'sor',
        },
            "PETSc solver parameters for 2D depth averaged momentum equation")
        self.add_option("solver_parameters_momentum_explicit", {
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        },
            "PETSc solver parameters for explicit 3D momentum equation")
        self.add_option("solver_parameters_momentum_implicit", {
            'snes_monitor': False,
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        },
            "PETSc solver parameters for implicit 3D momentum equation")
        self.add_option("solver_parameters_tracer_explicit", {
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        },
            "PETSc solver parameters for explicit 3D tracer equations")
        self.add_option("solver_parameters_tracer_implicit", {
            'snes_monitor': False,
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_ksp_type': 'preonly',
            'sub_pc_type': 'ilu',
        },
            "PETSc solver parameters for implicit 3D tracer equations")

        self._isfrozen = True

    def __setattr__(self, key, value):
        if self._allow_new_attributes or (hasattr(self, key) and key not in self.keys()):
            super(ModelOptions, self).__setattr__(key, value)
        else:
            if self._isfrozen and key not in self.keys():
                raise TypeError('Adding new option "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
            self[key] = value

    def __getattr__(self, key):
        """NOTE: __getattr__ is only called if key doesn not exist as a real class or object attribute already"""
        # NOTE: this may raise a KeyError instead of a AttributeError
        return self[key]

    def __setitem__(self, key, value):
        if self._isfrozen and key not in self.keys():
            raise TypeError('Adding new option "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(ModelOptions, self).__setitem__(key, value)

    def add_option(self, key, default_value, help, val_type=None, **kwargs):

        if val_type is None:
            val_type = KeyType.get_type(default_value)

        typed_key = TypedKey(key, val_type, help=help, **kwargs)
        self[typed_key] = default_value
