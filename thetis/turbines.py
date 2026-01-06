"""
Classes and utilities related to tidal turbine farms in Thetis.
"""
from firedrake import *
from firedrake.petsc import PETSc
from .log import *
from .callback import DiagnosticCallback
from .physical_constants import physical_constants
from .optimisation import DiagnosticOptimisationCallback
from .options import TidalTurbineFarmOptions, DiscreteTidalTurbineFarmOptions
import pyadjoint
import numpy
import json
import yaml


class TidalTurbine:
    def __init__(self, options, upwind_correction=False, rotor_weightings=None):
        self.diameter = options.diameter
        self.projected_diameter = options.projected_diameter or self.diameter
        self.C_support = options.C_support
        self.A_support = options.A_support
        self.upwind_correction = upwind_correction
        self.apply_shear_profile = options.apply_shear_profile
        self.shear_alpha = options.shear_alpha
        self.shear_beta = options.shear_beta
        self.rel_hub_height = options.rel_hub_height
        self.structure_type = options.structure_type
        self.rotor_weightings = (
            numpy.array(rotor_weightings)
            if rotor_weightings is not None else
            numpy.array([0.052, 0.0903, 0.1099, 0.1212, 0.1266,
                         0.1266, 0.1212, 0.1099, 0.0903, 0.052])
        )

    def _thrust_area(self, uv):
        C_T = self.thrust_coefficient(uv)
        A_T = pi * self.diameter**2 / 4
        fric = C_T * A_T
        if self.C_support:
            fric += self.C_support * self.A_support
        return fric

    def velocity_correction(self, uv, depth):
        fric = self._thrust_area(uv)
        if self.upwind_correction:
            return 0.5*(1+sqrt(1-fric/(self.projected_diameter*depth)))
        else:
            return 1

    def friction_coefficient(self, uv, depth):
        if self.apply_shear_profile:
            uv_eff = self.rotor_averaged_velocity(uv, depth)
        else:
            uv_eff = uv
        thrust_area = self._thrust_area(uv_eff)
        alpha = self.velocity_correction(uv_eff, depth)
        return thrust_area/2./alpha**2

    def rotor_averaged_velocity(self, uv, depth):
        if not self.apply_shear_profile:
            return uv

        if self.rel_hub_height is None:
            raise ValueError("`rel_hub_height` must be specified when applying a shear profile.")

        # Determine hub elevation depending on structure type
        if self.structure_type == "bottom-fixed":
            hub = self.rel_hub_height  # height above seabed
        elif self.structure_type == "floating":
            hub = depth - self.rel_hub_height  # depth below free surface
        else:
            raise ValueError(f"Unknown turbine structure type '{self.structure_type}'")

        # Vertical sampling
        N = len(self.rotor_weightings)  # sample the rotor at N points vertically, hardcoded weightings
        z_vals = numpy.linspace(hub - self.diameter / 2, hub + self.diameter / 2, N)

        # power-law shear profile
        u_samples = dot(uv, uv)**0.5 * (z_vals / (self.shear_beta * depth)) ** (1.0 / self.shear_alpha)
        u_cubed = u_samples ** 3
        rotor_avg = (sum(u_cubed * self.rotor_weightings)) ** (1 / 3)
        return rotor_avg

    def power(self, uv, depth):
        # ratio of discrete to upstream velocity (NOTE: should include support drag!)
        alpha = self.velocity_correction(uv, depth)
        A_T = pi * self.diameter**2 / 4  # power is based on true turbine diameter
        uv_eff = self.rotor_averaged_velocity(uv, depth)
        uv3 = dot(uv_eff, uv_eff)**1.5 / alpha**3  # upwind cubed velocity
        C_P = self.power_coefficient(uv3**(1/3))
        # this assumes the velocity through the turbine does not change due to the support (is this correct?)
        return 0.5*physical_constants['rho0']*A_T*C_P*uv3  # units: W


class ConstantThrustTurbine(TidalTurbine):
    def __init__(self, options, upwind_correction=False):
        super().__init__(options, upwind_correction=upwind_correction)
        self.C_T = options.thrust_coefficient
        self.C_P = options.power_coefficient or 0.5 * self.C_T * (1 + (1 - self.C_T) ** 0.5)

    def thrust_coefficient(self, uv):
        return self.C_T

    def power_coefficient(self, uv):
        return self.C_P


def linearly_interpolate_table(x_list, y_list, y_final, x):
    """Return UFL expression that linearly interpolates between y-values in x-points

    :param x_list: (1D) x-points
    :param y_list: y-values in those points
    :param y_final: value for x>x_list[-1]
    :param x: point to interpolate (assumed x>x_list[0])
    """
    # below x1, interpolate between x0 and x1:
    below_x1 = ((x_list[1]-x)*y_list[0] + (x-x_list[0])*y_list[1])/(x_list[1]-x_list[0])
    # above x1, interpolate from rest of the table, or take final value:
    if len(x_list) > 2:
        above_x1 = linearly_interpolate_table(x_list[1:], y_list[1:], y_final, x)
    else:
        above_x1 = y_final

    return conditional(x < x_list[1], below_x1, above_x1)


class TabulatedThrustTurbine(TidalTurbine):
    def __init__(self, options, upwind_correction=False):
        super().__init__(options, upwind_correction=upwind_correction)
        self.C_T = options.thrust_coefficients
        self.C_P = options.power_coefficients or [0.5 * c_t * (1 + (1 - c_t) ** 0.5) for c_t in self.C_T]
        self.speeds = options.thrust_speeds
        if not len(self.C_T) == len(self.speeds):
            raise ValueError("In tabulated thrust curve the number of thrust coefficients and speed values should be the same.")
        if not len(self.C_P) == len(self.speeds):
            raise ValueError("In tabulated thrust curve the number of power coefficients and speed values should be the same.")

    def thrust_coefficient(self, uv):
        umag = dot(uv, uv)**0.5
        return conditional(umag < self.speeds[0], 0, linearly_interpolate_table(self.speeds, self.C_T, 0, umag))

    def power_coefficient(self, uv):
        umag = dot(uv, uv) ** 0.5
        return conditional(umag < self.speeds[0], 0, linearly_interpolate_table(self.speeds, self.C_P, 0, umag))


class TidalTurbineFarm:
    def __init__(self, turbine_density, dx, options):
        """
        :arg turbine_density: turbine distribution density field or expression
        :arg dx: measure to integrate power output, n/o turbines
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """
        upwind_correction = getattr(options, 'upwind_correction', False)
        if options.turbine_type == 'constant':
            self.turbine = ConstantThrustTurbine(options.turbine_options, upwind_correction=upwind_correction)
        elif options.turbine_type == 'table':
            self.turbine = TabulatedThrustTurbine(options.turbine_options, upwind_correction=upwind_correction)
        self.dx = dx
        self.turbine_density = turbine_density
        self.break_even_wattage = options.break_even_wattage

    def number_of_turbines(self):
        return assemble(self.turbine_density * self.dx)

    def power_output(self, uv, depth):
        return assemble(self.turbine.power(uv, depth) * self.turbine_density * self.dx)

    def friction_coefficient(self, uv, depth):
        return self.turbine.friction_coefficient(uv, depth)


class DiscreteTidalTurbineFarm(TidalTurbineFarm):
    """
    Class that can be used for the addition of turbines in the turbine density field
    """

    def __init__(self, mesh, dx, options):
        """
        :arg mesh: mesh domain
        :arg dx: measure to integrate power output, n/o turbines
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """

        # Preliminaries
        self.mesh = mesh
        # this sets self.turbine_expr=0
        super().__init__(0, dx, options)

        # Adding turbine distribution in the domain
        self.add_turbines(options.turbine_coordinates)

    def add_turbines(self, coordinates):
        """
        :param coordinates: Array with turbine coordinates to be positioned
        :return: updated turbine density field (in place)
        """
        x = SpatialCoordinate(self.mesh)

        radius = self.turbine.projected_diameter * 0.5
        for coord in coordinates:
            dx0 = (x[0] - coord[0])/radius
            dx1 = (x[1] - coord[1])/radius
            psi_x = conditional(lt(abs(dx0), 1), exp(1-1/(1-dx0**2)), 0)
            psi_y = conditional(lt(abs(dx1), 1), exp(1-1/(1-dx1**2)), 0)
            bump = psi_x * psi_y

            unit_bump_integral = 1.45661  # integral of bump function for radius=1 (copied from OpenTidalFarm who used Wolfram)
            self.turbine_density = self.turbine_density + bump/(radius**2 * unit_bump_integral)


class TurbineFunctionalCallback(DiagnosticCallback):
    """
    :class:`.DiagnosticCallback` that evaluates the performance of each tidal turbine farm."""

    name = 'turbine'  # this name will be used in the hdf5 file
    variable_names = ['current_power', 'average_power', 'average_profit']

    @PETSc.Log.EventDecorator("thetis.TurbineFunctionalCallback.__init__")
    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: a :class:`.FlowSolver2d` object containing the tidal_turbine_farms
        :arg kwargs: see :class:`DiagnosticCallback`"""
        if not hasattr(solver_obj, 'tidal_farms'):
            solver_obj.create_equations()
        self.farms = solver_obj.tidal_farms
        nfarms = len(self.farms)
        super().__init__(solver_obj, array_dim=nfarms, **kwargs)

        self.uv, eta = split(solver_obj.fields.solution_2d)
        self.dt = solver_obj.options.timestep
        self.depth = solver_obj.fields.bathymetry_2d

        self.cost = [farm.number_of_turbines() for farm in self.farms]
        if self.append_to_log:
            print_output('Number of turbines = {}'.format(sum(self.cost)))
        self.break_even_wattage = [farm.break_even_wattage for farm in self.farms]

        self.instantaneous_power = [0] * nfarms
        # time-integrated quantities:
        self.integrated_power = [0] * nfarms
        self.average_power = [0] * nfarms
        self.average_profit = [0] * nfarms
        self.time_period = 0.

    def _evaluate_timestep(self):
        """Perform time integration and return current power and time-averaged power and profit."""
        self.time_period = self.time_period + self.dt
        current_power = []
        for i, farm in enumerate(self.farms):
            power = farm.power_output(self.uv, self.depth)
            current_power.append(power)
            self.instantaneous_power[i] = power
            self.integrated_power[i] += power * self.dt
            self.average_power[i] = self.integrated_power[i] / self.time_period
            self.average_profit[i] = self.average_power[i] - self.break_even_wattage[i] * self.cost[i]
        return current_power, self.average_power, self.average_profit

    def __call__(self):
        return self._evaluate_timestep()

    def message_str(self, current_power, average_power, average_profit):
        return 'Current power, average power and profit for each farm: {}, {}, {}'.format(current_power, average_power, average_profit)


class TurbineOptimisationCallback(DiagnosticOptimisationCallback):
    """
    :class:`DiagnosticOptimisationCallback` that evaluates the performance of each tidal turbine farm during an optimisation.

    See the :py:mod:`optimisation` module for more info about the use of OptimisationCallbacks."""
    name = 'farm_optimisation'
    variable_names = ['cost', 'average_power', 'average_profit']

    def __init__(self, solver_obj, turbine_functional_callback, **kwargs):
        """
        :arg solver_obj: a :class:`.FlowSolver2d` object
        :arg turbine_functional_callback: a :class:`.TurbineFunctionalCallback` used in the forward model
        :args kwargs: see :class:`.DiagnosticOptimisationCallback`"""
        self.tfc = turbine_functional_callback
        super().__init__(solver_obj, **kwargs)

    def compute_values(self, *args):
        costs = [x.block_variable.saved_output for x in self.tfc.cost]
        powers = [x.block_variable.saved_output for x in self.tfc.average_power]
        profits = [x.block_variable.saved_output for x in self.tfc.average_profit]
        return costs, powers, profits

    def message_str(self, cost, average_power, average_profit):
        return 'Costs, average power and profit for each farm: {}, {}, {}'.format(cost, average_power, average_profit)


class MinimumDistanceConstraints(pyadjoint.InequalityConstraint):
    """This class implements minimum distance constraints between turbines.

    .. note:: This class subclasses ``pyadjoint.InequalityConstraint``. The
        following methods must be implemented:

        * ``length(self)``
        * ``function(self, m)``
        * ``jacobian(self, m)``
    """
    def __init__(self, turbine_positions, minimum_distance):
        """Create MinimumDistanceConstraints

        :param turbine_positions: list of [x,y] where x and y are either float or Constant
        :param minimum_distance: The minimum distance allowed between turbines.
        """
        self._turbines = [float(xi) for xy in turbine_positions for xi in xy]
        self._minimum_distance = minimum_distance
        self._nturbines = len(turbine_positions)

    def length(self):
        """Returns the number of constraints ``len(function(m))``."""
        return int(self._nturbines*(self._nturbines-1)/2)

    def function(self, m):
        """Return an object which must be positive for the point to be feasible.

        :param m: The serialized paramaterisation of the turbines.
        :type m: numpy.ndarray.
        :returns: numpy.ndarray -- each entry must be positive for the positions to be
            feasible.
        """
        print_output("Calculating minimum distance constraints.")
        inequality_constraints = []
        for i in range(self._nturbines):
            for j in range(self._nturbines):
                if i <= j:
                    continue
                inequality_constraints.append((m[2*i]-m[2*j])**2 + (m[2*i+1]-m[2*j+1])**2 - self._minimum_distance**2)

        inequality_constraints = numpy.array(inequality_constraints)
        if any(inequality_constraints <= 0):
            print_output(
                "Minimum distance inequality constraints (should all "
                "be > 0): %s" % inequality_constraints)
        return inequality_constraints

    def jacobian(self, m):
        """Returns the gradient of the constraint function.

        Return a list of vector-like objects representing the gradient of the
        constraint function with respect to the parameter m.

        :param m: The serialized paramaterisation of the turbines.
        :type m: numpy.ndarray.
        :returns: numpy.ndarray -- the gradient of the constraint function with
            respect to each input parameter m.
        """
        print_output("Calculating gradient of equality constraint")

        grad_h = numpy.zeros((self.length(), self._nturbines*2))
        row = 0
        for i in range(self._nturbines):
            for j in range(self._nturbines):
                if i <= j:
                    continue

                grad_h[row, 2*i] = 2*(m[2*i] - m[2*j])
                grad_h[row, 2*j] = -2*(m[2*i] - m[2*j])
                grad_h[row, 2*i+1] = 2*(m[2*i+1] - m[2*j+1])
                grad_h[row, 2*j+1] = -2*(m[2*i+1] - m[2*j+1])
                row += 1

        return grad_h


def load_turbine(path, mesh2d, include_support=True, discrete=True):
    """
    Load a single tidal turbine definition file into a DiscreteTidalTurbineFarmOptions instance.

    This function reads a turbine configuration from a YAML or JSON file and
    constructs a `DiscreteTidalTurbineFarmOptions` object containing the turbine's
    physical and operational parameters. It supports both constant and table-based
    thrust and power coefficient definitions, and optionally includes support structure data.

    :param path: Path to the turbine definition file. The file must be in YAML
    (``.yaml`` / ``.yml``) or JSON format.
    :param mesh2d: The 2D mesh object used to construct the turbine density
        Function.
    :param include_support: Whether to include support structure parameters
        (if present in the file). If ``False``, support structure data is ignored
        even if provided.
    :param discrete: Whether to create a discrete turbine farm representation.
        If ``True``, returns a :class:`DiscreteTidalTurbineFarmOptions` object.
        If ``False``, returns a :class:`TidalTurbineFarmOptions` object suitable
        for continuous turbine density modelling.
    :returns: A configured turbine farm options object with parameters loaded
        from the file.
    """
    with open(path) as f:
        data = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)

    opts_cls = DiscreteTidalTurbineFarmOptions if discrete else TidalTurbineFarmOptions
    opts = opts_cls()
    opts.turbine_type = data.get("turbine_thrust_def", "constant")

    if opts.turbine_type == "table":
        opts.turbine_options.thrust_speeds = data["curves"]["speeds"]
        opts.turbine_options.thrust_coefficients = data["curves"]["thrust"]
        opts.turbine_options.power_coefficients = data["curves"]["power"]
    else:
        opts.turbine_options.thrust_coefficient = data["thrust_coefficient"]
        opts.turbine_options.power_coefficient = data["power_coefficient"]

    if include_support and "support_structure" in data:
        opts.turbine_options.structure_type = data["support_structure"]["type"]
        opts.turbine_options.rel_hub_height = data["support_structure"]["rel_hub_height"]
        opts.turbine_options.C_support = data["support_structure"]["C_support"]
        opts.turbine_options.A_support = data["support_structure"]["A_support"]

    opts.turbine_options.diameter = data["diameter"]
    if discrete:
        opts.upwind_correction = data.get("upwind_correction", True)

    opts.turbine_density = Function(FunctionSpace(mesh2d, "CG", 1),
                                    name=f"turbine_density_{data['name']}")
    return opts
