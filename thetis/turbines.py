"""
Classes related to tidal turbine farms in Thetis.
"""
from firedrake import *
from firedrake.petsc import PETSc
from .log import *
from .callback import DiagnosticCallback
from .optimisation import DiagnosticOptimisationCallback
import pyadjoint
import numpy


class TidalTurbine:
    def __init__(self, options, upwind_correction=False):
        self.diameter = options.diameter
        self.C_support = options.C_support
        self.A_support = options.A_support
        self.upwind_correction = upwind_correction

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
            return 0.5*(1+sqrt(1-fric/(self.diameter*depth)))
        else:
            return 1

    def friction_coefficient(self, uv, depth):
        thrust_area = self._thrust_area(uv)
        alpha = self.velocity_correction(uv, depth)
        return thrust_area/2./alpha**2

    def power(self, uv, depth):
        # ratio of discrete to upstream velocity (NOTE: should include support drag!)
        alpha = self.velocity_correction(uv, depth)
        C_T = self.thrust_coefficient(uv)
        A_T = pi * self.diameter**2 / 4
        uv3 = dot(uv, uv)**1.5 / alpha**3  # upwind cubed velocity
        # this assumes the velocity through the turbine does not change due to the support (is this correct?)
        return 0.25*C_T*A_T*(1+sqrt(1-C_T))*uv3


class ConstantThrustTurbine(TidalTurbine):
    def __init__(self, options, upwind_correction=False):
        super().__init__(options, upwind_correction=upwind_correction)
        self.C_T = options.thrust_coefficient

    def thrust_coefficient(self, uv):
        return self.C_T


class RatedThrustTurbine(TidalTurbine):
    def __init__(self, options, upwind_correction=False):
        super().__init__(options, upwind_correction=upwind_correction)
        self.C_T = options.thrust_coefficient
        self.rated_speed = options.rated_speed
        self.cut_in_speed = options.cut_in_speed
        self.cut_out_speed = options.cut_out_speed

    def thrust_coefficient(self, uv):
        umag = dot(uv, uv)**0.5
        # C_P for |u|>u_rated:
        y = self.C_T * (1+sqrt(1-self.C_T))/2 * self.cut_out_speed**3 / umag**3
        # from this compute C_T s.t. C_P=C_T*(1+sqrt(1-C_T)/2, or
        # equivalently: 4*C_P^2 - 4*C_T*C_P + C_T^3 = 0
        # a cube root from this cubic equation is obtained using Cardano's formula
        d = 4*y**4-64/27*y**3
        C = (-d+4*y**4)**(1/6)
        # the additiona 4pi/3 ensures we obtained the "right" cube with 0<C_T<0.85
        C_T_rated = (C+4*y/(3*C)) * cos(atan2(sqrt(-d), -2*y**2)/3 + 4*pi/3)

        return conditional(umag < self.cut_in_speed, 0,
                           conditional(umag < self.rated, self.C_T,
                                       conditional(umag < self.cut_out, C_T_rated, 0)))


def linearly_interpolate_table(x_list, y_list, y_final, x):
    """Return UFL expression that linearly interpolates between y-values in x-points

    :param x_list: (1D) x-points
    :param y_list: y-values in those points
    :param y_final: value for x>x_list[-1]
    :param x: point to interpolate (assumed x>x_list[0])
    """
    # below x1, interpolate between x0 and x1:
    below_x1 = ((x_list[1]-x)*y_list[0] + (x-x_list[0])*y_list[1])/(y_list[1]-y_list[0])
    # above x1, interpolate from rest of the table, or take final value:
    if len(x_list) > 2:
        above_x1 = linearly_interpolate(x_list[1:], y_list[1:], y_final, x)
    else:
        above_x1 = y_final

    return conditional(x < x_list[1], below_x1, above_x1)


class TabulatedThrustTurbine(TidalTurbine):
    def __init__(self, options, upwind_correction=False):
        super().__init__(options, upwind_correction=upwind_correction)
        self.C_T = options.thrust_coefficients
        self.speeds = options.thrust_speeds
        if not len(self.C_T) == len(self.speeds):
            raise ValueError("In tabulated thrust curve the number of thrust coefficients and speed values should be the same.")

    def thrust_coefficient(self, uv):
        umag = dot(uv, uv)**0.5
        return conditional(umag < self.speeds[0], 0, linearly_interpolate_table(self.speeds, self.C_T, 0, umag))


class TidalTurbineFarm:
    def __init__(self, turbine_density, subdomain, options):
        """
        :arg turbine_density: turbine distribution density field
        :arg subdomain: subdomain where this farm is applied
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """
        upwind_correction = getattr(options, 'upwind_correction', False)
        if options.turbine_type == 'constant':
            self.turbine = ConstantThrustTurbine(options.turbine_options, upwind_correction=upwind_correction)
        elif options.turbine_type == 'rated':
            self.turbine = RatedThrustTurbine(options.turbine_options, upwind_correction=upwind_correction)
        elif options.turbine_type == 'table':
            self.turbine = TabulatedThrustTurbine(options.turbine_options, upwind_correction=upwind_correction)
        self.subdomain = subdomain
        self.dx = dx(subdomain)
        self.turbine_density = turbine_density

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

    def __init__(self, turbine_density, subdomain, options):
        """
        :arg turbine_density: turbine distribution density field
        :arg subdomain: subdomain where this farm is applied
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """

        # Preliminaries
        super().__init__(turbine_density, subdomain, options)

        # Adding turbine distribution in the domain
        self.add_turbines(options.turbine_coordinates)

    def add_turbines(self, coordinates):
        """
        :param coords: Array with turbine coordinates to be positioned
        :param function: turbine density function to be adapted
        :param mesh: computational mesh domain
        :param radius: radius where the bump will be applied
        :return: updated turbine density field
        """
        x = SpatialCoordinate(self.turbine_density.ufl_domain())
        V = self.turbine_density.function_space()

        radius = self.turbine.diameter * 0.5
        for coord in coordinates:
            dx0 = (x[0] - coord[0])/radius
            dx1 = (x[1] - coord[1])/radius
            psi_x = conditional(lt(abs(dx0), 1), exp(1-1/(1-dx0**2)), 0)
            psi_y = conditional(lt(abs(dx1), 1), exp(1-1/(1-dx1**2)), 0)
            bump = psi_x * psi_y
            discrete_integral = assemble(interpolate(bump, V) * self.dx)

            unit_bump_integral = 1.45661  # integral of bump function for radius=1 (copied from OpenTidalFarm who used Wolfram)
            minimum_integral_frac = 0.9  # error if discrete integral falls below this fraction of the analytical integral
            min_discrete_integral = radius**2 * unit_bump_integral * minimum_integral_frac
            if discrete_integral < min_discrete_integral:
                raise ValueError("Could not place turbine due to low resolution. Either increase resolution or radius")

            # FIXME: pyadjoint workaround #22
            dic = Constant(0)
            dic.assign(discrete_integral)
            # NOTE that we still cap the discrete integral for the adjoint
            self.turbine_density.interpolate(self.turbine_density + bump/max_value(discrete_integral, min_discrete_integral))


class TurbineFarm(object):
    """
    Evaluates power output, costs and profit of tidal turbine farm

    Cost is simply the number of turbines evaluated as the integral of the
    turbine density.

    Power output is calculated as the amount of energy
    taken out of the flow by the turbine drag term. This is in general
    an over-estimate of the 'usefully extractable' energy. Furthermore no
    density is included, so that assuming a density of 1000 kg/m^3 and
    all further quantities in SI, the power is measured in kW.

    Profit is calculated as:

      Profit = Power - break_even_wattage * Cost

    With the above assumptions, break_even_wattage should be specified in
    kW and can be interpreted as the average power per turbine required
    to 'break even', i.e. Profit=0.

    Power output and profit are time-integrated (simple first order) and
    time-averages are available as average_power and average_profit.
    """
    def __init__(self, farm_options, subdomain_id, u, v, dt):
        """
        :arg farm_options: a :class:`TidalTurbineFarmOptions` object that define the farm and the turbines used
        :arg int subdomain_id: the farm is restricted to this subdomain
        :arg u,v: the depth-averaged velocity field
        :arg float dt: used for time-integration."""
        turbine_density = farm_options.turbine_density
        C_T = farm_options.turbine_options.thrust_coefficient
        A_T = pi*(farm_options.turbine_options.diameter/2.)**2
        C_D = C_T*A_T/2.*turbine_density
        self.power_integral = C_D * (u*u + v*v)**1.5 * dx(subdomain_id)
        # cost integral is n/o turbines = \int turbine_density
        self.cost = assemble(turbine_density * dx(subdomain_id))
        self.break_even_wattage = farm_options.break_even_wattage
        self.dt = dt

        # time-integrated quantities:
        self.integrated_power = 0.
        self.average_power = 0.
        self.average_profit = 0.
        self.time_period = 0.

    @PETSc.Log.EventDecorator("thetis.TurbineFarm.evaluate_timestep")
    def evaluate_timestep(self):
        """Perform time integration and return current power and time-averaged power and profit."""
        self.time_period = self.time_period + self.dt
        current_power = assemble(self.power_integral)
        self.integrated_power = self.integrated_power + current_power * self.dt
        self.average_power = self.integrated_power / self.time_period
        self.average_profit = self.average_power - self.break_even_wattage * self.cost
        return current_power, self.average_power, self.average_profit


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
        self.break_even_wattage = [getattr(farm, 'break_even_wattage', 0) for farm in self.farms]

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
        costs = [farm.cost.block_variable.saved_output for farm in self.tfc.farms]
        powers = [farm.average_power.block_variable.saved_output for farm in self.tfc.farms]
        profits = [farm.average_profit.block_variable.saved_output for farm in self.tfc.farms]
        return costs, powers, profits

    def message_str(self, cost, average_power, average_profit):
        return 'Costs, average power and profit for each farm: {}, {}, {}'.format(cost, average_power, average_profit)


class MinimumDistanceConstraints(pyadjoint.InequalityConstraint):
    """This class implements minimum distance constraints between turbines.

    .. note:: This class subclasses `pyadjoint.InequalityConstraint`_. The
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

        return grad_h
