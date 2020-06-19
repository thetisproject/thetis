"""
Classes related to tidal turbine farms in Thetis.
"""
from firedrake import *
from .log import *
from .callback import DiagnosticCallback
from .optimisation import DiagnosticOptimisationCallback
import numpy


class TidalTurbine:
    def __init__(self, diameter, C_support=None, A_support=None, correct_velocity=False):
        self.diameter = diameter
        self.C_support = C_support
        self.A_support = A_support
        self.correct_velocity = correct_velocity

    def velocity_correction(thrust_area, depth):
        if self.correct_velocity:
            return 0.5*(1+sqrt(1-thrust_area/(self.diameter*depth)))
        else:
            return 1

    def friction_coefficient(self, uv, depth):
        C_T = self.thrust_coefficient(uv)
        A_T = pi * self.diameter**2 / 4
        fric = C_T * A_T
        if self.C_support:
            fric += self.C_support * self.A_support
        alpha = self.velocity_correction(thrust_area, depth)
        return thrust_area/2./alpha**2


class ConstantThrustTurbine(TidalTurbine):
    def __init__(self, diameter, C_T, C_T_support=None, A_support=None):
        super().__init__(diameter, C_T_support=C_T_support, A_support=A_support)
        self.C_T = C_T

    def thrust_coefficient(uv):
        return self.C_T


class RatedThrustTurbine(TidalTurbine):
    def __init__(self, diameter, C_T, rated_speed, cut_in_speed, cut_out_speed, **kwargs):
        super().__init__(diameter, **kwargs)
        self.C_T = C_T
        self.rated_speed = rated_speed
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed

    def thrust_coefficient(uv):
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
    def __init__(self, diameter, C_T, speeds, **kwargs):
        super().__init__(diameter, **kwargs)
        if not len(C_T) == len(speeds):
            raise ValueError("In tabulated thrust curve the number of thrust coefficients and speed values should be the same.")
        self.C_T = C_T
        self.speeds = speeds

    def thrust_coefficient(uv):
        umag = dot(uv, uv)**0.5
        return conditional(umag < self.speeds[0], 0, linearly_interpolate_table(self.speeds, self.C_T, 0, umag))


def _create_turbine_from_options(velocity_correction, options):
    diameter = options.diameter
    turbine_kwargs = dict((key, options[key]) for key in ['C_support', 'A_spport'])
    if options.turbine_type == 'constant':
        C_T = options.thrust_coefficient
        turbine = ConstantThrustTurbine(diameter, C_T, **turbine_kwargs)
    elif options.turbine_type == 'rated':
        turbine_args = (options[key] for key in ['C_T', 'rated_speed', 'cut_in_speed', 'cut_out_speed'])
        turbine = RatedThrustTurbine(diameter, *turbine_args, **turbine_kwargs)
    elif options == 'table':
        turbine = TabulatedThrustTurbine(diameter, options.thrust_coefficients, options.thrust_speeds, **turbine_kwargs)
    return turbine


class TidalTurbineFarm:
    def __init__(self, turbine_density, subdomain, options, velocity_correction=False):
        """
        :arg turbine_density: turbine distribution density field
        :arg subdomain: subdomain where this farm is applied
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """
        self.turbine = _create_turbine_from_options(velocity_correction, options.turbine_options)
        self.subdomain = subdomain
        self.dx = dx(subdomain)
        self.turbine_density = turbine_density

    def number_of_turbines(self):
        return assemble(self.turbine_density * self.dx)

    def friction_coefficient(self, uv):
        return self.turbine.friction_coefficient(uv)


class DiscreteTidalTurbineFarm(TidalTurbineFarm):
    """
    Class that can be used for the addition of turbines in the turbine density field
    """

    def __init__(self, turbine_density, subdomain, options, velocity_correction=False):
        """
        :arg turbine_density: turbine distribution density field
        :arg subdomain: subdomain where this farm is applied
        :arg options: a :class:`TidalTurbineFarmOptions` options dictionary
        """

        # Preliminaries
        super().__init__(turbine, subdomain, options, velocity_correction=velocity_correction)

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
        for coord in self.coordinates:
            dx = (x-coord)/radius
            psi_x = conditional(lt(abs(dx[0]), 1), exp(1-1/(1-dx[0]**2)), 0)
            psi_x = conditional(lt(abs(dx[1]), 1), exp(1-1/(1-dx[1]**2)), 0)
            bump = psi_x * psi_y
            discrete_integral = assemble(interpolate(bump, V) * self.dx)

            unit_bump_integral = 1.45661  # integral of bump function for radius=1 (copied from OpenTidalFarm who used Wolfram)
            minimum_integral_frac = 0.9  # error if discrete integral falls below this fraction of the analytical integral
            if projection_integral < radius**2 * unit_bump_integral * minimum_integral_frac:
                raise ValueError("Could not place turbine due to low resolution. Either increase resolution or radius")

            self.turbine_density.interpolate(self.turbine_density + bump/discrete_integral)


class DiscreteTurbineOperation(DiagnosticCallback):
    """
    Callback that can be used for the following:
    a) Updating thrust and power coefficients for the solver
    b) Extract information about the turbine operation of a particular farm
    """

    def __init__(self, solver_obj, subdomain, farm_options,
                 name="Turbines2d",
                 support_structure={"C_sup": 0.6, "A_sup": None}, **kwargs):
        """
        :arg turbine: turbine characteristics
        :type turbine: object : a :class:`ThrustTurbine`
        :arg solver_obj: Thetis solver object
        :arg coordinates: Turbine coordinates array
        :arg turbine_density: turbine distribution density field
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        :arg support_structure: support structure characteristics. Add "A_sup" = None to use the total_depth
        """

        super(DiscreteTurbineOperation, self).__init__(solver_obj)
        kwargs.setdefault('append_to_log', False)
        kwargs.setdefault('export_to_hdf5', False)

        # Preliminaries
        self.solver = solver_obj
        self.subdomain_id = subdomain
        self.farm_options = farm_options
        self.turbine = farm_options.turbine_options
        self.support_structure = support_structure
        self.uv = self.solver.timestepper.solution.split()[0]
        self.functionspace = FunctionSpace(solver_obj.mesh2d, 'CG', 1)
        self.uv_ambient_correction = Function(self.functionspace, name='expected_ambient')

        # Initialising turbine related fields
        self._name = name
        self._variable_names = ["Power"]

    @property
    def name(self):
        return self._name

    @property
    def variable_names(self):
        return self._variable_names

    def wd_bathymetry_displacement(self):
        """
        Returns wetting and drying bathymetry displacement as described in:
        Karna et al.,  2011.
        """
        H = self.solver.fields["bathymetry_2d"]+self.solver.fields["elev_2d"]
        disp = Function(self.functionspace). \
            project(0.5 * (sqrt(H ** 2 + self.solver.options.wetting_and_drying_alpha ** 2) - H))
        return disp

    def compute_total_depth(self):
        """
        Returns effective depth by accounting for the wetting and drying algorithm
        """
        if hasattr(self.solver.options, 'use_wetting_and_drying') and self.solver.options.use_wetting_and_drying:
            return Function(self.functionspace). \
                project(self.solver.fields["bathymetry_2d"] + self.solver.fields["elev_2d"]
                            + self.wd_bathymetry_displacement())
        else:
            return Function(self.functionspace). \
                project(self.solver.fields["bathymetry_2d"] + self.solver.fields["elev_2d"])

    def calculate_turbine_coefficients(self, uv_mag):
        """
        :return: returns the thrust and power coefficient fields and updates the turbine drag fields
        """

        #self.farm_options.thrust_coefficient.project(
        #    conditional(le(uv_mag, self.turbine.cut_in_speed), 0,
        #                conditional(le(uv_mag, self.turbine.cut_out_speed), self.turbine.c_t_design,
        #                            self.turbine.c_t_design * self.turbine.cut_out_speed ** 3 / uv_mag ** 3)))
        self.farm_options.thrust_coefficient = Constant(self.turbine.c_t_design)

        self.farm_options.power_coefficient.\
            project(1/2 * (1 + sqrt(1 - self.farm_options.thrust_coefficient)) * self.farm_options.thrust_coefficient
                        * self.farm_options.turbine_density)

        H = self.compute_total_depth()
        if self.support_structure["A_sup"] is None:
            self.support_structure["A_sup"] = 1.0 * H
        if self.farm_options.upwind_correction is False:
            self.farm_options.turbine_drag.project((self.farm_options.thrust_coefficient * self.turbine.turbine_area
                                                       + self.support_structure["C_sup"] * self.support_structure["A_sup"])
                                                       / 2 * self.farm_options.turbine_density)
            self.uv_ambient_correction.project(uv_mag)
        else:
            self.farm_options.turbine_drag.\
                project(self.farm_options.thrust_coefficient * self.turbine.turbine_area / 2 * self.farm_options.turbine_density
                            * 4. / ((1. + sqrt(1 - self.turbine.turbine_area / (self.turbine.swept_diameter * H)
                                               * self.farm_options.thrust_coefficient)) ** 2) + self.support_structure["C_sup"]
                            * self.support_structure["A_sup"] / 2 * self.farm_options.turbine_density)

            self.uv_ambient_correction.project((1+1/4 * self.turbine.turbine_area/(self.turbine.swept_diameter * H)
                                                    * self.farm_options.thrust_coefficient) * uv_mag)

    def __call__(self):
        uv_norm = sqrt(dot(self.uv, self.uv))
        self.calculate_turbine_coefficients(uv_norm)
        return [assemble(0.5 * 1025 * self.farm_options.power_coefficient * self.turbine.turbine_area
                         * (self.uv_ambient_correction) ** 3 * dx(self.subdomain_id))]

    def message_str(self, *args):
        line = 'Tidal turbine power generated {:f} MW'.format(args[0] / 1e6)
        return line


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

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: a :class:`.FlowSolver2d` object containing the tidal_turbine_farms
        :arg kwargs: see :class:`DiagnosticCallback`"""
        nfarms = len(solver_obj.options.tidal_turbine_farms)
        super().__init__(solver_obj, array_dim=nfarms, **kwargs)

        solver_obj.create_equations()
        # TODO: was u, eta = split(solution)
        u, v, eta = solver_obj.fields.solution_2d
        dt = solver_obj.options.timestep

        self.farms = [TurbineFarm(farm_options, subdomain_id, u, v, dt) for subdomain_id, farm_options in solver_obj.options.tidal_turbine_farms.items()]
        """The sum of the number of turbines in all farms"""
        self.cost = sum(farm.cost for farm in self.farms)
        if self.append_to_log:
            print_output('Number of turbines = {}'.format(self.cost))

    def __call__(self):
        return numpy.transpose([farm.evaluate_timestep() for farm in self.farms])

    def message_str(self, current_power, average_power, average_profit):
        return 'Current power, average power and profit for each farm: {}, {}, {}'.format(current_power, average_power, average_profit)

    @property
    def average_profit(self):
        """The sum of the time-averaged profit output of all farms"""
        return sum(farm.average_profit for farm in self.farms)

    @property
    def average_power(self):
        """The sum of the time-averaged power output of all farms"""
        return sum(farm.average_power for farm in self.farms)

    @property
    def integrated_power(self):
        """The sum of the time-integrated power output of all farms"""
        return sum(farm.integrated_power for farm in self.farms)


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
