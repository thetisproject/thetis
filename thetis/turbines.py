"""
Classes related to tidal turbine farms in Thetis.
"""
from .firedrake import *
from .log import *
from .callback import DiagnosticCallback
from .optimisation import DiagnosticOptimisationCallback
import numpy


class BaseTurbine(object):
    """A base turbine class from which others are derived."""
    def __init__(self, diameter=None, minimum_distance=None,
                 controls=None, bump=False, projection_diameter=None):
        # Possible turbine parameters.
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._controls = controls
        self._projection_diameter = projection_diameter

        # Possible parameterisations.
        self._bump = bump

        # The integral of the unit bump function computed with Wolfram Alpha:
        # "integrate e^(-1/(1-x**2)-1/(1-y**2)+2) dx dy,
        #  x=-0.999..0.999, y=-0.999..0.999"
        # http://www.wolframalpha.com/input/?i=integrate+e%5E%28-1%2F%281-x**2%29-1%2F%281-y**2%29%2B2%29+dx+dy%2C+x%3D-0.999..0.999%2C+y%3D-0.999..0.999
        self._unit_bump_int = 1.45661

    @property
    def diameter(self):
        """The diameter of a turbine.
        :returns: The diameter of a turbine.
        :rtype: float
        """
        if self._diameter is None:
            raise ValueError("Diameter has not been set!")
        return self._diameter

    @property
    def radius(self):
        """The radius of a turbine.
        :returns: The radius of a turbine.
        :rtype: float
        """
        return self.diameter*0.5

    @property
    def integral(self):
        """The integral of the turbine bump function.
        :returns: The integral of the turbine bump function.
        :rtype: float
        """
        return self._unit_bump_int*self._diameter/4.

    @property
    def bump(self):
        return self._bump


class ThrustTurbine(BaseTurbine):
    """ Create a turbine that is modelled as a bump of bottom friction.
        In addition this turbine implements cut in and out speeds for the
        power production.
        This turbine introduces a non-linearity, which is handled explicitly. """
    def __init__(self,
                 diameter=20.,
                 swept_diameter=20.,
                 c_t_design=0.6,
                 cut_in_speed=1,
                 cut_out_speed=2.5,
                 minimum_distance=None):

        # Check for a given minimum distance.
        if minimum_distance is None:
            minimum_distance = diameter*1.5
        # Initialize the base class.
        super(ThrustTurbine, self).__init__(diameter=diameter, minimum_distance=minimum_distance)

        # To parametrise a square 2D plan-view turbine to characterise a
        # realistic tidal turbine with a circular swept area in the section
        # plane we assume that the specified 2D turbine diameter is equal to the
        # circular swept diameter
        self.swept_diameter = swept_diameter
        self.c_t_design = c_t_design
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed
        self.turbine_area = pi * self.diameter ** 2 / 4

        self.swept_area = pi * (swept_diameter) ** 2 / 4
        # Check that the parameter choices make some sense - these won't break
        # the simulation but may give unexpected results if the choice isn't
        # understood.
        if self.swept_diameter != self.diameter:
            log(INFO, 'Warning - swept_diameter and plan_diameter are not equal')


class DiscreteTidalfarm(object):
    """
    Class that can be used for the addition of turbines in the turbine density field
    """

    def __init__(self, solver_obj, turbine, coordinates, turbine_density, subdomain, **kwargs):
        """
        :arg turbine: turbine characteristics
        :type turbine: object : a :class:`ThrustTurbine`
        :arg solver_obj: Thetis solver object
        :arg coordinates: Turbine coordinates array
        :arg turbine_density: turbine distribution density field
        """

        # Preliminaries
        self.solver = solver_obj
        self.turbine = turbine
        self.coordinates = coordinates
        self.farm_density = turbine_density
        self.functionspace = FunctionSpace(solver_obj.mesh2d, 'CG', 1)
        self.subdomain_id = subdomain

        # Adding turbine distribution in the domain
        self.add_turbines()

    def add_turbines(self):
        """
        :param coords: Array with turbine coordinates to be positioned
        :param function: turbine density function to be adapted
        :param mesh: computational mesh domain
        :param radius: radius where the bump will be applied
        :return: updated turbine density field
        """
        self.farm_density.assign(0.0)
        x = SpatialCoordinate(self.solver.mesh2d)
        psi_x = Function(self.functionspace)
        psi_y = Function(self.functionspace)
        radius = self.turbine.swept_diameter * 0.5
        for coord in self.coordinates:
            psi_x.project(conditional(lt(abs((x[0]-coord[0])/radius), 1),
                                          exp(1-1/(1-pow(abs((x[0]-coord[0])/radius), 2))), 0))
            psi_y.project(conditional(lt(abs((x[1]-coord[1])/radius), 1),
                                          exp(1-1/(1-pow(abs((x[1]-coord[1])/radius), 2))), 0))
            projection_integral = assemble(Function(self.functionspace).
                                           project(psi_x * psi_y / (self.turbine._unit_bump_int * radius**2))
                                           * dx(self.subdomain_id))

            if projection_integral == 0.0:
                print_output("Could not place turbine due to low resolution. Either increase resolution or radius")
            else:
                density_correction = 1 / projection_integral
                self.farm_density.project(self.farm_density +  psi_x * psi_y
                                              / (self.turbine._unit_bump_int * radius ** 2))


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
