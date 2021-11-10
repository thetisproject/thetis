"""
Classes related to tidal turbine farms in Thetis.
"""
from firedrake import *
from firedrake.petsc import PETSc
from .log import *
from .callback import DiagnosticCallback
from .optimisation import DiagnosticOptimisationCallback
import numpy


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
