"""
Basic example of discrete turbine optimisation in a channel
============================================================

This test reproduces the "channel-optimisation" test case found in the Thetis
examples directory. It is designed to test adjoint operations with controls defined
in the real space using MPI parallelisation.
"""

from thetis import *
from firedrake.adjoint import *
import numpy as np
import random
import pytest
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def run_discrete_turbine_optimisation():
    continue_annotation()

    mesh2d = Mesh(os.path.join(script_dir, 'mesh.msh'))
    H = 50
    h_viscosity = Constant(2.)

    # create solver and set options
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
    options = solver_obj.options
    options.timestep = 1.
    options.simulation_export_time = 1.
    options.simulation_end_time = 0.5
    options.no_exports = True
    options.check_volume_conservation_2d = True
    options.element_family = 'dg-cg'
    options.swe_timestepper_type = 'SteadyState'
    # for steady state we use a direct solve (preonly+lu) in combination with a Newton snes solver
    options.swe_timestepper_options.solver_parameters = {'snes_monitor': None,
                                                         'snes_rtol': 1e-12
                                                         }
    options.horizontal_viscosity = h_viscosity

    options.quadratic_drag_coefficient = Constant(0.0025)

    left_tag = 1
    right_tag = 2
    coasts_tag = 3

    u_inflow = 3.0
    inflow_bc = {'uv': Constant((u_inflow, 0.0))}
    outflow_bc = {'elev': 0.}
    freeslip_bc = {'un': 0.}

    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: inflow_bc,
        right_tag: outflow_bc,
        coasts_tag: freeslip_bc
    }

    farm_options = DiscreteTidalTurbineFarmOptions()
    farm_options.turbine_type = 'constant'
    farm_options.turbine_options.diameter = 20
    farm_options.turbine_options.thrust_coefficient = 0.6
    farm_options.turbine_options.power_coefficient = 0.55
    farm_options.upwind_correction = False

    site_x = 320.
    site_y = 160.
    site_x_start = 160.
    site_y_start = 80.
    r = farm_options.turbine_options.diameter / 2.

    farm_options.turbine_coordinates = [[domain_constant(x + cos(y), mesh2d), domain_constant(y + numpy.sin(x), mesh2d)]
                                        for x in np.linspace(site_x_start + 4 * r, site_x_start + site_x - 4 * r, 4)
                                        for y in np.linspace(site_y_start + 0.5 * site_y - 2 * r,
                                                             site_y_start + 0.5 * site_y + 2 * r, 2)]

    options.discrete_tidal_turbine_farms[2] = [farm_options]

    solver_obj.assign_initial_conditions(uv=(as_vector((1e-3, 0.0))))

    cb = turbines.TurbineFunctionalCallback(solver_obj)
    solver_obj.add_callback(cb, 'timestep')

    solver_obj.iterate()

    power_output = sum(cb.integrated_power)
    interest_functional = power_output

    print_output("Functional in forward model {}".format(interest_functional))

    c = [Control(x) for xy in farm_options.turbine_coordinates for x in xy]

    turbine_density = Function(solver_obj.function_spaces.P1_2d, name='turbine_density')
    turbine_density.interpolate(solver_obj.tidal_farms[0].turbine_density)

    callback_list = optimisation.OptimisationCallbackList([
        optimisation.ConstantControlOptimisationCallback(solver_obj, array_dim=len(c)),
        optimisation.DerivativeConstantControlOptimisationCallback(solver_obj, array_dim=len(c)),
        optimisation.UserExportOptimisationCallback(solver_obj, [turbine_density, solver_obj.fields.uv_2d]),
        optimisation.FunctionalOptimisationCallback(solver_obj),
        turbines.TurbineOptimisationCallback(solver_obj, cb),
    ])

    def eval_cb_pre(controls):
        print_output("FORWARD RUN:")
        print_output("positions: {}".format([float(c_) for c_ in controls]))

    def derivative_cb_pre(controls):
        print_output("ADJOINT RUN:")
        print_output("positions: {}".format([float(c_.data()) for c_ in controls]))
        return controls

    rf = ReducedFunctional(-interest_functional, c, derivative_cb_post=callback_list,
                           eval_cb_pre=eval_cb_pre, derivative_cb_pre=derivative_cb_pre)

    def perturbation(r):
        dx = random.uniform(-r, r)
        return mesh2d.comm.bcast(dx, 0)

    m0 = [domain_constant(float(x) + perturbation(r), mesh2d) for xy in farm_options.turbine_coordinates for x in
          xy]

    h0 = [domain_constant(perturbation(1), mesh2d) for xy in farm_options.turbine_coordinates for x in xy]

    minconv = taylor_test(rf, m0, h0)
    print_output("Order of convergence with taylor test (should be 2) = {}".format(minconv))

    assert minconv > 1.95


@pytest.mark.parallel(2)
def test_discrete_turbine_optimisation_parallel():
    run_discrete_turbine_optimisation()


if __name__ == "__main__":
    run_discrete_turbine_optimisation()
