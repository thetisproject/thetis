from thetis import *
from firedrake.petsc import PETSc
import pytest
import os


def run_steady_turbine(**model_options):
    """
    Consider a simple test case with two turbines positioned in a channel. The mesh has been adapted
    with respect to fluid speed and so has strong anisotropy in the direction of flow.

    If the default SIPG parameter is used, this steady state problem fails to converge. However,
    using the automatic SIPG parameter functionality, it should converge.
    """

    # Load an anisotropic mesh from file
    plex = PETSc.DMPlex().create()
    abspath = os.path.realpath(__file__)
    plex.createFromFile(abspath.replace('test_anisotropic.py', 'anisotropic_plex.h5'))
    mesh2d = Mesh(plex)
    x, y = SpatialCoordinate(mesh2d)

    # Create steady state solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(40.0))
    options = solver_obj.options
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_rtol': 1e-8,
        'snes_monitor': None,
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    options.output_directory = 'outputs'
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.use_grad_div_viscosity_term = False
    options.element_family = 'dg-dg'
    options.horizontal_viscosity = Constant(1.0)
    options.quadratic_drag_coefficient = Constant(0.0025)
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_grad_depth_viscosity_term = False
    options.update(model_options)
    solver_obj.create_equations()

    # Apply boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {
        1: {'uv': Constant([3.0, 0.0])},
        2: {'elev': Constant(0.0)},
        3: {'un': Constant(0.0)},
    }

    def bump(fs, locs, scale=1.0):
        """Scaled bump function for turbines."""
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2]
            expr1 = (x-x0)*(x-x0) + (y-y0)*(y-y0)
            expr2 = scale*exp(1 - 1/(1 - (x-x0)*(x-x0)/r**2))*exp(1 - 1/(1 - (y-y0)*(y-y0)/r**2))
            i += conditional(lt(expr1, r*r), expr2, 0)
        return i

    # Set up turbine array
    L = 1000.0       # domain length
    W = 300.0        # domain width
    D = 18.0         # turbine diameter
    A = pi*(D/2)**2  # turbine area
    locs = [(L/2-8*D, W/2, D/2), (L/2+8*D, W/2, D/2)]  # turbine locations

    # NOTE: We include a correction to account for the fact that the thrust coefficient is based
    #       on an upstream velocity, whereas we are using a depth averaged at-the-turbine velocity
    #       (see Kramer and Piggott 2016, eq. (15)).
    correction = 4/(1 + sqrt(1-A/(40.0*D)))**2
    scaling = len(locs)/assemble(bump(solver_obj.function_spaces.P1DG_2d, locs)*dx)
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = bump(solver_obj.function_spaces.P1DG_2d, locs, scale=scaling)
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = 0.8*correction
    solver_obj.options.tidal_turbine_farms['everywhere'] = farm_options

    # Apply initial guess of inflow velocity and solve
    solver_obj.assign_initial_conditions(uv=Constant([3.0, 0.0]))
    solver_obj.iterate()


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[True, False])
def auto_sipg(request):
    return request.param


def test_sipg(auto_sipg):
    if not auto_sipg:
        pytest.xfail("The default SIPG parameter is not sufficient for this problem.")
    run_steady_turbine(use_automatic_sipg_parameter=auto_sipg, no_exports=True)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_steady_turbine(use_automatic_sipg_parameter=False, no_exports=False)
