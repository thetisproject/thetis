from thetis import *
import math
import pytest


@pytest.mark.parallel(nprocs=2)
def test_steady_state_channel(do_export=False):

    lx = 5e3
    ly = 1e3
    # we don't expect converge as the reference solution neglects the advection term
    mesh2d = RectangleMesh(10, 1, lx, ly)

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(100.0)

    n = 200  # number of timesteps
    dt = 1000.
    g = float(physical_constants['g_grav'])
    f = g/lx  # linear friction coef.

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.use_nonlinear_equations = False
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.simulation_end_time = n*dt
    solver_obj.options.no_exports = not do_export
    solver_obj.options.swe_timestepper_type = 'CrankNicolson'
    solver_obj.options.swe_timestepper_options.implicitness_theta = 1.0
    solver_obj.options.swe_timestepper_options.solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'snes_type': 'newtonls',
    }
    solver_obj.options.linear_drag_coefficient = Constant(f)
    solver_obj.options.timestep = dt

    # boundary conditions
    inflow_tag = 1
    outflow_tag = 2
    inflow_func = Function(p1_2d)
    inflow_func.interpolate(Constant(-1.0))  # NOTE negative into domain
    inflow_bc = {'un': inflow_func}
    outflow_func = Function(p1_2d)
    outflow_func.interpolate(Constant(0.0))
    outflow_bc = {'elev': outflow_func}
    solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
    parameters['quadrature_degree'] = 5

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(uv=Constant((1.0, 0.0)))

    solver_obj.iterate()

    uv, eta = solver_obj.fields.solution_2d.split()

    x_2d, y_2d = SpatialCoordinate(mesh2d)
    eta_ana = interpolate(1 - x_2d/lx, p1_2d)
    area = lx*ly
    l2norm = errornorm(eta_ana, eta)/math.sqrt(area)
    print_output(l2norm)
    assert l2norm < 1e-2
    print_output("PASSED")


if __name__ == '__main__':
    test_steady_state_channel(do_export=True)
