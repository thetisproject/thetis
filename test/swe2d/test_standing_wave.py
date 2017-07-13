# Test for temporal convergence of CrankNicolson and pressureprojection picard timesteppers,
# tests convergence of a single period of a standing wave in a rectangular channel.
# This only tests against a linear solution, so does not really test whether the splitting
# in PressureProjectionPicard between nonlinear momentum and linearized wave equation terms is correct.
# PressureProjectionPicard does need two iterations to ensure 2nd order convergence
from thetis import *
import pytest
import math


@pytest.mark.parametrize("timesteps,max_rel_err", [
    (10, 0.02), (20, 5e-3)])
# with nonlin=True and nx=100 this converges for the series
#  (10,0.02), (20,5e-3), (40, 1.25e-3)
# with nonlin=False further converge is possible
@pytest.mark.parametrize("timestepper", [
    'CrankNicolson', 'PressureProjectionPicard', ])
def test_standing_wave_channel(timesteps, max_rel_err, timestepper, do_export=False):

    lx = 5e3
    ly = 1e3
    nx = 50
    mesh2d = RectangleMesh(nx, 1, lx, ly)

    n = timesteps
    depth = 100.
    g = physical_constants['g_grav'].dat.data[0]
    c = math.sqrt(g*depth)
    period = 2*lx/c
    dt = period/n
    t_end = period-0.1*dt  # make sure we don't overshoot

    x = SpatialCoordinate(mesh2d)
    elev_init = cos(pi*x[0]/lx)

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(depth)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.use_nonlinear_equations = True
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.simulation_end_time = t_end
    solver_obj.options.no_exports = not do_export
    solver_obj.options.element_family = 'dg-dg'
    solver_obj.options.timestepper_type = timestepper
    if timestepper == 'CrankNicolson':
        # when linearising the equations, CrankNicolson (theta=0.5) stops being 2nd order
        # (for PressureProjectionPicard we restore 2nd order because of the 2 iterations)
        solver_obj.options.timestepper_options.use_semi_implicit_linearization = False
    if hasattr(solver_obj.options.timestepper_options, 'use_automatic_timestep'):
        solver_obj.options.timestepper_options.use_automatic_timestep = False
    solver_obj.options.timestep = dt

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(elev=elev_init)

    solver_obj.iterate()

    uv, eta = solver_obj.fields.solution_2d.split()

    area = lx*ly
    rel_err = errornorm(elev_init, eta)/math.sqrt(area)
    print_output(rel_err)
    assert(rel_err < max_rel_err)
    print_output("PASSED")


if __name__ == '__main__':
    test_standing_wave_channel(do_export=True)
