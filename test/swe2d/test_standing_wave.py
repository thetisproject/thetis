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
    solver_obj.options.timestepper_type = timestepper

    if timestepper == 'pressureprojectionpicard':
        # this approach currently only works well with dg-cg, because in dg-dg
        # the pressure gradient term puts an additional stabilisation term in the velocity block
        # (even without that term  this approach is not as fast, as the stencil for the assembled schur system
        # is a lot bigger for dg-dg than dg-cg)
        solver_obj.options.element_family = 'dg-cg'
        solver_obj.options.use_linearized_semi_implicit_2d = True
        # solver options for the linearized wave equation terms
        solver_obj.options.solver_parameters_sw = {
            'snes_type': 'ksponly',  # we've linearized, so no snes needed
            'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
            'mat_type': 'matfree',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            # velocity mass block:
            'fieldsplit_U_2d_ksp_type': 'gmres',
            'fieldsplit_U_2d_pc_type': 'python',
            'fieldsplit_U_2d_pc_python_type': 'firedrake.AssembledPC',
            'fieldsplit_U_2d_assembled_ksp_type': 'preonly',
            'fieldsplit_U_2d_assembled_pc_type': 'bjacobi',
            'fieldsplit_U_2d_assembled_sub_pc_type': 'ilu',
            # schur system: we tell it to explicitly assemble the schur system
            # which only works with pressureprojectionicard where the velocity block is just the mass matrix
            # and if the velocity is DG so that this mass matrix can be inverted explicitly
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_1_pc_type': 'python',
            'fieldsplit_1_pc_python_type': 'thetis.AssembledSchurPC',
            # options to solve the assembled schur system
            'fieldsplit_1_schur_ksp_type': 'preonly',
            'fieldsplit_1_schur_ksp_max_it': 100,
            'fieldsplit_1_schur_ksp_converged_reason': True,
            'fieldsplit_1_schur_pc_type': 'gamg',
        }
        solver_obj.options.solver_parameters_sw_momentum = {
            'snes_monitor': True,
            'snes_type': 'ksponly',
            'ksp_type': 'gmres',
            'ksp_converged_reason': True,
            'pc_type': 'bjacobi',
            'pc_bjacobi_type': 'ilu',
        }
    if hasattr(solver_obj.options.timestepper_options, 'use_automatic_timestep'):
        solver_obj.options.timestepper_options.use_automatic_timestep = False
    solver_obj.options.timestep = dt
    solver_obj.options.shallow_water_theta = 0.5

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
