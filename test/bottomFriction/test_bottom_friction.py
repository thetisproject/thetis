"""
Steady-state channel flow in 3D
===============================

Steady state flow in a channel subject to bottom friction.

This test reproduces the "channel" test case found in GOTM test suite [1]
and also [2].

This setup uses coarser mesh to save CPU time.

This case tests the turbulence closure model, vertical viscosity and bottom
boundary layer. Water column is initially at rest. Circulation is driven by
a constant elevation gradient until it reaches a steady state. Here the
elevation gradient is replaced by an equivalent source term in the
momentum equation.


[1] http://www.gotm.net/
[2] Karna et al. (2012). Coupling of a discontinuous Galerkin finite element
    marine model with a finite difference turbulence closure model.
    Ocean Modelling, 47:55-64.
    http://dx.doi.org/10.1016/j.ocemod.2012.01.001
"""
from thetis import *
import pytest


def run_bottom_friction(do_assert=True, do_export=False, **model_options):
    outputdir = 'outputs'
    # set mesh resolution
    dx = 2500.0
    layers = 20
    depth = 15.0

    nx = 3  # nb elements in flow direction
    lx = nx*dx
    ny = 2  # nb elements in cross direction
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x', reorder=True)

    print_output('Exporting to ' + outputdir)
    dt = 25.0
    t_end = 5 * 3600.0  # sufficient to reach ~steady state
    t_export = 400.0
    u_mag = 1.0

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry2d = Function(p1_2d, name='Bathymetry')
    bathymetry2d.assign(depth)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = True
    options.bottom_roughness = Constant(1.5e-3)
    options.use_turbulence = True
    options.vertical_viscosity = Constant(1.3e-6)  # background value
    options.vertical_diffusivity = Constant(1.4e-7)  # background value
    options.use_ale_moving_mesh = False
    options.use_limiter_for_tracers = True
    options.simulation_export_time = t_export
    options.timestep = dt
    options.simulation_end_time = t_end
    options.no_exports = not do_export
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(u_mag)
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'uv_dav_2d', 'eddy_visc_3d', 'shear_freq_3d',
                                'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
    options.update(model_options)
    if options.timestepper_type == 'LeapFrog':
        options.use_ale_moving_mesh = True
    if hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestepper_options.use_automatic_timestep = False

    solver_obj.create_function_spaces()

    # drive flow with momentum source term equivalent to constant surface slope
    surf_slope = -1.0e-5  # d elev/dx
    g = float(physical_constants['g_grav'])
    pressure_grad = -g * surf_slope
    options.momentum_source_2d = Constant((pressure_grad, 0))

    solver_obj.create_equations()

    # speed-up convergence by stating with u > 0
    u_init_2d = 0.5
    solver_obj.assign_initial_conditions(uv_2d=Constant((u_init_2d, 0)))
    # consistent 3d velocity with slope
    xyz = SpatialCoordinate(solver_obj.mesh)
    solver_obj.fields.uv_3d.project(as_vector((u_init_2d*0.3*(xyz[2]/depth + 0.5), 0, 0)))

    if do_assert:
        # compare against logarithmic velocity profile
        # u = u_b / kappa * log((z + bath + z_0)/z_0)
        u_b = sqrt(-g * surf_slope * depth)
        kappa = solver_obj.options.turbulence_model_options.kappa
        l2_tol = 0.05
        z_0 = float(options.bottom_roughness)
        log_uv = Function(solver_obj.function_spaces.P1DGv, name='log velocity')
        log_uv.project(as_vector((u_b / kappa * ln((xyz[2] + depth + z_0)/z_0), 0, 0)))
        if do_export:
            out = File(outputdir + '/log_uv/log_uv.pvd')
            out.write(log_uv)

    solver_obj.iterate()

    if do_assert:
        uv_p1_dg = Function(solver_obj.function_spaces.P1DGv, name='velocity p1dg')
        uv_p1_dg.project(solver_obj.fields.uv_3d + solver_obj.fields.uv_dav_3d)
        volume = lx*ly*depth
        uv_l2_err = errornorm(log_uv, uv_p1_dg)/numpy.sqrt(volume)
        assert uv_l2_err < l2_tol, 'L2 error is too large: {:} > {:}'.format(uv_l2_err, l2_tol)
        print_output('L2 error {:.4f} PASSED'.format(uv_l2_err))


@pytest.fixture(params=['rt-dg', 'dg-dg', 'bdm-dg'])
def element_family(request):
    return request.param


@pytest.fixture(params=['LeapFrog', 'SSPRK22'])
def timestepper_type(request):
    return request.param


def test_bottom_friction(element_family, timestepper_type):
    run_bottom_friction(do_assert=True, do_export=False,
                        element_family=element_family,
                        timestepper_type=timestepper_type)


if __name__ == '__main__':
    run_bottom_friction(element_family='dg-dg', timestepper_type='SSPRK22',
                        do_assert=True, do_export=True)
