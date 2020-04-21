"""
Idealised channel flow in 3D
============================

Solves shallow water equations in closed rectangular domain
with sloping bathymetry.

Flow is forced with tidal volume flux in the deep (ocean) end of the
channel, and a constant volume flux in the shallow (river) end.

This test is useful for testing open boundary conditions.
"""
from thetis import *


def test_closed_channel(**user_options):
    n_layers = 3
    outputdir = 'outputs'
    lx = 100e3
    ly = 6000.
    nx = 6
    ny = 1
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    print_output('Exporting to ' + outputdir)
    t_end = 6 * 3600
    t_export = 900.0

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')

    depth_max = 20.0
    depth_min = 7.0
    xy = SpatialCoordinate(mesh2d)
    bathymetry_2d.interpolate(depth_max - (depth_max-depth_min)*xy[0]/lx)
    u_max = 4.5
    w_max = 5e-3

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = True
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = True
    options.use_limiter_for_tracers = True
    options.use_lax_friedrichs_velocity = False
    options.use_lax_friedrichs_tracer = False
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.no_exports = True
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(u_max)
    options.vertical_velocity_scale = Constant(w_max)
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.check_salinity_conservation = True
    options.check_salinity_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d',
                                'uv_dav_2d']
    options.update(user_options)

    # initial elevation, piecewise linear function
    elev_init_2d = Function(P1_2d, name='elev_2d_init')
    max_elev = 6.0
    elev_slope_x = 30e3
    elev_init_2d.interpolate(conditional(xy[0] < elev_slope_x, -xy[0]*max_elev/elev_slope_x + max_elev, 0.0))
    salt_init_3d = Constant(4.5)

    solver_obj.assign_initial_conditions(elev=elev_init_2d, salt=salt_init_3d)
    solver_obj.iterate()

    vol2d, vol2d_rerr = solver_obj.callbacks['export']['volume2d']()
    assert vol2d_rerr < 1e-12, '2D volume is not conserved'
    if options.use_ale_moving_mesh:
        vol3d, vol3d_rerr = solver_obj.callbacks['export']['volume3d']()
        assert vol3d_rerr < 1e-12, '3D volume is not conserved'
    if options.solve_salinity:
        salt_int, salt_int_rerr = solver_obj.callbacks['export']['salt_3d mass']()
        assert salt_int_rerr < 1e-8, 'salt is not conserved'
        smin, smax, undershoot, overshoot = solver_obj.callbacks['export']['salt_3d overshoot']()
        max_abs_overshoot = max(abs(undershoot), abs(overshoot))
        overshoot_tol = 1e-6
        msg = 'Salt overshoots are too large: {:}'.format(max_abs_overshoot)
        assert max_abs_overshoot < overshoot_tol, msg


if __name__ == '__main__':
    test_closed_channel(no_exports=False)
