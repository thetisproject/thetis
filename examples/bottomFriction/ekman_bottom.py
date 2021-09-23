"""
Bottom Ekman layer test
=======================

Steady state flow in a channel subject to bottom friction and rotation.
Vertical viscosity is assumed to be constant to allow simple analytical
solution.
"""
from thetis import *

depth = 20.0
surf_slope = -5.0e-6  # d elev/dx


def bottom_ekman_test(layers=50, verify=True, iterate=True,
                      load_export_ix=None, **model_options):
    outputdir = 'outputs_ekman_bottom'
    # set mesh resolution
    dx = 2500.0
    nx = 3
    lx = nx*dx
    ny = 3
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='both',
                                   reorder=True)

    dt = 90.0
    t_end = 5 * 3600.0  # sufficient to reach ~steady state
    t_export = 450.0
    u_mag = 1.0

    f_coriolis = 1e-4
    nu_v = 5e-4

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_end = 5*t_export

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry2d = Function(p1_2d, name='Bathymetry')
    bathymetry2d.assign(depth)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = True
    options.bottom_roughness = Constant(1e-3)
    options.use_turbulence = False
    options.coriolis_frequency = Constant(f_coriolis)
    options.vertical_viscosity = Constant(nu_v)
    options.vertical_diffusivity = Constant(nu_v)
    options.simulation_export_time = t_export
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = dt
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = Constant(u_mag)
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'uv_dav_2d']
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d']
    options.update(model_options)
    layer_str = 'nz{:}'.format(layers)
    odir = '_'.join([outputdir, layer_str])
    options.output_directory = odir

    solver_obj.create_function_spaces()

    # drive flow with momentum source term equivalent to constant surface slope
    g = float(physical_constants['g_grav'])
    pressure_grad = -g * surf_slope
    options.momentum_source_2d = Constant((pressure_grad, 0))

    solver_obj.create_equations()

    v_init_2d = -0.49
    solver_obj.assign_initial_conditions(uv_2d=Constant((0, v_init_2d)))

    x, y, z = SpatialCoordinate(solver_obj.mesh)

    if iterate:
        print_output('Exporting to ' + options.output_directory)
        solver_obj.iterate()

        if verify and os.getenv('THETIS_REGRESSION_TEST') is None:
            # analytical solution (assuming no-slip bottom)
            v_max = 0.4905  # u = g/f d(elev)/dx
            d = sqrt(2*nu_v/f_coriolis)
            z_b = (depth + z)/d
            v_expr = -v_max * (1 - exp(-z_b)*cos(z_b))
            u_expr = v_max * exp(-z_b)*sin(z_b)

            uv_ana_expr = as_vector((u_expr, v_expr, 0))
            uv_ana = Function(solver_obj.function_spaces.P1DGv, name='solution')
            uv_ana.interpolate(uv_ana_expr)

            out = File(options.output_directory + '/uv_analytical/uv_analytical.pvd')
            out.write(uv_ana)

            l2_tol = 0.05
            uv_p1_dg = Function(solver_obj.function_spaces.P1DGv, name='velocity p1dg')
            uv_p1_dg.project(solver_obj.fields.uv_3d + solver_obj.fields.uv_dav_3d)
            volume = lx*ly*depth
            uv_l2_err = errornorm(uv_ana_expr, uv_p1_dg)/numpy.sqrt(volume)
            assert uv_l2_err < l2_tol, 'L2 error is too large: {:} > {:}'.format(uv_l2_err, l2_tol)
            print_output('L2 error {:.4f} PASSED'.format(uv_l2_err))
    elif load_export_ix is not None:
        print_output('Loading state: {:}'.format(load_export_ix))
        solver_obj.load_state(load_export_ix)

    return solver_obj


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run bottom friction test case',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', '--nlevels', type=int, default=50,
                        help='number of vertical levels')
    parser.add_argument('-v', '--verify', action='store_true',
                        help='Verify correctness against log profile.')

    args = parser.parse_args()
    bottom_ekman_test(layers=args.nlevels, verify=args.verify)
