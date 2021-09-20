"""
Surface Ekman layer test
=======================

Unstratified water column forced by constant wind stress.
Vertical viscosity is assumed to be constant to allow simple analytical
solution.
"""
from thetis import *

depth = 20.0


def surface_ekman_test(layers=50, verify=True, iterate=True,
                       load_export_ix=None, **model_options):
    outputdir = 'outputs_ekman_surface'
    # set mesh resolution
    dx = 2500.0
    nx = 3
    lx = nx*dx
    ny = 3
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='both',
                                   reorder=True)

    dt = 90.0
    t_end = 6 * 3600.0
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

    wind_stress_x = 0.1027  # Pa
    wind_stress_2d = Constant((wind_stress_x, 0))

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = False
    options.use_turbulence = False
    options.coriolis_frequency = Constant(f_coriolis)
    options.vertical_viscosity = Constant(nu_v)
    options.vertical_diffusivity = Constant(nu_v)
    options.wind_stress = wind_stress_2d
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
    solver_obj.create_equations()

    x, y, z = SpatialCoordinate(solver_obj.mesh)

    # analytical solution
    rho0 = physical_constants['rho0']
    d = sqrt(2*nu_v/f_coriolis)
    a = sqrt(2)/(f_coriolis*d*rho0)*wind_stress_x
    z_s = z/d
    u_expr = a*exp(z_s)*cos(z_s - pi/4)
    v_expr = a*exp(z_s)*sin(z_s - pi/4)

    uv_ana_expr = as_vector((u_expr, v_expr, 0))
    uv_ana = Function(solver_obj.function_spaces.P1DGv, name='solution')
    uv_ana.interpolate(uv_ana_expr)

    out = File(options.output_directory + '/uv_analytical/uv_analytical.pvd')
    out.write(uv_ana)

    # initialize with a linear v profile to speed-up convergence
    v_init_expr = conditional(z > -d, a*(1 + z_s), 0)
    solver_obj.assign_initial_conditions(uv_3d=as_vector((v_init_expr/3, -v_init_expr, 0)))

    if iterate:
        print_output('Exporting to ' + options.output_directory)

        def export_func():
            out.write(uv_ana)

        solver_obj.iterate(export_func=export_func)

        if verify and os.getenv('THETIS_REGRESSION_TEST') is None:

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
    surface_ekman_test(layers=args.nlevels, verify=args.verify)
