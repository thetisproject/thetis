from thetis import *


def construct_solver(store_station_time_series=True, **model_options):
    lx = 100e3
    nx = 30
    delta_x = lx/nx
    ny = 2
    ly = delta_x * ny
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    t_end = 8 * 3600.
    u_mag = Constant(6.0)
    t_export = 600.
    dt = 600.

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_end = 5*t_export
    pwd = os.path.abspath(os.path.dirname(__file__))

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    # assign bathymetry to a linear function
    x, y = SpatialCoordinate(mesh2d)
    depth_oce = 10.0
    depth_riv = 5.0
    bathymetry_2d.interpolate(depth_oce + (depth_riv - depth_oce)*x/lx)

    # friction Manning coefficient
    manning_2d = Function(P1_2d, name='Manning coefficient')
    manning_low = 1e-3
    manning_high = 1.5e-2
    manning_2d.interpolate(
        conditional(x < 60e3, manning_low, manning_high))

    # create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.manning_drag_coefficient = manning_2d
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = u_mag
    options.output_directory = f'{pwd}/outputs_forward'
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.swe_timestepper_type = 'CrankNicolson'
    if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt
    options.update(model_options)

    solver_obj.add_new_field(manning_2d, 'manning_2d',
                             manning_2d.name(),
                             'Manning2d', unit='s m-1/3')

    solver_obj.create_equations()

    if store_station_time_series:
        # store elevation time series at stations
        stations = [
            ('stationA', (30e3, ly/2)),
            ('stationB', (80e3, ly/2)),
            ('stationC', (10e3, ly/2)),
            ('stationD', (60e3, ly/2)),
            ('stationE', (96e3, ly/2)),
        ]
        for name, (sta_x, sta_y) in stations:
            cb = TimeSeriesCallback2D(
                solver_obj, ['elev_2d'], sta_x, sta_y, name,
                append_to_log=False
            )
            solver_obj.add_callback(cb)

    # set initial condition for elevation, piecewise linear function
    elev_init_2d = Function(P1_2d, name='Initial elevation')
    elev_height = 6.0
    elev_ramp_lx = 80e3
    elev_init_2d.interpolate(
        conditional(x < elev_ramp_lx, elev_height*(1 - x/elev_ramp_lx), 0.0))

    solver_obj.add_new_field(elev_init_2d, 'elev_init_2d',
                             elev_init_2d.name(),
                             'InitialElev2d', unit='m')

    return solver_obj
