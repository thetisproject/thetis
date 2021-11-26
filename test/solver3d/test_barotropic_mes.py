"""
Convergence test for a standing wave in a closed rectangular domain with passive
tracers.

"""
from thetis import *
from scipy import stats
import pytest


def run(refinement=1, ncycles=2, **kwargs):

    print_output(' --------\nRunning refinement {:}'.format(refinement))

    conservation_check = kwargs.pop('conservation_check', False)

    g_grav = float(physical_constants['g_grav'])
    depth = 100.0
    c_wave = numpy.sqrt(g_grav*depth)

    n_base = 20
    nx = n_base*refinement
    ny = 1
    lx = 60000.
    ly = lx/nx
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    elev_amp = 10.0 if conservation_check else 0.01
    n_layers = 2*refinement
    # estimate of max advective velocity used to estimate time step
    u_mag = Constant(0.5)
    print_output('Triangle edge length {:} m'.format(lx/nx))
    print_output('Number of layers {:}'.format(n_layers))

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # set time step, export interval and run duration
    n_steps = 1 if not conservation_check else 20
    T_cycle = lx/c_wave
    t_step = T_cycle/n_steps
    t_end = ncycles*T_cycle
    t_export = t_end if not conservation_check else t_step

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SSPRK22'
    options.use_nonlinear_equations = True
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = False
    options.use_baroclinic_formulation = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = True
    options.use_limiter_for_tracers = True
    options.use_limiter_for_velocity = False
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = u_mag
    options.check_volume_conservation_2d = conservation_check
    options.check_volume_conservation_3d = conservation_check
    options.check_salinity_conservation = conservation_check
    options.check_salinity_overshoot = conservation_check
    options.check_temperature_conservation = conservation_check
    options.check_temperature_overshoot = conservation_check
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                                'uv_dav_2d']
    options.fields_to_export_hdf5 = []
    options.update(kwargs)

    # need to call creator to create the function spaces
    solver_obj.create_equations()
    xy = SpatialCoordinate(solver_obj.mesh2d)
    elev_init = -elev_amp*cos(2*pi*xy[0]/lx)
    salt_init3d = temp_init3d = None
    if options.solve_salinity:
        salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
        salt_init3d.assign(4.5)
    if options.solve_temperature:
        temp_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
        x, y, z = SpatialCoordinate(solver_obj.mesh)
        temp_init3d.interpolate(5*sin(2*pi*x/lx) + 10.)

    solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, temp=temp_init3d)
    solver_obj.iterate()

    area = lx*ly
    elev_err = errornorm(elev_init, solver_obj.fields.elev_2d)/numpy.sqrt(area)
    uv_err = errornorm(as_vector((0., 0.)), solver_obj.fields.uv_2d)/numpy.sqrt(area)
    print_output('L2 error elev={:}, uv={:}'.format(elev_err, uv_err))

    return elev_err, uv_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    if saveplot and COMM_WORLD.size > 1:
        raise Exception('Cannot use matplotlib in parallel')
    polynomial_degree = options.get('polynomial_degree', 1)
    space_str = options.get('element_family')
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log_elev = numpy.log10(numpy.array([v[0] for v in l2_err]))
    y_log_uv = numpy.log10(numpy.array([v[1] for v in l2_err]))
    setup_name = 'standingwave'

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot, ax):
        slope_rtol = 0.07
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if saveplot:
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_str)
        if expected_slope is not None:
            err_msg = '{:}: Wrong {:} convergence rate {:.4f}, expected {:.4f}'.format(setup_name, field_str, slope, expected_slope)
            assert slope > expected_slope*(1 - slope_rtol), err_msg
            print_output('{:}: {:} convergence rate {:.4f} PASSED'.format(setup_name, field_str, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    ax_list = [None, None]
    if saveplot:
        import matplotlib.pyplot as plt
        fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 4))

    try:
        check_convergence(x_log, y_log_elev, polynomial_degree+1, 'Elevation', saveplot, ax_list[0])
    except Exception as e1:
        print(e1)
    try:
        check_convergence(x_log, y_log_uv, polynomial_degree+1, 'Velocity', saveplot, ax_list[1])
    except Exception as e2:
        print(e2)

    if saveplot:
        ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
        degree_str = 'o{:}'.format(polynomial_degree)
        imgfile = '_'.join(['convergence', setup_name, ref_str, degree_str, space_str])
        imgfile += '.pdf'
        imgdir = create_directory('plots')
        imgfile = os.path.join(imgdir, imgfile)
        print_output('saving figure {:}'.format(imgfile))
        plt.savefig(imgfile, dpi=200, bbox_inches='tight')


@pytest.mark.parallel(nprocs=2)
def test_standing_wave():
    run_convergence([1, 2, 4, 8],
                    polynomial_degree=1, element_family='dg-dg',
                    saveplot=False, no_exports=True)


if __name__ == '__main__':
    run_convergence([1, 2, 4, 6, 8, 10],
                    polynomial_degree=1, element_family='dg-dg',
                    saveplot=False, no_exports=True)
