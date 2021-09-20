"""
Shallow water test cases on the sphere by Willamson et al. (1992) and
Läuter et al (2005).

[1] Williamson et al., 1992. A standard test set for numerical approximations
    to the shallow water equations in spherical geometry. Journal of
    Computational Physics, (1):211–224.
    https://doi.org/10.1016/S0021-9991(05)80016-6
[2] Läuter et al., 2005. Unsteady analytical solutions of the spherical shallow
    water equations. Journal of Computational Physics, (2):535–553.
    https://doi.org/10.1016/j.jcp.2005.04.022
"""
from thetis import *
from scipy import stats
import pytest

r_earth = 6371220.  # radius of Earth
omega = 7.292e-5  # Earth's angular velocity


def coords_xyz_to_lonlat(mesh):
    """
    Convert Earth-centered Cartesian coordinates to (longitude, latitude)
    """
    x, y, z = SpatialCoordinate(mesh)
    z_norm = z / sqrt(x**2 + y**2 + z**2)
    z_norm = Min(Max(z_norm, -1.0), 1.0)  # avoid silly roundoff errors
    lat = asin(z_norm)
    lon = atan_2(y, x)
    return lon, lat


def vector_enu_to_xyz(mesh, uvw_enu_expr):
    """
    Convert vector from local tanget plane to Earth-centered Cartesian system.

    :arg x, y, z: spatial coordinates of the mesh
    :arg uvw_enu_expr: vectorin local East-North-Up (ENU) tangent plane
        coordinate system (on a spherical Earth).
    """
    x, y, z = SpatialCoordinate(mesh)
    epsilon = Constant(1e-3)
    r_h = sqrt(x**2 + y**2 + epsilon)
    # local tangent plane coordinate system unit vectors
    ne = as_vector((-y, x, 0)) * 1 / r_h  # east
    nn = as_vector((-x * z, -y * z, x**2 + y**2)) * 1 / r_h / r_earth  # north
    nu = as_vector((x, y, z)) / r_earth  # up
    # map vectors from local ENU coordinates to ECEF
    M = as_tensor((ne, nn, nu)).T
    uvw_expr = M * uvw_enu_expr
    return uvw_expr


def williamson2_init_fields(mesh, u_max, depth):
    """
    Initial elevation and velocity for Williamson 2 test case.
    """
    g = physical_constants['g_grav']
    x, y, z = SpatialCoordinate(mesh)
    uv_expr = as_vector([-u_max * y / r_earth, u_max * x / r_earth, 0.0])
    elev_expr = depth - \
        ((r_earth * omega * u_max + u_max**2 / 2.0) * z**2 / r_earth**2) / g
    return elev_expr, uv_expr


def setup_williamson2(mesh, time):
    """
    Williamson (1992) shallow water test case 2:
    Global steady state nonlinear zonal geostrophic flow
    """
    depth = 5960.
    u_max = 2 * pi * r_earth / (12 * 24 * 3600.)
    elev_expr, uv_expr = williamson2_init_fields(mesh, u_max, depth)
    bath_expr = Constant(depth)

    analytical_solution = True
    return elev_expr, uv_expr, bath_expr, analytical_solution


def setup_williamson5(mesh, time):
    """
    Williamson (1992) shallow water test case 5:
    Zonal flow over an isolated mountain
    """
    depth = 5960.
    u_max = 20.
    elev_expr, uv_expr_w2 = williamson2_init_fields(mesh, u_max, depth)

    lon, lat = coords_xyz_to_lonlat(mesh)
    R0 = pi / 9.
    lon_c = -pi / 2.
    lat_c = pi / 6.
    r = sqrt(Min(R0**2, (lon - lon_c)**2 + (lat - lat_c)**2))
    bath_expr = depth - 2000 * (1 - r / R0)

    # NOTE scale uv to fit the modified bathymetry to reduce initial shock
    # this is not in the original test case
    h_w2 = depth + elev_expr
    h_w5 = bath_expr + elev_expr
    uv_expr = uv_expr_w2 * h_w2 / h_w5

    analytical_solution = False
    return elev_expr, uv_expr, bath_expr, analytical_solution


def setup_lauter3(mesh, time):
    """
    Läuter (2005) shallow water test case, example 3:
    Unsteady solid body rotation
    """
    x, y, z = SpatialCoordinate(mesh)
    g = physical_constants['g_grav']
    # define initial state
    alpha = pi / 4.
    k1 = 133681.
    u_0 = 2 * pi * r_earth / (12 * 24 * 3600.)
    epsilon = Constant(1e-3)
    r_h = sqrt(x**2 + y**2 + epsilon)
    # velocity in East, North, Up tangent plane system
    xt = cos(omega * time)
    yt = sin(omega * time)
    u_enu_expr = u_0 / r_earth / r_h * (
        sin(alpha) * z * (x * xt - y * yt) + cos(alpha) * r_h**2
    )
    v_enu_expr = -u_0 * sin(alpha) / r_h * (y * xt + x * yt)
    uv_enu_expr = as_vector([u_enu_expr, v_enu_expr, 0.0])
    uv_expr = vector_enu_to_xyz(mesh, uv_enu_expr)

    orog_expr = (omega * z)**2 / g / 2
    b = (sin(alpha) * (-x * xt + y * yt) + cos(alpha) * z) / r_earth
    c = 12e3  # set constant elevation to bathymetry
    elev_expr = (-0.5 * (u_0 * b + omega * z)**2 + k1) / g + orog_expr - c
    bath_expr = - orog_expr + c

    analytical_solution = True
    return elev_expr, uv_expr, bath_expr, analytical_solution


def run(refinement, cell='triangle', setup=setup_williamson2, **model_options):
    print_output('--- running refinement {:}'.format(refinement))

    if cell == 'triangle':
        mesh2d = IcosahedralSphereMesh(
            radius=r_earth, refinement_level=refinement, degree=3)
    elif cell == 'quad':
        # NOTE cube sphere has lower resolution
        mesh2d = CubedSphereMesh(
            radius=r_earth, refinement_level=refinement + 1, degree=3)
    else:
        raise NotImplementedError(f'Unsupported cell type: {cell:}')

    mesh2d.init_cell_orientations(SpatialCoordinate(mesh2d))

    outputdir = 'outputs'
    t_end = 24 * 3600
    t_export = 4 * 3600.
    # NOTE dt must be relatively low as solution exhibits dt depended phase lag
    dt = 1200.

    time = Constant(0)
    elev_expr, uv_expr, bath_expr, ana_sol_exists = setup(mesh2d, time)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.project(bath_expr)

    # Coriolis forcing
    x, y, z = SpatialCoordinate(mesh2d)
    f_expr = 2 * omega * z / r_earth
    coriolis_2d = Function(P1_2d)
    coriolis_2d.interpolate(f_expr)

    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.element_family = 'bdm-dg'
    options.polynomial_degree = 1
    options.coriolis_frequency = coriolis_2d
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.swe_timestepper_type = 'CrankNicolson'
    options.timestep = dt
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(0.1)
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
    options.no_exports = True
    options.update(model_options)

    solver_obj.create_function_spaces()
    if not options.no_exports:
        # Store analytical elevation to disk
        out = File(outputdir + '/Elevation2d_ana/Elevation2d_ana.pvd')
        ana_elev = Function(solver_obj.function_spaces.H_2d, name='Elevation')

    def export():
        if not options.no_exports:
            time.assign(solver_obj.simulation_time)
            ana_elev.project(elev_expr)
            out.write(ana_elev)

    solver_obj.assign_initial_conditions(elev=elev_expr, uv=uv_expr)
    solver_obj.iterate(export_func=export)

    if ana_sol_exists:
        time.assign(solver_obj.simulation_time)
        area = 4 * pi * r_earth**2
        elev_err = errornorm(elev_expr, solver_obj.fields.elev_2d) / sqrt(area)
        uv_err = errornorm(uv_expr, solver_obj.fields.uv_2d) / sqrt(area)
        print_output(f'L2 error elev {elev_err:.12f}')
        print_output(f'L2 error uv {uv_err:.12f}')
        return elev_err, uv_err
    return None, None


def run_convergence(ref_list, saveplot=False, **options):
    """
    Runs test for a list of refinements and computes error convergence rate
    """
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    l2_err = numpy.log10(numpy.array(l2_err))
    elev_err = l2_err[:, 0]
    uv_err = l2_err[:, 1]
    delta_x = numpy.log10(0.5**numpy.array(ref_list))
    setup_name = options['setup'].__name__

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot):
        slope_rtol = 0.20
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(x_log, y_log)
        if saveplot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05 * (x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope * xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2 * int(npoints / 3)], yy[2 * int(npoints / 3)],
                    '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(' '.join([setup_name, field_str]))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str])
            imgfile += '.png'
            imgdir = create_directory('plots')
            imgfile = os.path.join(imgdir, imgfile)
            print_output('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = f'{setup_name:}: Wrong convergence rate ' \
                f'{slope:.4f}, expected {expected_slope:.4f}'
            assert slope > expected_slope * (1 - slope_rtol), err_msg
            print_output(
                f'{setup_name:}: {field_str:} convergence rate '
                f'{slope:.4f} PASSED'
            )
        else:
            print_output(
                f'{setup_name:}: {field_str:} convergence rate {slope:.4f}'
            )
        return slope

    check_convergence(delta_x, elev_err, 2, 'elevation', saveplot)
    check_convergence(delta_x, uv_err, 2, 'velocity', saveplot)


@pytest.fixture(params=[setup_williamson2, setup_lauter3],
                ids=['williamson2', 'lauter3'])
def setup(request):
    return request.param


@pytest.mark.parametrize(
    ('element_family', 'cell'),
    [
        ('rt-dg', 'triangle'),
        ('rt-dg', 'quad'),
        ('bdm-dg', 'triangle'),
        pytest.param(
            'bdm-dg', 'quad',
            marks=pytest.mark.xfail(reason='Firedrake does not currently support BDMCE element')),
    ]
)
def test_convergence(element_family, cell, setup):
    run_convergence([1, 2, 3], cell=cell, setup=setup,
                    element_family=element_family)


def test_convergence_explicit():
    run_convergence([1, 2, 3], cell='triangle', setup=setup_williamson2,
                    element_family='bdm-dg',
                    swe_timestepper_type='SSPRK33')


def test_williamson5():
    """
    Test that williamson5 case runs.
    """
    run(2, setup=setup_williamson5, cell='triangle', element_family='bdm-dg',
        timestep=3600., simulation_end_time=10 * 3600.,
        no_exports=True)


if __name__ == '__main__':
    run(4, setup=setup_williamson5, cell='triangle', element_family='bdm-dg',
        timestep=3 * 3600., simulation_end_time=24 * 24 * 3600.,
        simulation_export_time=3 * 3600.,
        no_exports=False)
