"""
Wind-driver entrainment test case.

Based on Kato-Phillips laboratory tests.

Initial water column is stratified with a constant density gradient.
Circulation is driven by constant wind stress at the surface. Wind-induced
mixing begins to destroy stratification, creating a well-mixed surfacelayer.
The depth of the surface layer follows an empirical relation versus time.

Surface friction velocity is u_s = 0.01 m s-1.
Initially buoyancy frequency is constant N = 0.01 s-1.


[1] http://www.gotm.net/
[2] Karna et al. (2012). Coupling of a discontinuous Galerkin finite element
    marine model with a finite difference turbulence closure model.
    Ocean Modelling, 47:55-64.
    http://dx.doi.org/10.1016/j.ocemod.2012.01.001
"""
from thetis import *
import pytest


def run_katophillips(**model_options):
    physical_constants['rho0'] = 1027.0  # NOTE must match empirical setup

    outputdir = 'outputs'
    # set mesh resolution
    dx = 2500.0
    layers = 20
    depth = 50.0

    # generate unit mesh and transform its coords
    nx = 3  # nb elements in flow direction
    lx = nx*dx
    ny = 2  # nb elements in cross direction
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x', reorder=True)
    # move mesh, center to (0,0)
    mesh2d.coordinates.dat.data[:, 0] -= lx/2
    mesh2d.coordinates.dat.data[:, 1] -= ly/2

    dt = 60.0
    t_end = 4 * 3600.0
    t_export = 5*60.0
    u_mag = 1.0

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry2d = Function(p1_2d, name='Bathymetry')
    bathymetry2d.assign(depth)

    wind_stress_x = 0.1027  # Pa
    wind_stress_2d = Constant((wind_stress_x, 0))

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.use_nonlinear_equations = False
    options.solve_salinity = True
    options.solve_temperature = False
    options.constant_temperature = Constant(10.0)
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = False
    options.use_turbulence = True
    options.use_ale_moving_mesh = False
    options.use_baroclinic_formulation = True
    options.use_limiter_for_tracers = False
    options.vertical_viscosity = Constant(1.3e-6)  # background value
    options.vertical_diffusivity = Constant(1.4e-7)  # background value
    options.wind_stress = wind_stress_2d
    options.no_exports = True
    options.simulation_export_time = t_export
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = dt
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(u_mag)
    options.check_salinity_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d',
                                'baroc_head_3d', 'uv_dav_2d', 'eddy_visc_3d',
                                'shear_freq_3d', 'buoy_freq_3d',
                                'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
    options.update(model_options)
    if options.timestepper_type in ['LeapFrog', 'SSPRK22']:
        options.use_ale_moving_mesh = True
    if hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestepper_options.use_automatic_timestep = False

    solver_obj.create_function_spaces()

    # initial conditions
    buoyfreq0 = 0.01
    rho_grad = -buoyfreq0**2 * physical_constants['rho0'] / physical_constants['g_grav']
    beta = 0.7865  # haline contraction coefficient [kg m-3 psu-1]
    salt_grad = rho_grad/beta
    salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
    x, y, z = SpatialCoordinate(solver_obj.mesh)
    salt_init_expr = salt_grad*z
    salt_init3d.interpolate(salt_init_expr)

    solver_obj.assign_initial_conditions(salt=salt_init3d)

    solver_obj.iterate()

    # check mixed layer depth
    npoints = layers*4
    tol = 1e-4
    z = numpy.linspace(-tol, -depth + tol, npoints)
    x = numpy.zeros_like(z)
    y = numpy.zeros_like(z)
    xyz = numpy.vstack((x, y, z)).T
    tke_arr = numpy.array(solver_obj.fields.tke_3d.at(tuple(xyz)))
    # mixed layer depth: lowest point where tke > tol
    tke_tol = 1e-5
    ix = tke_arr > tke_tol
    if ix.any():
        ml_depth = -z[ix].min()
    else:
        ml_depth = 0.0
    # correct depth
    u_s = 0.01
    buoyfreq0 = 0.01
    target = 1.05*u_s*numpy.sqrt(solver_obj.simulation_time/buoyfreq0)
    rtol = 0.05
    rel_err = (ml_depth - target)/target
    assert rel_err > -rtol, 'mixed layer is too shallow: {:} < {:}'.format(ml_depth, target)
    assert rel_err < rtol, 'mixed layer is too deep: {:} > {:}'.format(ml_depth, target)
    print_output('Mixed layer depth: {:.2f} (target: {:.2f}) PASSED'.format(ml_depth, target))


@pytest.fixture(params=['rt-dg', 'dg-dg', 'bdm-dg'])
def element_family(request):
    return request.param


@pytest.fixture(params=['LeapFrog', 'SSPRK22'])
def timestepper_type(request):
    return request.param


def test_katophillips(element_family, timestepper_type):
    run_katophillips(element_family=element_family,
                     timestepper_type=timestepper_type)


if __name__ == '__main__':
    run_katophillips(element_family='dg-dg',
                     timestepper_type='SSPRK22',
                     no_exports=False)
