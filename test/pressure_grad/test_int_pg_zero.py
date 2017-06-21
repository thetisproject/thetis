"""
Compute internal pressure gradient error in static test case
"""
from thetis import *
import pytest


def compute_pg_error(**kwargs):
    # test specific params
    kwargs.setdefault('lin_strat', True)
    kwargs.setdefault('geometry', 'warped')
    kwargs.setdefault('iterate', False)
    lin_strat = kwargs.pop('lin_strat')
    geometry = kwargs.pop('geometry')
    iterate = kwargs.pop('iterate')

    rho_0 = 1000.0
    physical_constants['rho0'] = rho_0

    delta_x = 25e3
    lx = 300e3
    ly = 600e3
    nx = int(lx/delta_x)
    ny = int(ly/delta_x)

    mesh2d = RectangleMesh(nx, ny, lx, ly)
    layers = 8

    # density setup
    delta_rho = 50.0   # density anomaly [kg/m3]
    temp_lim = [10.0, 20.0]
    alpha = delta_rho/(temp_lim[1] - temp_lim[0])  # thermal expansion coeff
    beta = 0.0  # haline contraction coeff
    t_ref = temp_lim[1]
    salt_const = 33.0

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')

    xy = SpatialCoordinate(mesh2d)
    depth_lim = [3600., 600.]
    if geometry == 'warped':
        # like DOME test case bathymetry, except transitions into nonlinear slope
        y_slope = [300e3, 600e3]
        lin_bath_expr = (depth_lim[1] - depth_lim[0])/(y_slope[1] - y_slope[0])*(xy[1] - y_slope[0]) + depth_lim[0]
        tanh_bath_expr = 0.5*(depth_lim[1] + depth_lim[0])*(1 - 0.6*tanh(4*(xy[1]-ly/2)/ly))
        blend = 0.5*(1 - tanh(10*(xy[0]-lx/2)/lx))
        bathymetry_2d.interpolate(blend*lin_bath_expr + (1-blend)*tanh_bath_expr)
        bathymetry_2d.dat.data[bathymetry_2d.dat.data > depth_lim[0]] = depth_lim[0]
        bathymetry_2d.dat.data[bathymetry_2d.dat.data < depth_lim[1]] = depth_lim[1]
    elif geometry == 'seamount':
        # "easy" sea mount, inspired by the set up in Ezer at al. 2002
        mount_A = 0.14
        mount_L = 50e3
        bath_expr = depth_lim[0]*(1 - mount_A*exp(-((xy[0]-lx/2)**2 + (xy[1]-ly/2)**2)/mount_L**2))
        bathymetry_2d.interpolate(bath_expr)
    else:
        raise Exception('unsupported geometry option {:}'.format(geometry))

    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.timestepper_type = 'ssprk22'
    options.solve_salinity = False
    options.solve_temperature = True
    options.constant_salinity = Constant(salt_const)
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = True
    options.use_baroclinic_formulation = True
    options.use_lax_friedrichs_velocity = False
    options.use_lax_friedrichs_tracer = False
    options.coriolis_frequency = None
    options.use_limiter_for_tracers = False
    options.vertical_viscosity = None
    options.horizontal_viscosity = None
    options.horizontal_diffusivity = None
    options.simulation_export_time = 900.
    options.simulation_end_time = 4*900.
    options.timestep = 100.
    options.no_exports = True
    options.check_temperature_overshoot = False
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'temp_3d', 'density_3d',
                                'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                                'smag_visc_3d', 'int_pg_3d',
                                'hcc_metric_3d']
    options.linear_equation_of_state_parameters = {
        'rho_ref': rho_0,
        's_ref': salt_const,
        'th_ref': t_ref,
        'alpha': alpha,
        'beta': beta,
    }
    options.update(kwargs)

    solver_obj.create_function_spaces()

    xyz = SpatialCoordinate(solver_obj.mesh)

    temp_init_3d = Function(solver_obj.function_spaces.H, name='init temperature')
    if lin_strat:
        # linear expression
        temp_expr = (temp_lim[1] - temp_lim[0])*(depth_lim[0] + xyz[2])/depth_lim[0] + temp_lim[0]
    else:
        # exponential expression
        temp_expr = temp_lim[0] + (temp_lim[1] - temp_lim[0])*((1 - exp((xyz[2]+depth_lim[0])/depth_lim[0]))/(1 - exp(1.0)))
    temp_init_3d.interpolate(temp_expr)

    solver_obj.create_equations()

    solver_obj.assign_initial_conditions(temp=temp_init_3d)

    hcc_obj = Mesh3DConsistencyCalculator(solver_obj)
    hcc_obj.solve()

    int_pg_mag = np.abs(solver_obj.fields.int_pg_3d.dat.data).max()
    print_output('int pg error: {:9.2e}'.format(int_pg_mag))

    if iterate:
        solver_obj.iterate()

    uv_mag = np.abs(solver_obj.fields.uv_3d.dat.data).max()
    if iterate:
        print_output('uv error: {:9.2e}'.format(uv_mag))

    res = {}
    res['int_pg_3d'] = int_pg_mag
    res['uv_3d'] = uv_mag
    return res


@pytest.fixture(params=[
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'seamount',
        'target': 2e-6,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': False,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 7e-4,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 1e-13,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 7e-6,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': True,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 1e-6,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 3e-5,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 1e-5,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'seamount',
        'target': 9.0,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': False,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 850.,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 1e-7,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 30.,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': True,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 9.,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 60.,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 40.,
    },
],)
def pg_test_setup(request):
    return request.param


def test_int_pg(pg_test_setup):
    """
    Initialize model and check magnitude of internal pressure gradient error

    Correct pressure gradient is zero.
    """
    target = pg_test_setup.pop('target')
    error = compute_pg_error(**pg_test_setup)
    assert error['int_pg_3d'] < target


@pytest.fixture(params=[
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 0.02,
    },
    {
        'element_family': 'dg-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 1e-11,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state': 'full',
        'geometry': 'warped',
        'target': 80e3,
    },
    {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 5e-4,
    },
],)
def stability_setup(request):
    return request.param


def test_stability(stability_setup):
    """
    Run model for a few time steps, check magnitude of the velocity field.
    """
    target = stability_setup.pop('target')
    stability_setup['iterate'] = True
    error = compute_pg_error(**stability_setup)
    assert error['uv_3d'] < target


if __name__ == '__main__':
    options = {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state': 'linear',
        'geometry': 'warped',
        'target': 1e-4,
    }
    options['no_exports'] = False
    options['iterate'] = True
    test_stability(options)
