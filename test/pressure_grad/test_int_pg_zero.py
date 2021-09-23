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
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
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
    options.timestepper_type = 'SSPRK22'
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
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = 100.
    options.no_exports = True
    options.check_temperature_overshoot = False
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'temp_3d', 'density_3d',
                                'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                                'smag_visc_3d', 'int_pg_3d',
                                'hcc_metric_3d']
    options.update(kwargs)
    if options.equation_of_state_type == 'linear':
        options.equation_of_state_options.rho_ref = rho_0
        options.equation_of_state_options.s_ref = salt_const
        options.equation_of_state_options.th_ref = t_ref
        options.equation_of_state_options.alpha = alpha
        options.equation_of_state_options.beta = beta

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

    int_pg_mag = numpy.abs(solver_obj.fields.int_pg_3d.dat.data).max()
    print_output('int pg error: {:9.2e}'.format(int_pg_mag))

    if iterate:
        solver_obj.iterate()

    uv_mag = numpy.abs(solver_obj.fields.uv_3d.dat.data).max()
    if iterate:
        print_output('uv error: {:9.2e}'.format(uv_mag))

    res = {}
    res['int_pg_3d'] = int_pg_mag
    res['uv_3d'] = uv_mag
    return res


options_dict = {
    'setup1': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state_type': 'full',
        'geometry': 'seamount',
    },
    'setup2': {
        'use_quadratic_pressure': False,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state_type': 'linear',
        'geometry': 'warped',
    },
    'setup3': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state_type': 'linear',
        'geometry': 'warped',
    },
    'setup4': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state_type': 'full',
        'geometry': 'warped',
    },
    'setup5': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': True,
        'equation_of_state_type': 'full',
        'geometry': 'warped',
    },
    'setup6': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': False,
        'equation_of_state_type': 'full',
        'geometry': 'warped',
    },
    'setup7': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state_type': 'full',
        'geometry': 'warped',
    },
    'setup8': {
        'use_quadratic_pressure': True,
        'use_quadratic_density': True,
        'lin_strat': False,
        'equation_of_state_type': 'full',
        'geometry': 'warped',
    },
}


@pytest.mark.parametrize('setup,element_family,target', [
    ('setup1', 'dg-dg', 2e-6,),
    ('setup2', 'dg-dg', 7e-4,),
    ('setup3', 'dg-dg', 1e-13,),
    ('setup4', 'dg-dg', 7e-6,),
    ('setup5', 'dg-dg', 1e-6,),
    ('setup6', 'dg-dg', 3e-5,),
    ('setup7', 'dg-dg', 1e-5,),
    ('setup1', 'rt-dg', 9.0,),
    ('setup2', 'rt-dg', 850.,),
    ('setup3', 'rt-dg', 1e-7,),
    ('setup4', 'rt-dg', 30.,),
    ('setup5', 'rt-dg', 9.,),
    ('setup6', 'rt-dg', 60.,),
    ('setup7', 'rt-dg', 40.,),
    ('setup1', 'bdm-dg', 9.0,),
    ('setup2', 'bdm-dg', 950.,),
    ('setup3', 'bdm-dg', 1e-7,),
    ('setup4', 'bdm-dg', 40.,),
    ('setup5', 'bdm-dg', 8.,),
    ('setup6', 'bdm-dg', 80.,),
    ('setup7', 'bdm-dg', 60.,),
],)
def test_int_pg(setup, element_family, target):
    """
    Initialize model and check magnitude of internal pressure gradient error

    Correct pressure gradient is zero.
    """
    options = options_dict[setup]
    options['element_family'] = element_family
    error = compute_pg_error(**options)
    assert error['int_pg_3d'] < target


@pytest.mark.parametrize('setup,element_family,target', [
    ('setup8', 'dg-dg', 0.005),
    ('setup3', 'dg-dg', 1e-12),
    ('setup8', 'rt-dg', 80e3),
    ('setup3', 'rt-dg', 1e-4),
    ('setup8', 'bdm-dg', 90e3),
    ('setup3', 'bdm-dg', 5e-4),
],)
def test_stability(setup, element_family, target):
    """
    Run model for a few time steps, check magnitude of the velocity field.
    """
    options = options_dict[setup]
    options['element_family'] = element_family
    options['iterate'] = True
    error = compute_pg_error(**options)
    assert error['uv_3d'] < target


if __name__ == '__main__':
    options = {
        'element_family': 'rt-dg',
        'use_quadratic_pressure': True,
        'use_quadratic_density': False,
        'lin_strat': True,
        'equation_of_state_type': 'linear',
        'geometry': 'warped',
        'target': 1e-4,
    }
    options['no_exports'] = False
    options['iterate'] = True
    test_stability(options)
