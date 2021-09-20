"""
Bottom Ekman layer test
=======================

Steady state flow in a channel subject to bottom friction and rotation.
Vertical viscosity is assumed to be constant to allow simple analytical
solution.
"""
from thetis import *
import pytest


def run_test(layers=25, tolerance=0.05, verify=True, **model_options):
    depth = 20.0

    # set mesh resolution
    dx = 2500.0
    nx = 3
    lx = nx*dx
    ny = 3
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='both',
                                   reorder=True)

    dt = 90.0
    t_end = 4 * 3600.0  # sufficient to reach ~steady state
    t_export = 450.0
    u_mag = 1.0

    f_coriolis = 1e-4
    nu_v = 5e-4

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
    options.no_exports = True
    options.update(model_options)

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

    # initialize with a linear v profile to speed-up convergence
    v_init_expr = conditional(z > -d, a*(1 + z_s), 0)
    solver_obj.assign_initial_conditions(uv_3d=as_vector((v_init_expr/3, -v_init_expr, 0)))

    solver_obj.iterate()

    if verify:
        uv_p1_dg = Function(solver_obj.function_spaces.P1DGv, name='velocity p1dg')
        uv_p1_dg.project(solver_obj.fields.uv_3d + solver_obj.fields.uv_dav_3d)
        volume = lx*ly*depth
        uv_l2_err = errornorm(uv_ana_expr, uv_p1_dg)/numpy.sqrt(volume)
        assert uv_l2_err < tolerance, 'L2 error is too large: {:} > {:}'.format(uv_l2_err, tolerance)
        print_output('L2 error {:.4f} PASSED'.format(uv_l2_err))

    return solver_obj


@pytest.fixture(params=['dg-dg', 'rt-dg', 'bdm-dg'])
def element_family(request):
    return request.param


@pytest.fixture(params=['LeapFrog', 'SSPRK22'])
def timestepper_type(request):
    return request.param


@pytest.mark.parametrize("nlayers,max_err",
                         [(25, 0.035), (5, 0.065)],
                         ids=['nz25', 'nz5'])
def test_bottom_friction(nlayers, max_err, element_family, timestepper_type):
    run_test(nlayers, tolerance=max_err, verify=True,
             element_family=element_family, timestepper_type=timestepper_type)
