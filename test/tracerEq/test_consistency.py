"""
Tracer box in 3D
================

Solves a standing wave in a rectangular basin using wave equation.

This version uses a constant tracer to check local/global conservation of tracers.

Initial condition for elevation corresponds to a standing wave.
Time step and export interval are chosen based on theoretical
oscillation frequency. Initial condition repeats every 20 exports.
"""
from thetis import *
import pytest


def run_tracer_consistency(**model_options):
    meshtype = model_options.pop('meshtype')

    t_cycle = 2000.0  # standing wave period
    depth = 50.0
    lx = numpy.sqrt(9.81*depth)*t_cycle  # wave length
    ly = 3000.0
    nx = 18
    ny = 2
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    salt_value = 4.5
    n_layers = 6
    elev_amp = 2.0
    # estimate of max advective velocity used to estimate time step
    u_mag = Constant(1.0)

    sloped = False
    warped = False
    if meshtype == 'sloped':
        sloped = True
    elif meshtype == 'warped':
        warped = True

    suffix = ''
    if sloped:
        suffix = '_sloped'
    if warped:
        suffix = '_warped'
    outputdir = 'outputs' + suffix

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)
    x_2d, y_2d = SpatialCoordinate(mesh2d)
    if sloped:
        bathymetry_2d.interpolate(depth + 20.0*x_2d/lx)

    # set time step, export interval and run duration
    n_steps = 8
    t_export = round(float(t_cycle/n_steps))
    t_end = 2*t_cycle

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)

    if warped:
        # warp interior mesh, top/bottom surfaces are horizontal
        coords = solver_obj.mesh.coordinates
        z = coords.dat.data[:, 2].copy()
        x = coords.dat.data[:, 0]
        p = 2.5*x/lx + 0.5
        sigma = -depth * (0.5*numpy.tanh(p*(-2.0*z/depth - 1.0))/numpy.tanh(p) + 0.5)
        coords.dat.data[:, 2] = sigma

    options = solver_obj.options
    options.use_nonlinear_equations = True
    options.solve_salinity = True
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = False
    options.use_limiter_for_tracers = False
    options.use_lax_friedrichs_tracer = False
    options.use_lax_friedrichs_velocity = False
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = Constant(u_mag)
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.check_salinity_conservation = True
    options.check_salinity_overshoot = True
    options.check_temperature_conservation = True
    options.check_temperature_overshoot = True
    options.output_directory = outputdir
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                                'uv_dav_2d']
    options.update(model_options)
    if not options.no_exports:
        print_output('Exporting to {:}'.format(options.output_directory))

    solver_obj.create_function_spaces()
    elev_init = Function(solver_obj.function_spaces.H_2d)
    elev_init.project(-elev_amp*cos(2*pi*x_2d/lx))

    salt_init3d = None
    temp_init3d = None
    if options.solve_salinity:
        salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
        salt_init3d.assign(salt_value)
    if options.solve_temperature:
        temp_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
        x, y, z = SpatialCoordinate(solver_obj.mesh)
        temp_l = 0
        temp_r = 30.0
        temp_init3d.interpolate(temp_l + (temp_r - temp_l)*0.5*(1.0 + sign(x - lx/2)))

    solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, temp=temp_init3d)
    solver_obj.iterate()

    # TODO do these checks every export ...
    vol2d, vol2d_rerr = solver_obj.callbacks['export']['volume2d']()
    assert vol2d_rerr < 1e-9 if options.element_family == 'rt-dg' else 1e-10, '2D volume is not conserved'
    if options.use_ale_moving_mesh:
        vol3d, vol3d_rerr = solver_obj.callbacks['export']['volume3d']()
        assert vol3d_rerr < 1e-10, '3D volume is not conserved'
    if options.solve_salinity:
        salt_int, salt_int_rerr = solver_obj.callbacks['export']['salt_3d mass']()
        assert salt_int_rerr < 1e-9, 'salt is not conserved'
        smin, smax, undershoot, overshoot = solver_obj.callbacks['export']['salt_3d overshoot']()
        max_abs_overshoot = max(abs(undershoot), abs(overshoot))
        overshoot_tol = 1e-10 if warped else 1e-12
        if options.use_ale_moving_mesh:
            overshoot_tol = 1e-6
        msg = 'Salt overshoots are too large: {:}'.format(max_abs_overshoot)
        assert max_abs_overshoot < overshoot_tol, msg
    if options.solve_temperature:
        temp_int, temp_int_rerr = solver_obj.callbacks['export']['temp_3d mass']()
        mass_tol = 1e-4 if options.use_ale_moving_mesh else 1e-3
        assert temp_int_rerr < mass_tol, 'temp is not conserved'
        smin, smax, undershoot, overshoot = solver_obj.callbacks['export']['temp_3d overshoot']()
        max_abs_overshoot = max(abs(undershoot), abs(overshoot))
        overshoot_tol = 1e-11 if warped else 1e-12
        if options.use_ale_moving_mesh:
            overshoot_tol = 1e-6
        msg = 'Temp overshoots are too large: {:}'.format(max_abs_overshoot)
        assert max_abs_overshoot < overshoot_tol, msg


@pytest.mark.parametrize('element_family', ['dg-dg', 'rt-dg', 'bdm-dg'])
@pytest.mark.parametrize('meshtype', ['regular', 'sloped', 'warped'])
@pytest.mark.parametrize('timestepper_type', ['LeapFrog', 'SSPRK22'])
def test_ale_const_tracer(element_family, meshtype, timestepper_type):
    """
    Test ALE timeintegrators without slope limiters
    One constant tracer, should remain constants
    """
    run_tracer_consistency(element_family=element_family,
                           meshtype=meshtype,
                           use_ale_moving_mesh=True,
                           solve_salinity=True,
                           solve_temperature=False,
                           use_limiter_for_tracers=False,
                           timestepper_type=timestepper_type,
                           no_exports=True)


@pytest.mark.parametrize('element_family', ['dg-dg', 'rt-dg', 'bdm-dg'])
@pytest.mark.parametrize('meshtype', ['regular', 'sloped', 'warped'])
@pytest.mark.parametrize('timestepper_type', ['LeapFrog', 'SSPRK22'])
def test_ale_nonconst_tracer(element_family, meshtype, timestepper_type):
    """
    Test ALE timeintegrators with slope limiters
    One constant and one non-trivial tracer, should see no overshoots
    """
    run_tracer_consistency(element_family=element_family,
                           meshtype=meshtype,
                           use_ale_moving_mesh=True,
                           solve_salinity=True,
                           solve_temperature=True,
                           use_limiter_for_tracers=True,
                           timestepper_type=timestepper_type,
                           no_exports=True)


if __name__ == '__main__':
    run_tracer_consistency(element_family='dg-dg',
                           meshtype='regular',
                           use_nonlinear_equations=True,
                           timestepper_type='LeapFrog',
                           use_ale_moving_mesh=True,
                           solve_salinity=True,
                           solve_temperature=True,
                           use_limiter_for_tracers=True,
                           no_exports=False)
