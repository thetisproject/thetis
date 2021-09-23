"""
Tracer box in 2D
================

Solves a standing wave in a rectangular basin using wave equation.

This version uses a constant tracer to check local/global conservation of tracers.

Initial condition for elevation corresponds to a standing wave.
Time step and export interval are chosen based on theoretical
oscillation frequency. Initial condition repeats every 20 exports.
"""
from thetis import *
import pytest


def run_tracer_consistency(constant_c=True, **model_options):

    t_cycle = 2000.0  # standing wave period
    depth = 50.0  # average depth
    lx = numpy.sqrt(9.81*depth)*t_cycle  # wave length
    ly = 3000.0
    nx = 18
    ny = 2
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    tracer_value = 4.5
    elev_amp = 2.0
    # estimate of max advective velocity used to estimate time step
    u_mag = Constant(1.0)

    outputdir = 'outputs'

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    x_2d, y_2d = SpatialCoordinate(mesh2d)
    # non-trivial bathymetry, to properly test 2d tracer conservation
    bathymetry_2d.interpolate(depth + depth/10.*sin(x_2d/lx*pi))

    # set time step, export interval and run duration
    n_steps = 8
    t_export = round(float(t_cycle/n_steps))
    # for testing tracer conservation, we don't want to come back to the initial condition
    t_end = 2.5*t_cycle

    # create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.use_limiter_for_tracers = not constant_c
    options.use_nonlinear_equations = True
    conservative = model_options.pop('use_tracer_conservative_form', False)
    options.add_tracer_2d('tracer_2d', 'Depth averaged tracer', 'Tracer2d',
                          use_conservative_form=conservative)
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = Constant(u_mag)
    options.check_volume_conservation_2d = True
    options.check_tracer_conservation = True
    options.check_tracer_overshoot = True
    options.set_timestepper_type(model_options.pop('timestepper_type', 'CrankNicolson'))
    options.output_directory = outputdir
    options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
    options.update(model_options)

    if not options.no_exports:
        print_output('Exporting to {:}'.format(options.output_directory))

    solver_obj.create_function_spaces()
    elev_init = Function(solver_obj.function_spaces.H_2d)
    elev_init.project(-elev_amp*cos(2*pi*x_2d/lx))

    tracer_init2d = None
    if constant_c:
        tracer_init2d = Function(solver_obj.function_spaces.Q_2d, name='initial tracer')
        tracer_init2d.assign(tracer_value)
    else:
        tracer_init2d = Function(solver_obj.function_spaces.Q_2d, name='initial tracer')
        tracer_l = 0
        tracer_r = 30.0
        tracer_init2d.interpolate(tracer_l + (tracer_r - tracer_l)*0.5*(1.0 + sign(x_2d - lx/4)))

    solver_obj.assign_initial_conditions(elev=elev_init, tracer=tracer_init2d)
    solver_obj.iterate()

    # TODO do these checks every export ...
    vol2d, vol2d_rerr = solver_obj.callbacks['export']['volume2d']()
    assert vol2d_rerr < 1e-10, '2D volume is not conserved'
    tracer_int, tracer_int_rerr = solver_obj.callbacks['export']['tracer_2d mass']()
    assert abs(tracer_int_rerr) < 1.2e-4, 'tracer is not conserved'
    smin, smax, undershoot, overshoot = solver_obj.callbacks['export']['tracer_2d overshoot']()
    max_abs_overshoot = max(abs(undershoot), abs(overshoot))
    overshoot_tol = 1e-11
    if not conservative:
        msg = 'Tracer overshoots are too large: {:}'.format(max_abs_overshoot)
        assert max_abs_overshoot < overshoot_tol, msg

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.fixture(params=['CrankNicolson', 'SSPRK33', 'ForwardEuler', 'BackwardEuler', 'DIRK22', 'DIRK33'])
def stepper(request):
    return request.param


def test_const_tracer(stepper):
    """
    Test timeintegrator without slope limiters
    Constant tracer, should remain constant
    """
    run_tracer_consistency(constant_c=True,
                           use_nonlinear_equations=True,
                           use_limiter_for_tracers=False,
                           no_exports=True,
                           timestepper_type=stepper)


def test_nonconst_tracer(stepper):
    """
    Test timeintegrator with slope limiters
    Non-trivial tracer, should see no overshoots and be conserved
    """
    run_tracer_consistency(constant_c=False,
                           use_nonlinear_equations=True,
                           use_limiter_for_tracers=True,
                           no_exports=True,
                           timestepper_type=stepper)


def test_nonconst_tracer_conservative(stepper):
    """
    Test timeintegrator without slope limiters
    Non-trivial tracer, should be conserved
    """
    run_tracer_consistency(constant_c=False,
                           use_nonlinear_equations=True,
                           use_limiter_for_tracers=False,
                           no_exports=True,
                           use_tracer_conservative_form=True,
                           timestepper_type=stepper)


# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_tracer_consistency(constant_c=False,
                           use_nonlinear_equations=True,
                           use_limiter_for_tracers=False,
                           use_tracer_conservative_form=True,
                           no_exports=False)
