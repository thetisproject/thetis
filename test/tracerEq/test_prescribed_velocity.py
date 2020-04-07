"""
Simple 2D tracer advection problem in a doubly periodic domain with prescribed fluid velocities.
"""
from thetis import *
import pytest


def run(refinement_level, velocity_type, **model_options):
    print_output("--- running refinement level {:d} with {:s} prescribed velocity".format(refinement_level, velocity_type))

    # Set up domain
    n, L = 10*refinement_level, 10.0
    mesh2d = PeriodicRectangleMesh(n, n, L, L)
    x, y = SpatialCoordinate(mesh2d)
    T = 10.0

    # Physics
    P1_2d = FunctionSpace(mesh2d, "CG", 1)
    bathymetry2d = Function(P1_2d).assign(1.0)

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
    options = solver_obj.options
    options.timestep = 0.5/refinement_level
    options.simulation_export_time = T/10
    options.simulation_end_time = T
    options.fields_to_export_hdf5 = []
    options.no_exports = True
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = None
    options.update(model_options)
    solver_obj.create_function_spaces()

    # Prescribed velocity
    if velocity_type == 'constant':
        update_forcings = None
    elif velocity_type == 'sinusoidal':
        def update_forcings(t):
            solver_obj.fields.uv_2d.interpolate(as_vector([L/T, cos(pi*t/5.0)]))
    else:
        raise ValueError("Velocity type '{:s}' not recognised.".format(velocity_type))

    # Apply initial conditions
    init = project(exp(-((x-L/2)**2 + (y-L/2)**2)), solver_obj.function_spaces.Q_2d)
    fluid_velocity = interpolate(as_vector([L/T, 0.0]), solver_obj.function_spaces.U_2d)
    solver_obj.assign_initial_conditions(uv=fluid_velocity, tracer=init)
    init = solver_obj.fields.tracer_2d.copy(deepcopy=True)

    # Solve and check advection cycle is complete
    solver_obj.iterate(update_forcings=update_forcings)
    error = errornorm(init, solver_obj.fields.tracer_2d)
    print_output("Relative L2 error: {:.2f}%".format(100*error/norm(init)))
    return error


def run_convergence(ref_list, velocity_type, **options):
    """Runs test for a list of refinements and computes error convergence rate."""
    setup_name = 'prescribed-velocity'
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, velocity_type, **options))

    def check_convergence(ref_list, l2_err, field_str):
        slope_rtol = 0.2
        setup_name = 'prescribed-velocity'

        # Check convergence of L2 errors
        for i in range(1, len(l2_err)):
            slope = l2_err[i-1]/l2_err[i]
            expected_slope = ref_list[i]/ref_list[i-1]
            err_msg = '{:s}: Wrong convergence rate {:.4f}, expected {:.4f}'
            assert slope > expected_slope*(1 - slope_rtol), err_msg.format(setup_name, slope, expected_slope)
            print_output('{:s}: {:s} convergence rate {:.4f}'.format(setup_name, field_str, slope))

        # Check magnitude of L2 errors
        for i in range(len(l2_err)):
            msg = "{:s}: L2 error {:.4e} does not match recorded value, expected < 0.7"
            assert l2_err[i] < 0.7, msg.format(setup_name, l2_err[i])
            print_output("{:s}: L2 error magnitude index {:d} PASSED".format(setup_name, i))

    check_convergence(ref_list, l2_err, 'tracer')


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['CrankNicolson'])  # TODO: Implement other time integration methods
def stepper(request):
    return request.param


@pytest.fixture(params=['constant', 'sinusoidal'])
def velocity_type(request):
    return request.param


def test_periodic(stepper, velocity_type):
    run_convergence([1, 2, 4], velocity_type, timestepper_type=stepper)


# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == "__main__":
    options = {
        'no_exports': False,
        'fields_to_export': ['tracer_2d'],
    }
    run_convergence([1, 2, 4], 'sinusoidal', **options)
