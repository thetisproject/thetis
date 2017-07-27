"""
Tests whether we can compute a consistent gradient of some functional
based on the forward model with respect to the bottom friction
via firedrake_adjoint.

Stephan Kramer 25-05-16
"""
import pytest
from thetis_adjoint import *
op2.init(log_level=INFO)

velocity_u = 2.0

def basic_setup():
    lx = 100.0
    ly = 50.0
    nx = 20.0
    ny = 10.0
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # export interval in seconds
    t_export = 0.5
    timestep = 0.5

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')

    depth = 50.0
    bathymetry_2d.assign(depth)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.timestepper_type = 'CrankNicolson'
    options.timestep = timestep
    options.horizontal_viscosity = Constant(2.0)

    # create function spaces
    solver_obj.create_function_spaces()

    # create drag function and set it with a bump function representing a turbine
    drag_func = Function(solver_obj.function_spaces.P1_2d, name='bottomdrag')
    x = SpatialCoordinate(mesh2d)
    drag_center = 12.0
    drag_bg = 0.0025
    x0 = lx/2
    y0 = ly/2
    sigma = 20.0
    drag_func.project(drag_center*exp(-((x[0]-x0)**2 + (x[1]-y0)**2)/sigma**2) + drag_bg, annotate=False)
    # assign fiction field
    options.quadratic_drag_coefficient = drag_func

    # assign boundary conditions
    inflow_tag = 1
    outflow_tag = 2
    inflow_bc = {'un': Constant(-velocity_u)}  # NOTE negative into domain
    outflow_bc = {'elev': Constant(0.0)}

    solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc,
                                                 outflow_tag: outflow_bc}
    return solver_obj


def setup_steady():
    solver_obj = basic_setup()
    solver_obj.options.timestepper_type = 'SteadyState'
    solver_obj.options.simulation_end_time = 0.499
    solver_obj.options.timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': True,
        'snes_type': 'newtonls',
    }
    solver_obj.create_equations()
    return solver_obj


def setup_unsteady():
    solver_obj = basic_setup()
    solver_obj.options.timestepper_type = 'CrankNicolson'
    solver_obj.options.simulation_end_time = 2.0
    solver_obj.options.timestepper_options.implicitness_theta = 1.0
    solver_obj.options.timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': True,
        'snes_type': 'newtonls',
    }
    solver_obj.create_equations()
    return solver_obj


@pytest.fixture(params=[setup_steady, setup_unsteady])
def setup(request):
    return request.param


def test_gradient_from_adjoint(setup):
    solver_obj = setup()
    solver_obj.assign_initial_conditions(uv=as_vector((velocity_u, 0.0)), elev=Constant(0.0))
    solver_obj.iterate()
    J0 = assemble(solver_obj.fields.solution_2d[0]*dx)

    drag_func = solver_obj.options.quadratic_drag_coefficient
    Jhat = ReducedFunctional(J0, drag_func)

    c = Function(drag_func)
    dc = Function(c)
    from numpy.random import rand
    c.vector()[:] = rand(*c.dat.shape)
    dc.vector()[:] = rand(*dc.dat.shape)
    minconv = taylor_test(Jhat, c, dc)
    assert minconv > 1.90
