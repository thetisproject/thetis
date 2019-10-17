from thetis import *
import pytest
import numpy as np


def run(**model_options):

    # Domain
    lx = 3e3
    ly = 1e3
    nx = 30
    ny = 12
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    depth = 40.0

    # Time interval
    dt = 10.0
    t_end = 3000.0
    t_export = t_end/20.0

    # Bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathy_2d = Function(P1_2d, name='Bathymetry')
    bathy_2d.assign(depth)

    # Diffusivity
    nu = Constant(1e-3, domain=mesh2d)

    # Solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathy_2d)
    options = solver_obj.options
    options.output_directory = 'outputs'
    #options.no_exports = True
    options.timestep = dt
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    solver_obj.create_function_spaces()
    uv_tracer = Function(solver_obj.function_spaces.U_2d, name='uv tracer')
    uv_tracer.interpolate(as_vector([5.0, 0.0]))
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = Constant(1e-3)
    options.use_limiter_for_tracers = True
    options.fields_to_export = ['tracer_2d']
    options.update(model_options)

    # Boundary conditions
    bnd_influx = {'diff_flux': 0.5*nu}
    bnd_outflow = {'outflow': None}  # NOTE: The key used here is arbitrary
    solver_obj.bnd_functions['tracer'] = {1: bnd_influx, 2: bnd_outflow}
    # NOTE: Zero diff_flux boundaries are enforced by default on 3 and 4

    # Run model
    solver_obj.assign_initial_conditions(uv=uv_tracer)
    solver_obj.iterate()
    sol = solver_obj.fields.tracer_2d

    # Check boundary conditions are satisfied
    tol = 1e-5
    n = FacetNormal(mesh2d)
    diff_tensor = as_matrix([[nu, 0, ],
                             [0, nu, ]])
    diff_flux = dot(diff_tensor, grad(sol))
    msg = "Inflow boundary not satisfied: {:.4e}"
    inflow = assemble(dot(diff_flux, n)*ds(1))
    assert inflow > tol, msg.format(inflow)
    inflow_exact = assemble(bnd_influx['diff_flux']*ds(1))
    msg = "Outflow boundary not satisfied: {:.4e}"
    outflow = np.abs(assemble(dot(diff_flux, n)*ds(2)))
    assert outflow > tol, msg.format(outflow)
    msg = "Zero diff_flux boundary not satisfied: {:.4e}"
    north_wall = np.abs(assemble(dot(diff_flux, n)*ds(3)))
    assert north_wall < tol, msg.format(north_wall)
    south_wall = np.abs(assemble(dot(diff_flux, n)*ds(4)))
    assert south_wall < tol, msg.format(south_wall)


@pytest.fixture(params=[1, 2])
def polynomial_degree(request):
    return request.param


@pytest.mark.parametrize(('stepper'),
                         [('CrankNicolson')])

def test_horizontal_advection(polynomial_degree, stepper):
    run(polynomial_degree=polynomial_degree,
        timestepper_type=stepper)


if __name__ == '__main__':
    run(polynomial_degree=1,
        timestepper_type='CrankNicolson')
