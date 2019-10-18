from thetis import *
import pytest
import numpy as np


def fourier_series_solution(mesh, time):
    """
    We have a diffusion problem with inhomogeneous Neumann conditions and zero initial condition.
    In order to solve it analytically, we decompose it into two diffusion problems:
     - A diffusion problem with homogeneous Neumann conditions and a nonzero initial condition;
     - A diffusion problem with homogeneous Neumann conditions and a nonzero source term.
    Solving the inhomogeneous problem amounts to summing the solutions of the homogeneous problems.
    """
    lx = 10
    x, y = SpatialCoordinate(mesh)
    nu = 0.1
    diff_flux = 0.2
    dt = 0.02
    P1 = FunctionSpace(mesh, 'CG', 1)
    ic = Function(P1).interpolate(diff_flux*0.5*(lx - x)*(lx - x)/lx)
    source = Constant(-diff_flux/lx, domain=mesh)

    # The solution uses truncated Fourier expansions, meaning we need the following...

    def phi(n):
        return cos(n*pi*x/lx)

    def ic_fourier_coeff(n):
        return assemble(2/lx*ic*phi(n)*dx)

    def source_fourier_coeff(n):
        return assemble(2/lx*source*phi(n)*dx)

    def source_term(n, time):
        I = 0
        tau = 0
        while tau < time - 0.5*dt:
            I += exp(-nu*(n*pi/lx)**2*(t-tau))
            tau += dt
        I *= source_fourier_coeff(n)
        return I*phi(n)

    def ic_term(n, time):
        return ic_fourier_coeff(n)*exp(-nu*(n*pi/lx)**2*time)*phi(n)

    # Assemble truncated Fourier expansion
    sol = Function(P1, name='Fourier expansion')
    num_terms_source = 1  # Only one needed since source is constant
    num_terms_ic = 100
    expr = Constant(0.5*source_fourier_coeff(0)*time)
    expr = expr + Constant(0.5*ic_fourier_coeff(0))
    for k in range(1, num_terms_source):
        expr = expr + source_term(k, time)
    for k in range(1, num_terms_ic):
        expr = expr + ic_term(k, time)
    expr -= ic
    sol.interpolate(-expr)

    return sol


def run(refinement, **model_options):

    # Domain
    lx = 10
    ly = 1
    nx = 50*refinement
    ny = 4
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    depth = 40.0

    # Time interval
    dt = 0.01/refinement
    t_end = 1.0
    t_export = 0.1

    # Bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathy_2d = Function(P1_2d, name='Bathymetry')
    bathy_2d.assign(depth)

    # Diffusivity
    nu = Constant(0.1)

    # Solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathy_2d)
    options = solver_obj.options
    options.output_directory = 'outputs'
    #options.no_exports = True
    options.timestep = dt
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = nu
    options.use_limiter_for_tracers = True
    options.fields_to_export = ['tracer_2d']
    options.update(model_options)

    # Boundary conditions
    solver_obj.bnd_functions['tracer'] = {1: {'diff_flux': 0.2*nu}}
    # NOTE: Zero diff_flux boundaries are enforced elsewhere by default

    # Run model
    solver_obj.assign_initial_conditions()
    solver_obj.iterate()
    sol = solver_obj.fields.tracer_2d

    # Get truncated Fourier series solution
    fsol = fourier_series_solution(mesh2d, t_end)

    File('outputs/compare{:d}.pvd'.format(refinement)).write(sol, fsol)
    return errornorm(sol, fsol))


def run_convergence(**model_options):
    errors = []
    for refinement in (1, 2, 4):
        errors.append(run(refinement, **model_options))
    print(errors)
    # FIXME: why does the Fourier series solution not decay to zero?
    # TODO: convergence test


@pytest.fixture(params=[1, 2])
def polynomial_degree(request):
    return request.param


@pytest.mark.parametrize(('stepper'),
                         [('CrankNicolson')])

def test_horizontal_advection(polynomial_degree, stepper):
    run(polynomial_degree=polynomial_degree,
        timestepper_type=stepper)


if __name__ == '__main__':
    run_convergence(polynomial_degree=1,
                    timestepper_type='CrankNicolson')
