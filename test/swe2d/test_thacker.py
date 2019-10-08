# Test for wetting-drying scheme functionality, using Thacker test case
# Test case details in Gourgue et al (2009)

from thetis import *
import pytest


@pytest.mark.parametrize("stepper,n,dt,alpha,max_err",
                         [
                             ('BackwardEuler', 10, 600., 2., 0.22),
                             ('BackwardEuler', 25, 300., 2., 0.14),
                             ('CrankNicolson', 10, 600., 2., 0.07),
                             ('CrankNicolson', 25, 300., 2., 0.007),
                             ('DIRK22', 10, 600., 2., 0.025),
                             ('DIRK22', 25, 300., 2., 0.006),
                             ('DIRK33', 10, 600., 2., 0.07),
                             ('DIRK33', 25, 300., 2., 0.007),
                         ],
                         ids=[
                             'BackwardEuler-coarse',
                             'BackwardEuler-fine',
                             'CrankNicolson-coarse',
                             'CrankNicolson-fine',
                             'DIRK22-coarse',
                             'DIRK22-fine',
                             'DIRK33-coarse',
                             'DIRK33-fine',
                             ])
def test_thacker(stepper, n, dt, alpha, max_err):
    # Domain size
    l_mesh = 951646.46
    # Mesh
    mesh2d = SquareMesh(n, n, l_mesh)

    # Bathymetry and initial condition parameters
    D0 = 50.
    L = 430620.
    eta0 = 2.
    A = ((D0+eta0)**2-D0**2)/((D0+eta0)**2+D0**2)
    X0 = Y0 = l_mesh/2  # Domain offset

    # Bathymetry
    bathymetry = Function(FunctionSpace(mesh2d, "CG", 1), name='bathymetry')
    x = SpatialCoordinate(mesh2d)
    bathymetry.interpolate(D0*(1-((x[0]-X0)*(x[0]-X0)+(x[1]-Y0)*(x[1]-Y0))/(L*L)))

    # Solver
    solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    options = solverObj.options

    options.timestep = dt
    options.simulation_end_time = 43200
    options.simulation_export_time = options.timestep
    options.no_exports = True
    options.timestepper_type = stepper
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(alpha)

    # Initial conditions
    x = SpatialCoordinate(mesh2d)
    elev_init = D0*(sqrt(1-A*A)/(1-A) - 1 - ((x[0]-X0)*(x[0]-X0)+(x[1]-Y0)*(x[1]-Y0))*((1+A)/(1-A)-1)/(L*L))
    solverObj.assign_initial_conditions(elev=elev_init)

    # Iterate solver
    solverObj.iterate()

    # Extract final fields
    uv, eta = solverObj.fields.solution_2d.split()

    # Calculate relative error at domain centre
    rel_err = abs((eta.at(X0, Y0) - eta0)/eta0)

    print_output(rel_err)
    assert(rel_err < max_err)
    print_output("PASSED")
