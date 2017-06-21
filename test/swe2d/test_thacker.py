# Test for wetting-drying scheme functionality, using Thacker test case
# Test case details in Gourgue et al (2009)

from thetis import *
import pytest


@pytest.mark.parametrize("n,dt,alpha,max_err", [(25, 300., 4., 0.009), (10, 600., 8., 0.06)], ids=['fine', 'coarse'])
def test_thacker(n, dt, alpha, max_err):
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
    options.t_end = 43200 - 0.1*options.timestep
    options.t_export = options.timestep
    options.no_exports = True
    options.timestepper_type = 'cranknicolson'
    options.shallow_water_theta = 0.5
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(alpha)
    options.solver_parameters_sw = {
        'snes_type': 'newtonls',
        'snes_monitor': True,
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
    }

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
