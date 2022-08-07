"""
Thacker wetting-drying test case.

The analytical test case is defined in [1]; the model setup is derived from
[2].

[1] Thacker (1981). Some exact solutions to the nonlinear shallow-water
    wave equations. Journal of Fluid Mechanics. doi:10.1017/S0022112081001882
[2] Gourgue et al. (2009). A flux-limiting wetting-drying method for
    finite-element shallow-water models, with application to the Scheldt
    Estuary. Advances in Water Resources.DOI: 10.1016/j.advwatres.2009.09.005
"""
from thetis import *
import pytest


@pytest.mark.parametrize("stepper,n,dt,max_err",
                         [
                             ('BackwardEuler', 10, 600., 0.33),
                             ('BackwardEuler', 25, 300., 0.19),
                             ('CrankNicolson', 10, 600., 0.26),
                             ('CrankNicolson', 25, 300., 0.15),
                             ('DIRK22', 10, 600., 0.26),
                             ('DIRK22', 25, 300., 0.15),
                             ('DIRK33', 10, 600., 0.26),
                             ('DIRK33', 25, 300., 0.15),
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
def test_thacker(stepper, n, dt, max_err):
    """
    Run Thacker wetting-drying test case
    """
    l_mesh = 951646.46  # domain size
    mesh2d = SquareMesh(n, n, l_mesh)

    # bathymetry and initial condition parameters
    D0 = 50.
    L = 430620.
    eta0 = 2.
    A = ((D0+eta0)**2-D0**2)/((D0+eta0)**2+D0**2)
    X0 = Y0 = l_mesh/2  # Domain offset

    # bathymetry
    bathymetry = Function(get_functionspace(mesh2d, "CG", 1), name='bathymetry')
    x, y = SpatialCoordinate(mesh2d)
    bath_expr = D0*(1-((x-X0)**2+(y-Y0)**2)/L**2)
    bathymetry.interpolate(bath_expr)

    # solver
    solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    options = solverObj.options

    options.timestep = dt
    options.simulation_end_time = 43200
    options.simulation_export_time = 600.
    options.no_exports = True
    options.swe_timestepper_type = stepper
    options.use_wetting_and_drying = True
    options.use_automatic_wetting_and_drying_alpha = True

    # initial conditions
    elev_init = D0*(sqrt(1-A*A)/(1-A) - 1
                    - ((x-X0)**2+(y-Y0)**2)*((1+A)/(1-A)-1)/L**2)
    solverObj.assign_initial_conditions(elev=elev_init)

    # run for one cycle
    solverObj.iterate()
    uv, eta = solverObj.fields.solution_2d.split()

    # mask out dry areas with a smooth function
    r = sqrt((x-X0)**2 + (y-Y0)**2)
    mask = 0.5*(1 - tanh((r - 420000.)/1000.))
    correct = mask * elev_init
    eta.project(mask * eta)  # mask ~= 1.0 in the center

    # compute L2 error
    l2_err = errornorm(correct, eta)/l_mesh
    print_output('elev L2 error {:.12f}'.format(l2_err))
    assert l2_err < max_err
