# Atmospheric pressure test case
# Constant pressure gradient applied at surface
# Result compared to analytic solution

from thetis import *
import pytest
import numpy as np


@pytest.mark.parametrize("nx,dt,max_rel_err", [(10, 1200, 2e-6)])
def test_pressure_forcing(nx, dt, max_rel_err):
    lx = 21000
    ly = 5000

    mesh2d = RectangleMesh(nx, 1, lx, ly)

    # Simulation time
    t_end = 4*24*3600.

    # bathymetry
    P1 = FunctionSpace(mesh2d, "DG", 1)
    bathymetry = Function(P1, name='bathymetry')
    x = SpatialCoordinate(mesh2d)
    bathymetry.interpolate(Constant(5.0))

    # bottom drag
    mu_manning = Constant(2.0)

    # atmospheric pressure
    atmospheric_pressure = Function(P1, name='atmospheric_pressure')
    atmospheric_pressure.interpolate(x[0])

    # --- create solver ---
    solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    options = solverObj.options
    options.dt = dt
    options.t_export = options.dt
    options.t_end = t_end - 0.1*options.dt
    solverObj.options.no_exports = True
    options.check_vol_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.timestepper_type = 'cranknicolson'
    options.shallow_water_theta = 0.5
    options.mu_manning = mu_manning
    options.atmospheric_pressure = atmospheric_pressure
    options.solver_parameters_sw = {
        'snes_type': 'newtonls',
        'snes_monitor': False,
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
    }

    solverObj.assign_initial_conditions(uv=Constant((1e-7, 0.)))

    solverObj.iterate()

    eta = solverObj.fields.elev_2d

    analytic = Function(P1, name='analytic')
    rho0 = physical_constants['rho0']
    g = physical_constants['g_grav']
    analytic.interpolate(-(x[0]-lx/2)/(rho0*g))

    area = lx*ly
    rel_err = errornorm(analytic, eta)/np.sqrt(area)
    print_output(rel_err)
    assert(rel_err < max_rel_err)
    print_output("PASSED")

if __name__ == '__main__':
    test_pressure_forcing(10, 1200, 2e-6)
