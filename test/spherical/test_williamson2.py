# Tuomas Karna 2015-03-03
from thetis import *
import math


def test_steady_state_channel(do_export=False):

    n = 200  # number of timesteps
    dt = 1000.
    g = float(physical_constants['g_grav'])

    R = 6371220.
    h0 = 5960.
    day = 86400.
    u0 = 2*pi*R/(12*day)
    Omega = 2*pi/day
    nonlin = True
    coriolis = True
    family = 'rt-dg'
    refinement_level = 3
    degree = 3

    outputdir = 'outputs_{nonlin}_{coriolis}_{family}_{ref}_{deg}'.format(**{
        'nonlin': 'nonlin' if nonlin else 'lin',
        'coriolis': 'coriolis' if coriolis else 'nocor',
        'family': family,
        'ref': refinement_level,
        'deg': degree})

    mesh2d = IcosahedralSphereMesh(radius=R,
            refinement_level=refinement_level, degree=degree)
    x = mesh2d.coordinates
    mesh2d.init_cell_orientations(x)

    uv0 = as_vector([-u0*x[1]/R, u0*x[0]/R, 0.0])
    eta0 = -(Constant(coriolis)*R*Omega*u0+nonlin*u0**2/2.)*x[2]*x[2]/(R*R*g)
    f = coriolis*2*Omega

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(h0)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.element_family = family
    solver_obj.options.nonlin = nonlin
    solver_obj.options.t_export = dt
    solver_obj.options.outputdir = outputdir
    solver_obj.options.t_end = n*dt
    solver_obj.options.no_exports = not do_export
    solver_obj.options.timestepper_type = 'cranknicolson'
    solver_obj.options.shallow_water_theta = 1.0
    solver_obj.options.solver_parameters_sw = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': False,
        'snes_type': 'newtonls',
    }
    solver_obj.options.coriolis = f
    solver_obj.options.dt = dt

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}
    parameters['quadrature_degree'] = 5

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(uv=uv0, elev=eta0)

    solver_obj.iterate()

    uv, eta = solver_obj.fields.solution_2d.split()

    assert True
    print_output("PASSED")


if __name__ == '__main__':
    test_steady_state_channel(do_export=True)
