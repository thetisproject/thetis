from thetis import *
import numpy
import uptide
import datetime


def latlon_from_xyz(xyz):
    r = numpy.linalg.norm(xyz, axis=1)
    lat = numpy.arcsin(xyz[:, 2]/r)
    lon = numpy.arctan2(xyz[:, 1], xyz[:, 0])
    return lat, lon


def test_steady_state_channel(do_export=False):

    n = 200  # number of timesteps
    dt = 600.

    R = 6371220.
    h0 = 5e4  # a *very* deep ocean
    day = 86400.
    Omega = 2*pi/day
    nonlin = True
    coriolis = True
    family = 'rt-dg'
    refinement_level = 3
    degree = 2

    outputdir = 'outputs_{nonlin}_{ref}'.format(**{
        'nonlin': 'nonlin' if nonlin else 'lin',
        'coriolis': 'coriolis' if coriolis else 'nocor',
        'family': family,
        'ref': refinement_level,
        'deg': degree})

    # mesh2d = Mesh('ModernModernEarth.node', dim=3)
    mesh2d = IcosahedralSphereMesh(radius=R,
                                   refinement_level=refinement_level, degree=degree)
    x = mesh2d.coordinates
    mesh2d.init_cell_orientations(x)

    f = coriolis*2*Omega

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(h0)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.element_family = family
    solver_obj.options.use_nonlinear_equations = nonlin
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.output_directory = outputdir
    solver_obj.options.simulation_end_time = n*dt
    solver_obj.options.no_exports = not do_export
    solver_obj.options.fields_to_export = ['uv_2d', 'elev_2d', 'equilibrium_tide']
    solver_obj.options.timestepper_type = 'CrankNicolson'
    solver_obj.options.timestepper_options.implicitness_theta = 1.0
    solver_obj.options.timestepper_options.solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': False,
        'snes_type': 'newtonls',
    }
    solver_obj.options.coriolis_frequency = Constant(f)
    solver_obj.options.timestep = dt

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}
    parameters['quadrature_degree'] = 5
    solver_obj.create_function_spaces()
    H_2d = solver_obj.function_spaces.H_2d

    # by defining this function, the equilibrium tide will be applied
    equilibrium_tide = Function(H_2d)
    solver_obj.options.equilibrium_tide = equilibrium_tide
    # Love numbers:
    solver_obj.options.equilibrium_tide_alpha = Constant(0.693)
    solver_obj.options.equilibrium_tide_beta = Constant(0.953)

    h2dxyz = Function(H_2d*H_2d*H_2d)
    for i, h2dxyzi in enumerate(h2dxyz.split()):
        h2dxyzi.interpolate(x[i])
    lat, lon = latlon_from_xyz(numpy.vstack(h2dxyz.vector()[:]).T)

    tide = uptide.Tides(uptide.ALL_EQUILIBRIUM_TIDAL_CONSTITUENTS)
    tide.set_initial_time(datetime.datetime(2013, 1, 1))

    # a function called every timestep that updates the equilibrium tide
    def update_forcings(t):
        equilibrium_tide.vector()[:] = uptide.equilibrium_tide(tide, lat, lon, t)

    update_forcings(0)

    # we start with eta=equilibrium tide to avoid initial shock and need for ramp up
    solver_obj.assign_initial_conditions(elev=equilibrium_tide)

    solver_obj.iterate(update_forcings=update_forcings)

    uv, eta = solver_obj.fields.solution_2d.split()

    assert True
    print_output("PASSED")


if __name__ == '__main__':
    test_steady_state_channel(do_export=True)
