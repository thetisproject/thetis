# Lock Exchange Test case
# =======================
# Wei Pan 2019-03-06
from thetis import *

def run_lockexchange(reso_str='medium2', poly_order=1, element_family='dg-dg',
                     reynolds_number=1, use_limiter=True, dt=0.01,
                     viscosity='const', laxfriedrichs=0.0, load_export_ix=None, iterate=True, **custom_options):
    """
    Runs lock exchange problem with a bunch of user defined options.
    """
    comm = COMM_WORLD

    depth = 0.1
    refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
                  'medium2': 10, 'fine': 20}
    # set mesh resolution
    if '-' in reso_str:
        words = reso_str.split('-')
        delta_x, delta_z = [float(f) for f in words]
        layers = int(np.ceil(depth/delta_z))
    else:
        delta_x = 0.02/refinement[reso_str]
        layers = int(round(10*refinement[reso_str]))

    # generate unit mesh and transform its coords
    use_2d_horizontal_domain = False
    x_max = 0.4
    x_min = -0.4
    n_x = int((x_max - x_min)/delta_x)
    if use_2d_horizontal_domain:
        mesh2d = UnitSquareMesh(n_x, 1)
        coords = mesh2d.coordinates
        # x in [x_min, x_max], y in [-dx, dx]
        coords.dat.data[:, 0] = coords.dat.data[:, 0]*(x_max - x_min) + x_min
        coords.dat.data[:, 1] = coords.dat.data[:, 1]*0.2
    else:
        mesh2d = UnitIntervalMesh(n_x)
        coords = mesh2d.coordinates
        coords.dat.data[:] = coords.dat.data[:]*(x_max - x_min) + x_min

    # temperature and salinity, for linear eq. of state (from Petersen, 2015)
    salt_left = 0.
    salt_right = 1.3592
    tmp_const = 0.
    rho_0 = 999.972
    physical_constants['rho0'].assign(rho_0)

    # compute horizontal viscosity
    uscale = 0.001
    nu_scale = 0.5E-6#uscale * delta_x / reynolds_number

    t_end = 30
    t_export = 0.1

    lim_str = '_lim' if use_limiter else ''
    options_str = '_'.join([reso_str,
                            element_family,
                            'p{:}'.format(poly_order),
                            'visc-{:}'.format(viscosity),
                            'Re{:}'.format(reynolds_number),
                            'lf{:.1f}'.format(laxfriedrichs),
                            ]) + lim_str
    outputdir = 'outputs_sigma_' + options_str

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # create solver
    solver_obj = nhsolver_sigma.FlowSolver(mesh2d, bathymetry_2d, layers)
    options = solver_obj.options
    options.polynomial_degree = poly_order
    options.element_family = element_family
    options.timestepper_type = 'SSPRK22'
    options.update_free_surface = False
    options.use_ale_moving_mesh = False
    options.solve_salinity = True
    options.solve_temperature = False
    options.constant_temperature = Constant(tmp_const)
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_baroclinic_formulation = True
    if laxfriedrichs is None or laxfriedrichs == 0.0:
        options.use_lax_friedrichs_velocity = False
        options.use_lax_friedrichs_tracer = False
    else:
        options.use_lax_friedrichs_velocity = True
        options.use_lax_friedrichs_tracer = True
        options.lax_friedrichs_velocity_scaling_factor = Constant(laxfriedrichs)
        options.lax_friedrichs_tracer_scaling_factor = Constant(laxfriedrichs)
    options.use_limiter_for_tracers = use_limiter
    options.use_limiter_for_velocity = use_limiter
    # To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
    if viscosity == 'smag':
        options.use_smagorinsky_viscosity = True
        options.smagorinsky_coefficient = Constant(1.0/np.sqrt(reynolds_number))
    elif viscosity == 'const':
        options.horizontal_viscosity = Constant(nu_scale)
    else:
        raise Exception('Unknow viscosity type {:}'.format(viscosity))
    options.horizontal_viscosity = Constant(nu_scale)
    options.vertical_viscosity = Constant(nu_scale)
    options.horizontal_diffusivity = Constant(nu_scale)
    options.vertical_diffusivity = Constant(nu_scale)
    options.timestepper_options.use_automatic_timestep = True
    if dt is not None:
        options.timestepper_options.use_automatic_timestep = False
        options.timestep = dt
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_temperature_conservation = True
    options.check_temperature_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d', 'salt_3d', 'density_3d']
    options.fields_to_export_hdf5 = list(options.fields_to_export)
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.rho_ref = rho_0
    options.equation_of_state_options.s_ref = 0.0
    options.equation_of_state_options.th_ref = 0.0
    options.equation_of_state_options.alpha = 0.0
    options.equation_of_state_options.beta = rho_0*0.75E-3
    options.update(custom_options)

    solver_obj.create_equations()

    print_output('Running lock exchange problem with options:')
    print_output('Resolution: {:}'.format(reso_str))
    print_output('Reynolds number: {:}'.format(reynolds_number))
    print_output('Use slope limiters: {:}'.format(use_limiter))
    print_output('Horizontal viscosity: {:}'.format(nu_scale))
    print_output('Lax-Friedrichs factor: {:}'.format(laxfriedrichs))
    print_output('Exporting to {:}'.format(outputdir))

    esize = solver_obj.fields.h_elem_size_2d
    min_elem_size = comm.allreduce(np.min(esize.dat.data), op=MPI.MIN)
    max_elem_size = comm.allreduce(np.max(esize.dat.data), op=MPI.MAX)
    print_output('Elem size: {:} {:}'.format(min_elem_size, max_elem_size))

    salt_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
    if use_2d_horizontal_domain:
        x, y, z = SpatialCoordinate(solver_obj.mesh)
    else:
        x, z = SpatialCoordinate(solver_obj.mesh)

    # vertical barrier
    use_smooth_condition = False
    if use_smooth_condition:
        sigma = 1E-4
        salt_init3d.interpolate(salt_left -
                                (salt_left - salt_right)*0.5*(tanh(x/sigma) + 1.0))
    else:
        salt_init3d.interpolate(conditional(x > 0.0, salt_right, salt_left))

    if load_export_ix is None:
        solver_obj.assign_initial_conditions(salt=salt_init3d)
    else:
        assert isinstance(load_export_ix, int)
        solver_obj.load_state(load_export_ix)

    if iterate:
        solver_obj.iterate()

    return solver_obj

if __name__ == '__main__':
    run_lockexchange()
