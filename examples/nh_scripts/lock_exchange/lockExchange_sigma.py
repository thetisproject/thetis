# Lock Exchange Test case
# =======================
#
# Solves non-hydrostatic flow in a closed rectangular channel.
#
# Dianeutral mixing depends on mesh Reynolds number [1]
# Re_h = U dx / nu
# U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(g_h drho/rho_0)
# dx = horizontal mesh size
# nu = background viscosity
#
#
# Smagorinsky factor should be C_s = 1/sqrt(Re_h)
#
# Mesh resolutions:
# - ilicak [1]:  dx =  500 m,  20 layers
# COMODO lock exchange benchmark [2]:
# - coarse:      dx = 2000 m,  10 layers
# - coarse2 (*): dx = 1000 m,  20 layers
# - medium:      dx =  500 m,  40 layers
# - medium2 (*): dx =  250 m,  80 layers
# - fine:        dx =  125 m, 160 layers
# (*) not part of the original benchmark
#
# [1] Ilicak et al. (2012). Spurious dianeutral mixing and the role of
#     momentum closure. Ocean Modelling, 45-46(0):37-58.
#     http://dx.doi.org/10.1016/j.ocemod.2011.10.003
# [2] COMODO Lock Exchange test.
#     http://indi.imag.fr/wordpress/?page_id=446
# [3] Petersen et al. (2015). Evaluation of the arbitrary Lagrangian-Eulerian
#     vertical coordinate method in the MPAS-Ocean model. Ocean Modelling,
#     86:93-113.
#     http://dx.doi.org/10.1016/j.ocemod.2014.12.004
#
# Tuomas Karna 2015-03-03

from thetis import *
from diagnostics import *
from plotting import *


def run_lockexchange(reso_str='coarse', poly_order=1, element_family='dg-dg',
                     reynolds_number=1, use_limiter=not True, dt=0.01,
                     viscosity='const', laxfriedrichs=0.0, non_hydrostatic = not False,
                     load_export_ix=None, iterate=True, **custom_options):
    """
    Runs lock exchange problem with a bunch of user defined options.
    """
    comm = COMM_WORLD

    depth = 0.1
    refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
                  'medium2': 10, 'fine': 20, 'ilicak': 4}
    # set mesh resolution
    if '-' in reso_str:
        words = reso_str.split('-')
        delta_x, delta_z = [float(f) for f in words]
        layers = int(np.ceil(depth/delta_z))
    else:
        delta_x = 0.02/refinement[reso_str]
        layers = int(round(10*refinement[reso_str]))
        if reso_str == 'ilicak':
            layers = 20

    # generate unit mesh and transform its coords
    use_2d_horizontal_domain = not True
    x_max = 0.4
    x_min = -0.4
    n_x = (x_max - x_min)/delta_x
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

    u_max = 1.0 # set for use_automatic_timestep
    w_max = 1.2E-2

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
    outputdir = 'outputs_sscale_' + options_str

    #########################################
    nh_str = '_nh' if non_hydrostatic else '_hydro'
    outputdir = outputdir + nh_str + '_sigma_diff10-6'
    #########################################

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # create solver
    #solver_obj = solver_nh.FlowSolver(mesh2d, bathymetry_2d, layers)
    if non_hydrostatic is True:
        solver_obj = solver_sigma.FlowSolver(mesh2d, bathymetry_2d, layers)
    else:
        solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
    options = solver_obj.options
    options.polynomial_degree = poly_order
    options.element_family = element_family
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = True
    options.solve_temperature = False
    options.constant_temperature = Constant(tmp_const)
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = not True
    options.use_baroclinic_formulation = True
    if laxfriedrichs is None or laxfriedrichs == 0.0:
        options.use_lax_friedrichs_velocity = False
        options.use_lax_friedrichs_tracer = False
    else:
        options.use_lax_friedrichs_velocity = True
        options.use_lax_friedrichs_tracer = True
        options.lax_friedrichs_velocity_scaling_factor = Constant(laxfriedrichs)
        options.lax_friedrichs_tracer_scaling_factor = Constant(laxfriedrichs)
    options.use_limiter_for_tracers = True#use_limiter
    options.use_limiter_for_velocity = True#use_limiter
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
    options.horizontal_viscosity_scale = Constant(nu_scale)
    options.horizontal_velocity_scale = Constant(u_max)
    options.vertical_velocity_scale = Constant(w_max)
    options.timestepper_options.use_automatic_timestep = True
    if dt is not None:
        options.timestepper_options.use_automatic_timestep = False
        options.timestep = dt
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.check_temperature_conservation = True
    options.check_temperature_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d', 'density_3d',
                                #'uv_dav_2d', 'uv_dav_3d', #'baroc_head_3d',
                                'smag_visc_3d']
    options.fields_to_export_hdf5 = list(options.fields_to_export)
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.rho_ref = rho_0
    options.equation_of_state_options.s_ref = 0.0
    options.equation_of_state_options.th_ref = 0.0
    options.equation_of_state_options.alpha = 0.0
    options.equation_of_state_options.beta = rho_0*0.75E-3
    options.update(custom_options)

    if comm.size == 1:
        solver_obj.add_callback(RPECalculator(solver_obj))
        solver_obj.add_callback(FrontLocationCalculator(solver_obj))
        # solver_obj.add_callback(PlotCallback(solver_obj, append_to_log=False))

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
    # smooth condition
    sigma = 1E-4
    salt_init3d.interpolate(salt_left -
                            (salt_left - salt_right)*0.5*(tanh(x/sigma) + 1.0))
    # vertical barrier
    salt_init3d.interpolate(conditional(x > 0.0, salt_right, salt_left))

    if load_export_ix is None:
        solver_obj.assign_initial_conditions(salt=salt_init3d)
    else:
        assert isinstance(load_export_ix, int)
        solver_obj.load_state(load_export_ix)

    if iterate:
        solver_obj.iterate()

    return solver_obj


def get_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reso_str', type=str,
                        help='mesh resolution string. A named mesh  or "dx-dz" string',
                        default='coarse')
    parser.add_argument('--no-limiter', action='store_false', dest='use_limiter',
                        help='do not use slope limiter for tracers')
    parser.add_argument('-p', '--poly_order', type=int, default=1,
                        help='order of finite element space')
    parser.add_argument('-f', '--element-family', type=str,
                        help='finite element family', default='dg-dg')
    parser.add_argument('-re', '--reynolds-number', type=float, default=1.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('-dt', '--dt', type=float,
                        help='force value for 3D time step')
    parser.add_argument('-visc', '--viscosity', type=str,
                        help='Type of horizontal viscosity',
                        default='const',
                        choices=['const', 'smag'])
    parser.add_argument('-lf', '--laxfriedrichs', type=float,
                        help='Lax-Friedrichs flux factor for uv and temperature',
                        default=0.0)
    return parser


def parse_options():
    parser = get_argparser()
    args, unknown_args = parser.parse_known_args()
    args_dict = vars(args)
    run_lockexchange(**args_dict)


if __name__ == '__main__':
    #parse_options()
    run_lockexchange()
