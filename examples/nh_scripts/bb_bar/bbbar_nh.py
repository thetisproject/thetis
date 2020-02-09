# Beji and Battjes (1993), case A: wave propagation over a bar
# ===============================
# Wei Pan 2017-07-25

from thetis import *
import numpy as np

horizontal_domain_is_2d = not True
lx = 35.0
ly = 0.1
nx = 350
ny = 1
if horizontal_domain_is_2d:
    mesh = RectangleMesh(nx, ny, lx, ly)
else:
    mesh = IntervalMesh(nx, lx)
n_layers = 3
outputdir = 'outputs_bbbar_nh'
print_output('Exporting to ' + outputdir)

# --- bathymetry ---
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
def get_bathymetry():
    mesh = bathymetry_2d.ufl_domain()
    if horizontal_domain_is_2d:
        x_vector = mesh.coordinates.dat.data[:, 0]
    else:
        x_vector = mesh.coordinates.dat.data[:]
    b_vector = bathymetry_2d.dat.data
    assert x_vector.shape[0] == b_vector.shape[0]
    for i, xy in enumerate(x_vector):
        s = 25.
        if xy <= 6.:
            b_vector[i] = 0.4
        elif xy <= 12.:
            b_vector[i] = -0.05*xy + 0.7
        elif xy <= 14.:
            b_vector[i] = 0.1
        elif xy <= 17.:
            b_vector[i] = 0.1*xy - 1.3
        elif xy <= 19.:
            b_vector[i] = 0.4
        elif xy <= s:
            b_vector[i] = -0.04*xy + 1.16
        else:
            b_vector[i] = -0.04*s + 1.16
    return bathymetry_2d
bathymetry_2d = get_bathymetry()

# set time step, export interval and run duration
dt = 0.01
t_export = 0.1
t_end = 40

# --- create solver ---
solver_obj = solver_nh.FlowSolver(mesh, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'#'LeapFrog'#'SSPRK22'
options.use_nonlinear_equations = True
options.update_free_surface = True
# for three-layer NH model, suggest to set alpha as 0.1, beta 0.45
# for coupled two-layer NH model, suggest to set alpha as 0.2
# for reduced model, alpha and beta depend on specific cases,
# as recommended by Cui et al. (2014), alpha = 0.15 and beta = 1.0
# for multi-layer case,
# layer thickness accounting for total height defined by alpha_nh list
options.alpha_nh = [] # [] means uniform layers
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.timestepper_options.use_automatic_timestep = False
options.output_directory = outputdir
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
# for horizontal 1D case
options.set_vertical_2d = True
# set sponge layer to absorb relected waves
options.sponge_layer_length = [5., 0.] # default is None, i.e. no absorption
options.sponge_layer_xstart = [30., 0.] # default is 0.
##### --- wetting and drying --- #####
options.constant_mindep = True
# if True, the thin-film depth at wetting-drying interface is not varied and equals to wd_mindep
# if False, the thin-film depth at wetting-drying interface at each step is determine by wd_mindep,
# which here refers to the thin-film depth at the lowest depth, i.e. highest bathymetry point
#
### note: if options.thin_film is True, thin-film wd scheme will be used ###
options.thin_film = False
options.wd_mindep = 0.

# need to call creator to create the function spaces
solver_obj.create_equations()

# --- input wave ---
pi = 4*np.arctan(1.)
eta_amp = 0.01
period = 2.02
def get_inputelevation(t):
    return eta_amp*np.sin(2*pi/period*t)

# --- boundary condition ---
t = 0.
solver_obj.create_function_spaces()
H_2d = solver_obj.function_spaces.H_2d
H_3d = solver_obj.function_spaces.H
ele_bc = Function(H_2d, name="boundary elevation").assign(get_inputelevation(t))
ele_bc_3d = Function(H_3d, name="3D boundary elevation").assign(get_inputelevation(t))
inflow_tag = 1
solver_obj.bnd_functions['shallow_water'] = {inflow_tag: {'elev': ele_bc}}
#solver_obj.bnd_functions['momentum'] = {inflow_tag: {'elev3d': ele_bc_3d}}

# --- time updated ---
def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 2d mode"""
    #print_output("Updating boundary condition at t={}".format(t_new))
    ele_bc.assign(get_inputelevation(t_new))
    ele_bc_3d.assign(get_inputelevation(t_new))

# --- initial conditions, create all function spaces, equations etc ---
# set uv for Manning drag
solver_obj.assign_initial_conditions()
solver_obj.iterate(update_forcings = update_forcings)

