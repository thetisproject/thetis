# Beji and Battjes (1993), case A: wave propagation over a bar
# ===============================
# Wei Pan 2017-07-25

from thetis import *
import numpy as np

horizontal_domain_is_2d = True
lx = 35.0
ly = 1
nx = 350
ny = 1
n_layers = 3
outputdir = 'outputs_bbbar_cf'
print_output('Exporting to ' + outputdir)

use_multi_resolution = not True
if use_multi_resolution: # higher resolution within x = 10 - 22
    # x at interfaces including ends; dx within ranges
    x_mr = [0., 10., 22., 35.]; dx_mr = [0.5, 0.1, 0.5]
    assert len(x_mr) == len(dx_mr) + 1
    # sum number of elements before interfaces
    nx_mr = []
    sum_nx = 0
    for i in range(len(dx_mr)):
        sum_nx += int((x_mr[i+1] - x_mr[i])/dx_mr[i])
        nx_mr.append(sum_nx)
    # total number of elements in x-direction
    nx = nx_mr[len(dx_mr) - 1]
    # x in unit mesh at interface
    x_unit_mr = [0.]
    x_unit_mr.extend([nxi/nx for nxi in nx_mr])
if horizontal_domain_is_2d:
    mesh = RectangleMesh(nx, ny, lx, ly)
    if use_multi_resolution:
        mesh = UnitSquareMesh(nx, 1)
        coords = mesh.coordinates
        tmp = [c for c in coords.dat.data[:, 0]]
        for i, x in enumerate(coords.dat.data[:, 0]):
            for ii in range(len(dx_mr)):
                if x <= x_unit_mr[ii+1] and x >= x_unit_mr[ii]:
                    tmp[i] = (coords.dat.data[i, 0] - x_unit_mr[ii])/(x_unit_mr[ii+1] - x_unit_mr[ii])*(x_mr[ii+1] - x_mr[ii]) + x_mr[ii]
        coords.dat.data[:, 0] = tmp[:]
        coords.dat.data[:, 1] = coords.dat.data[:, 1]*ly
else:
    mesh = IntervalMesh(nx, lx)

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
solver_obj = nhsolver_cf.FlowSolver(mesh, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'
options.timestepper_options.use_automatic_timestep = False
# free surface elevation
options.update_free_surface = True
options.solve_separate_elevation_gradient = True
# tracer
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
# flux
options.use_hllc_flux = False
# note conservative form is more stable at least for this case
options.solve_conservative_momentum = True
options.use_vert_dg0 = not True
# limiter
options.use_limiter_for_velocity = False
options.use_limiter_for_tracers = False
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.no_exports = False
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']
# sponge layer
options.sponge_layer_length = [5., 0.]
options.sponge_layer_start = [30., 0.]

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
ele_bc = Function(H_2d, name="boundary elevation").assign(get_inputelevation(t))
inflow_tag = 1
solver_obj.bnd_functions['shallow_water'] = {inflow_tag: {'elev': ele_bc}}

# --- time updated ---
def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 2d mode"""
    ele_bc.assign(get_inputelevation(t_new))

# --- initial conditions, create all function spaces, equations etc ---
solver_obj.assign_initial_conditions()
solver_obj.iterate(update_forcings = update_forcings)

