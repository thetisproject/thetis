# Berkhoff et al. (1982), evaluating wave refraction and deffraction
# ===============================
# Wei Pan 28/07/2020

from thetis import *
import numpy as np

lx = 28
ly = 20
nx = 280
ny = 200
mesh2d = RectangleMesh(nx, ny, lx, ly)
n_layers = 1

outputdir = 'outputs_berkhoff_cf'
print_output('Exporting to '+outputdir)

use_multi_resolution = True
if use_multi_resolution: # higher resolution within x = 10 - 22
    # x at interfaces including ends; dx within ranges
    x_mr = [0., 5.4, 22., 28.]; dx_mr = [0.3, 0.1, 0.3]
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

    if True:
        mesh2d = UnitSquareMesh(nx, ny)
        coords = mesh2d.coordinates
        tmp = [c for c in coords.dat.data[:, 0]]
        for i, x in enumerate(coords.dat.data[:, 0]):
            for ii in range(len(dx_mr)):
                if x <= x_unit_mr[ii+1] and x >= x_unit_mr[ii]:
                    tmp[i] = (coords.dat.data[i, 0] - x_unit_mr[ii])/(x_unit_mr[ii+1] - x_unit_mr[ii])*(x_mr[ii+1] - x_mr[ii]) + x_mr[ii]
        coords.dat.data[:, 0] = tmp[:]
        coords.dat.data[:, 1] = coords.dat.data[:, 1]*ly

# --- bathymetry ---
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

def get_bathymetry():
    mesh2d = bathymetry_2d.ufl_domain()
    x_vector = mesh2d.coordinates.dat.data
    b_vector = bathymetry_2d.dat.data
    assert x_vector.shape[0] == b_vector.shape[0]
    for i, xy in enumerate(x_vector):
        deg = -20./180.*pi
        y = (xy[1]-10)*np.cos(deg) - (xy[0]-10)*np.sin(deg)
        x = (xy[0]-10)*np.cos(deg) + (xy[1]-10)*np.sin(deg)
        if x < -5.484:
            b_vector[i] = 0.45
        elif (x/3.)**2 + (y/4.)**2 <= 1.:
            b_vector[i] = max(0.10, 0.45 - 0.02*(5.484+x)) - (-0.3 + 0.5*np.sqrt(1 - (y/5.)**2 - (x/3.75)**2))
        else:
            b_vector[i] = max(0.10, 0.45 - 0.02*(5.484+x))
    return bathymetry_2d
bathymetry_2d = get_bathymetry()

# set time step, export interval and run duration
dt = 0.01
t_export = 0.05
t_end = 30.

# --- create solver ---
solver_obj = nhsolver_cf.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'
options.timestepper_options.use_automatic_timestep = False
# free surface elevation
options.update_free_surface = True
options.solve_separate_elevation_gradient = True
# lax-friedrichs
options.use_lax_friedrichs_velocity = True
options.lax_friedrichs_velocity_scaling_factor = Constant(5.)
# conservative form
options.solve_conservative_momentum = True
options.use_vert_dg0 = not True
# tracer
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
# limiter
options.use_limiter_for_velocity = False
options.use_limiter_for_tracers = False
options.use_limiter_for_elevation = False
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['elev_2d']
# sponge layer
options.sponge_layer_length = [3., 0.]
options.sponge_layer_start = [25., 0.]
# flux
options.use_hllc_flux = not True
# wetting and drying
options.use_wetting_and_drying = False
options.wetting_and_drying_threshold = 1e-4
options.depth_wd_interface = 1e-2

# need to call creator to create the function spaces
solver_obj.create_equations()

# --- input wave ---
eta_amp = 0.0232
period = 1.
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

