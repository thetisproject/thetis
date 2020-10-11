"""
Wave over submerged bar test case
=================================

A regular wave inlets from left boundary with amplitude of 1 cm and period of 2.02 s.

This example represents wave shoaling after interaction with uneven bottom.
"""
from thetis import *
import numpy as np

lx = 35.0
ly = 1
nx = 700
ny = 1

outputdir = 'outputs_bbbar_2d'

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

# --- bathymetry ---
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
def get_bathymetry():
    mesh = bathymetry_2d.ufl_domain()
    x_vector = mesh.coordinates.dat.data[:, 0]
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
dt = 0.005
t_export = 0.1
t_end = 40

# choose if using non-hydrostatic model
solve_nonhydrostatic_pressure = True
use_2d_solver = True
n_layers = 1

# --- create solver ---
solver_obj = solver_nh.FlowSolverNH(mesh, bathymetry_2d, n_layers, use_2d_solver)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'CrankNicolson'
# time stepper
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = False
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']
# non-hydrostatic
if solve_nonhydrostatic_pressure:
    options_nh = options.nh_model_options
    options_nh.solve_nonhydrostatic_pressure = solve_nonhydrostatic_pressure
    options_nh.use_2d_solver = use_2d_solver
    options_nh.n_layers = n_layers

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
solver_obj.iterate(update_forcings = update_forcings)
