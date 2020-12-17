"""
Standing wave test case
=======================

Solves a standing wave using wave equation in conservative form.

Initial condition for elevation corresponds to a standing wave.

This example tests solver in conservative form.
"""
from thetis import *

lx = 20.0
ly = 2.0
nx = 10
ny = 1
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 80.
elev_amp = 0.1

outputdir = 'outputs_standing_wave_cf'

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
dt = 0.005
t_export = 0.1
t_end = 20.

# --- create solver ---
solver_obj = solver2d_cf.FlowSolverCF(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
# time stepper
options.timestepper_type = 'SSPRK33'
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = False
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['h_2d']
options.fields_to_export_hdf5 = ['h_2d']

# --- create equations ---
solver_obj.create_equations()

# set initial elevation
elev_init = Function(solver_obj.function_spaces.H_2d)
x, y = SpatialCoordinate(mesh2d)
elev_init.interpolate(elev_amp*cos(2*pi*x/lx))
solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()

# error show
anal_elev = Function(solver_obj.function_spaces.H_2d)
anal_elev.interpolate(elev_amp*cos(2*pi*x/lx)*cos(sqrt(9.81*2*pi/lx)*t_end))
solver_obj.fields.h_2d.assign(solver_obj.fields.h_2d - depth)
L2_elev = errornorm(anal_elev, solver_obj.fields.h_2d)/sqrt(lx*ly)
print('L2 error for surface elevation is ', L2_elev)
