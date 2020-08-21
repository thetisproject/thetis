# solves a standing wave in a rectangular basin based on a multi-layer solver
# ===================
# Wei Pan 2018-01-08
from thetis import *

lx = 20.0
ly = 2.0
nx = 20
ny = 1
mesh = RectangleMesh(nx, ny, lx, ly)
x, y = SpatialCoordinate(mesh)
depth = 40
elev_amp = 0.1
n_layers = 2
outputdir = 'outputs_standing_wave_ml'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
dt = 0.01
t_export = 0.1
t_end = 20.

# --- create solver ---
solver_obj = nhsolver_ml.FlowSolver(mesh, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 2
options.timestepper_type = 'CrankNicolson'
# free surface elevation
options.update_free_surface = True
# multi layer
options.n_layers = n_layers
options.alpha_nh = [0.15] # [] means uniform layers
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
#options.set_vertical_2d = True
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']

# need to call creator to create the function spaces
solver_obj.create_equations()

# set initial elevation
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.interpolate(elev_amp*cos(2*pi*x/lx))

solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()

# error show
anal_elev = Function(solver_obj.function_spaces.H_2d).interpolate(elev_amp*cos(2*pi*x/lx)*cos(sqrt(9.81*2*pi/lx)*t_end))
L2_elev = errornorm(anal_elev, solver_obj.fields.elev_2d)/sqrt(lx*ly)

print (L2_elev)

