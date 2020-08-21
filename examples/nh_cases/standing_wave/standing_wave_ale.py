# standing wave in a rectangular basin
# ===================
# Wei Pan 2018-01-08
from thetis import *

horizontal_domain_is_2d = True
lx = 20.0
ly = 2.0
nx = 20
ny = 1
if horizontal_domain_is_2d:
    mesh = RectangleMesh(nx, ny, lx, ly)
    x, y = SpatialCoordinate(mesh)
else:
    mesh = IntervalMesh(nx, lx)
    x = SpatialCoordinate(mesh)[0]
depth = 80
elev_amp = 0.1
n_layers = 3
outputdir = 'outputs_standing_wave_ale'
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
solver_obj = nhsolver_ale.FlowSolver(mesh, bathymetry_2d, n_layers)
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
# limiter
options.use_limiter_for_velocity = False
options.use_limiter_for_tracers = False
# mesh update
options.use_ale_moving_mesh = True
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d']

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

