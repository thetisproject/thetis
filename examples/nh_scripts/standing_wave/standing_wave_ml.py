# 1st test case of Thetis-NH solver
# ===================
#
# Solves a standing wave in a rectangular basin using 3D Non-hydrostatic equations
#
# Initial condition for elevation corresponds to a standing wave.
#
# This example tests dispersion of surface waves improved by non-hydrostatic pressure.
#
# Wei Pan 08/Jan/2018
from thetis import *

horizontal_domain_is_2d = not True
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
n_layers = 4
outputdir = 'outputs_standing_wave_ml'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
dt = 0.005 # note: more vertical layers need lower time step
t_export = 0.1
t_end = 20.

# --- create solver ---
if horizontal_domain_is_2d:
    solver_obj = solver_ml.FlowSolver(mesh, bathymetry_2d)
else:
    solver_obj = solver2d_ml.FlowSolver(mesh, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 2
options.timestepper_type = 'CrankNicolson'
options.use_nonlinear_equations = True
options.update_free_surface = True
# for three-layer NH model, suggest to set alpha as 0.1, beta 0.45
# for coupled two-layer NH model, suggest to set alpha as 0.2
# for reduced model, alpha and beta depend on specific cases,
# as recommended by Cui et al. (2014), alpha = 0.15 and beta = 1.0
# for multi-layer case,
# layer thickness accounting for total height defined by alpha_nh list
#options.use_nh_solver = True
options.n_layers = n_layers
options.alpha_nh = [0.15] # [] means uniform layers
options.output_directory = outputdir
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.fields_to_export = ['uv_2d', 'elev_2d']

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

# set initial elevation
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.interpolate(elev_amp*cos(2*pi*x/lx))

solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()

# error show
anal_elev = Function(solver_obj.function_spaces.H_2d).interpolate(elev_amp*cos(2*pi*x/lx)*cos(sqrt(9.81*2*pi/lx)*t_end))
L2_elev = errornorm(anal_elev, solver_obj.fields.elev_2d)/sqrt(lx*ly)

print (L2_elev)

