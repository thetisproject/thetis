# This is a dam break case to test solver in conservative form with wetting and drying treatment
#
# Wei Pan 2020-03-23

from thetis import *

lx = 2000.
ly = 1e2
nx = 500
ny = 1
mesh = UnitSquareMesh(nx, ny)
coords = mesh.coordinates
# x in [x_min, x_max], y in [-dx, dx]
x_min = -0.5*lx
coords.dat.data[:, 0] = coords.dat.data[:, 0]*lx + x_min
coords.dat.data[:, 1] = coords.dat.data[:, 1]*ly

outputdir = 'dam_break_cf_b'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(0.)

# set time step size, export interval and run duration
dt = 0.01
t_export = 1.
t_end = 15.

# --- create solver ---
solver_obj = nhsolver2d_cf.FlowSolver(mesh, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK33'#'CrankNicolson'
if options.timestepper_type == 'SSPRK33':
    options.timestepper_options.use_automatic_timestep = False
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['h_ls']
# flux
options.use_hllc_flux = True
# limiter
options.use_limiter_for_elevation = False
# granular flow
options.flow_is_granular = True
options.no_wave_flow = True
options.phi_i = 0. # internal friction angle
options.phi_b = 0. # bed friction angle
options.lamda = 0.
slope_rad = 20./180.*pi
options.bed_slope = conditional(SpatialCoordinate(mesh)[0] < lx, 
                                as_vector((cos(0.5*pi - slope_rad), cos(0.5*pi), cos(slope_rad))), 
                                as_vector((0, 0, 1)))

# wetting and drying
options.use_wetting_and_drying = True
options.wetting_and_drying_threshold = 1e-5

# inflow boundary
solver_obj.bnd_functions['landslide_motion'] = {1: {'inflow': None}, 2: {'outflow': None}}

# need to call creator to create the function spaces
solver_obj.create_equations()

# set initial elevation
h_init = Function(solver_obj.function_spaces.H_2d)
h_init.interpolate(conditional(SpatialCoordinate(mesh)[0] < 0, 20., 0.))

solver_obj.assign_initial_conditions(h_ls=h_init)

solver_obj.iterate()

# error show
x, y = SpatialCoordinate(mesh)
grav = 9.81
h0 = 20.
c0 = sqrt(grav*h0*cos(slope_rad))
m = -grav*sin(slope_rad) + grav*cos(slope_rad)*tan(options.phi_b)
t = t_end
hs = 1./(9.*grav*cos(slope_rad))*(x/t - 2.*c0 + 0.5*m*t)**2
xl = -c0*t - 0.5*m*t**2
xr = 2*c0*t - 0.5*m*t**2
anal_hs = Function(solver_obj.function_spaces.H_ls).interpolate(conditional(x > xl, conditional(x < xr, hs, options.wetting_and_drying_threshold), h0))
L2_error = errornorm(anal_hs, solver_obj.fields.h_ls)/sqrt(lx*ly)

print (L2_error)

