"""
Dam break test case a
=====================

Solves equations in conservative form.

Granular material initially occupies the left hand side of the domain.

This example tests solver in conservative form with wetting and drying treatment.
"""
from thetis import *

lx = 2000.
ly = 1e2
nx = 500
ny = 1
mesh2d = UnitSquareMesh(nx, ny)
coords = mesh2d.coordinates
x_min = -0.5*lx
coords.dat.data[:, 0] = coords.dat.data[:, 0]*lx + x_min
coords.dat.data[:, 1] = coords.dat.data[:, 1]*ly

outputdir = 'outputs_dam_break_cf_a'

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(0.)

# set time step, export interval and run duration
dt = 0.01
t_export = 1.
t_end = 15.

# --- create solver ---
solver_obj = solver2d_cf.FlowSolverCF(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK33'
# time stepper
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = False
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['h_2d']
options.fields_to_export_hdf5 = ['h_2d']
# granular flow
options_nh = options.nh_model_options
options_nh.flow_is_granular = True
options_nh.phi_i = 0.  # internal friction angle
options_nh.phi_b = 0.  # bed friction angle
options_nh.lamda = 0.
slope_rad = 0./180.*pi
options_nh.bed_slope = Constant((cos(0.5*pi - slope_rad), cos(0.5*pi), cos(slope_rad)))  # flat bed
# wetting and drying
options_nh.use_explicit_wetting_and_drying = True
options_nh.wetting_and_drying_threshold = 1e-5

# --- boundary condition ---
solver_obj.bnd_functions['shallow_water'] = {1: {'inflow': None}, 2: {'outflow': None}}

# --- create equations ---
solver_obj.create_equations()

# set initial elevation
h_init = Function(solver_obj.function_spaces.H_2d)
x, y = SpatialCoordinate(mesh2d)
h_init.interpolate(conditional(x < 0, 20., 0.))
solver_obj.assign_initial_conditions(elev=h_init)


# --- time updated ---
def update_forcings(t_new):
    if options_nh.use_explicit_wetting_and_drying:
        solver_obj.wd_modification.apply(
            solver_obj.fields.solution_2d,
            options_nh.wetting_and_drying_threshold
        )


solver_obj.iterate(update_forcings=update_forcings)

# error show
grav = 9.81
h0 = 20.
c0 = sqrt(grav*h0*cos(slope_rad))
m = -grav*sin(slope_rad) + grav*cos(slope_rad)*tan(options_nh.phi_b)
t = t_end
hs = 1./(9.*grav*cos(slope_rad))*(x/t - 2.*c0 + 0.5*m*t)**2
xl = -c0*t - 0.5*m*t**2
xr = 2*c0*t - 0.5*m*t**2
anal_hs = Function(solver_obj.function_spaces.H_2d).interpolate(conditional(x > xl, conditional(x < xr, hs, options_nh.wetting_and_drying_threshold), h0))
L2_error = errornorm(anal_hs, solver_obj.fields.h_2d)/sqrt(lx*ly)

print('L2 error for water height is ', L2_error)
