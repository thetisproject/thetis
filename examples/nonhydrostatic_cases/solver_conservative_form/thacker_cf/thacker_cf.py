"""
Thacker wetting-drying test case
================================

"""
from thetis import *

mesh2d = Mesh('meshes/thacker_e.msh')

outputdir = 'outputs_thacker_cf'
print_output('Exporting to ' + outputdir)

# Model setup
D0 = Constant(50.)
L = Constant(430620.)

# Time steps, total simulation time
dt = 30
t_export = 360
t_end = 3600*42

# bathymetry
P1_2d = FunctionSpace(mesh2d, "DG", 1)
bathymetry_2d = Function(P1_2d, name='bathymetry')
x = SpatialCoordinate(mesh2d)
bathy = D0*(1 - (x[0]**2 + x[1]**2)/(L**2))
bathymetry_2d.interpolate(bathy)

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
options.fields_to_export = ['elev_2d']
# wetting and drying
options_nh = options.nh_model_options
options_nh.use_explicit_wetting_and_drying = True
options_nh.wetting_and_drying_threshold = 1e-2
options_nh.use_limiter_for_elevation = False
options_nh.use_limiter_for_momentum = False

# --- boundary condition ---
solver_obj.bnd_functions['shallow_water'] = {0: {'inflow': None}}

# --- create equations ---
solver_obj.create_equations()


def anal_eta(t):
    eta0 = Constant(2.)
    a = ((D0 + eta0)**2 - D0**2)/((D0 + eta0)**2 + D0**2)
    A = Constant(a)
    omega = sqrt(8*9.81*D0/L**2)
    return D0*(sqrt(1 - A**2)/(1 - A*cos(omega*t)) - (x[0]**2 + x[1]**2)/(L**2)*((1 - A**2)/(1 - A*cos(omega*t))**2 - 1) - 1)


# set initial elevation
solver_obj.assign_initial_conditions(elev=anal_eta(0))

# User-defined output
bath_minus = Function(P1_2d, name="bath_minus").assign(-bathymetry_2d)
File(os.path.join(outputdir, 'minus_bath.pvd')).write(bath_minus)

solver_obj.iterate()

# error show
anal_elev = Function(solver_obj.function_spaces.H_2d)
anal_elev.interpolate(anal_eta(t_end))
L2_elev = errornorm(anal_elev, solver_obj.fields.elev_2d)
print('L2 error for surface elevation is ', L2_elev)
