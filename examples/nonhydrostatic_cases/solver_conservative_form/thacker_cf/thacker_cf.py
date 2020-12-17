"""
Thacker wetting-drying test case
================================

"""
from thetis import *

mesh2d = Mesh('meshes/04.msh')

outputdir = 'outputs_thacker_cf'
print_output('Exporting to ' + outputdir)

# Model setup
D0 = Constant(50.)
L = Constant(430620.)
eta0 = Constant(2.)
a = ((D0 + eta0)**2 - D0**2)/((D0 + eta0)**2 + D0**2)
A = Constant(a)

# Time steps, total simulation time
dt = 72
t_export = 1440
t_end = 2*43200. - 0.1*1440

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
options_nh.wetting_and_drying_threshold = 1e-1

# --- create equations ---
solver_obj.create_equations()

# set initial elevation
elev_init = D0*(sqrt(1 - A**2)/(1 - A) - 1 - (x[0]**2 + x[1]**2)*((1 + A)/(1 - A) - 1)/L**2)
solver_obj.assign_initial_conditions(elev=elev_init)

# User-defined output
bath_minus = Function(P1_2d, name="bath_minus").assign(-bathymetry_2d)
File(os.path.join(outputdir, 'minus_bath.pvd')).write(bath_minus)

solver_obj.iterate()
