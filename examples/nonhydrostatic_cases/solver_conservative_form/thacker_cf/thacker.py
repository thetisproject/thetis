"""
Thacker wetting-drying test case
================================

"""
from thetis import *

mesh2d = Mesh('meshes/04.msh')

outputdir = 'outputs_thacker'
print_output('Exporting to ' + outputdir)

# Model setup
D0 = Constant(50.)
L = Constant(430620.)
eta0 = Constant(2.)
a = ((D0 + eta0)**2 - D0**2)/((D0 + eta0)**2 + D0**2)
A = Constant(a)

# Time steps, total simulation time
dt = 1440
t_export = 1440
t_end = 2*43200. - 0.1*1440

# bathymetry
P1_2d = FunctionSpace(mesh2d, "DG", 1)
bathymetry_2d = Function(P1_2d, name='bathymetry')
x = SpatialCoordinate(mesh2d)
bathy = D0*(1 - (x[0]**2 + x[1]**2)/(L**2))
bathymetry_2d.interpolate(bathy)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
# time stepper
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']
# wetting and drying
options.use_wetting_and_drying = True
options.wetting_and_drying_alpha = Constant(0.5)

# --- create equations ---
solver_obj.create_equations()

# set initial elevation
elev_init = D0*(sqrt(1 - A**2)/(1 - A) - 1 - (x[0]**2 + x[1]**2)*((1 + A)/(1 - A) - 1)/L**2)
solver_obj.assign_initial_conditions(elev=elev_init)

# User-defined output: moving bathymetry and eta_tilde
wd_bathfile = File(os.path.join(outputdir, 'moving_bath.pvd'))
moving_bath = Function(P1_2d, name="moving_bath")
eta_tildefile = File(os.path.join(outputdir, 'eta_tilde.pvd'))
eta_tilde = Function(P1_2d, name="eta_tilde")
minus_bathfile = File(os.path.join(outputdir, 'minus_bath.pvd'))
bath_minus = Function(P1_2d, name="bath_minus")
bath_minus.assign(-bathymetry_2d)
minus_bathfile.write(bath_minus)


# user-specified export function
def export_func():
    wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
    eta = solver_obj.fields.elev_2d
    moving_bath.project(bathymetry_2d + wd_bath_displacement(eta))
    wd_bathfile.write(moving_bath)
    eta_tilde.project(eta + wd_bath_displacement(eta))
    eta_tildefile.write(eta_tilde)


solver_obj.iterate(export_func=export_func)
