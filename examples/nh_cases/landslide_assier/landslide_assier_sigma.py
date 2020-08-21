# Landslide sediment case --- Assier-Rzadkiewicz
# ===================
# Wei Pan 2019-12-24

from thetis import *
import math

horizontal_domain_is_2d = False
lx = 4.
ly = 0.1
nx = 200
ny = 1
if horizontal_domain_is_2d:
    mesh = RectangleMesh(nx, ny, lx, ly)
    mesh.coordinates.dat.data[:, 0] = mesh.coordinates.dat.data[:, 0] - 1
    x, y = SpatialCoordinate(mesh)
else:
    mesh = IntervalMesh(nx, lx)
    mesh.coordinates.dat.data[:] = mesh.coordinates.dat.data[:] - 1
    x = SpatialCoordinate(mesh)
n_layers = 50
outputdir = 'outputs_landslide_assier_sigma'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
class BathyExpression(Expression):
    def eval(self, value, x):
         if x[0] <= 0.:
             value[:] = 0.1
         elif x[0] >= 1.5:
             value[:] = 1.6
         else:
             value[:] = x[0] + 0.1
bathymetry_2d.interpolate(BathyExpression())

# set time step, export interval and run duration
dt = 0.0001
t_export = 0.1
t_end = 1.

# density and sediment concentration
rho_0 = 1000.
rho_1 = 2650.
sedi_w = 0.
sedi_s = 0.58

# --- create solver ---
solver_obj = nhsolver_sigma.FlowSolver(mesh, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'
options.timestepper_options.use_automatic_timestep = False
# free surface elevation
options.update_free_surface = True
options.solve_separate_elevation_gradient = True
# mesh update
options.use_ale_moving_mesh = False
# tracer
options.use_baroclinic_formulation = True
options.solve_salinity = True
options.solve_temperature = False
# density
options.rho_fluid = rho_0
options.rho_slide = rho_1
# turbulence
options.use_turbulence = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
# viscosity
options.horizontal_viscosity = Constant(1e-5)
options.horizontal_diffusivity = Constant(1e-5)
options.vertical_viscosity = Constant(1e-5)
options.vertical_diffusivity = Constant(1e-5)
# limiter
use_limiter = True
options.use_limiter_for_tracers = use_limiter
options.use_limiter_for_velocity = use_limiter
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d', 'elev_3d', 'salt_3d', 'density_3d']

if True:
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.rho_ref = rho_0
    options.equation_of_state_options.s_ref = 0.0
    options.equation_of_state_options.th_ref = 0.0
    options.equation_of_state_options.alpha = 0.0
    options.equation_of_state_options.beta = rho_1 - rho_0

# need to call creator to create the function spaces
solver_obj.create_equations()

sedi_init3d = Function(solver_obj.function_spaces.H)
if horizontal_domain_is_2d:
    x, y, z = SpatialCoordinate(solver_obj.mesh)
else:
    x, z = SpatialCoordinate(solver_obj.mesh)

sedi_init3d.interpolate(conditional(x >= 0.0, conditional(x <= 0.65, conditional(z <= x/(x+0.1), sedi_s, sedi_w), sedi_w), sedi_w))

solver_obj.assign_initial_conditions(salt=sedi_init3d)
solver_obj.iterate()

