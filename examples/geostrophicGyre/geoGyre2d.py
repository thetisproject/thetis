# Geostrophic gyre test case in 2D
# ================================
#
# Stationary gyre test case according to [1].
# Initial condition for elevation is Gaussian bell funcition.
# initial velocity is obtained from analytical solution corresponding to
# geostrophic balance. The model should retain the initial solution
# indefinitely long time.
#
#
#
# Tuomas Karna 2015-04-28

from thetis import *

# set physical constants
physical_constants['z0_friction'].assign(0.0)
g_grav = physical_constants['g_grav']

lx = 1.0e6
nx = 20
mesh2d = RectangleMesh(nx, nx, lx, lx)
nonlin = False
depth = 1000.0
elev_amp = 3.0
t_end = 75*12*2*3600
t_export = 3600*2

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
x, y = SpatialCoordinate(mesh2d)
coriolis_2d = Function(P1_2d)
f0 = Constant(1.0e-4)
coriolis_2d.assign(f0)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.use_nonlinear_equations = False
options.coriolis_frequency = coriolis_2d
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.timestepper_type = 'CrankNicolson'
options.timestep = 20.0
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']

solver_obj.create_equations()
sigma = Constant(160.0e3)
elev_init = Function(solver_obj.function_spaces.H_2d)
x_0 = y_0 = lx/2
elev_expr = elev_amp*exp(-((x-x_0)**2+(y-y_0)**2)/sigma**2)
elev_init.project(elev_expr)

# initial velocity: u = -g/f deta/dy, v = g/f deta/dx
uv_init = Function(solver_obj.function_spaces.U_2d)
uv_init.project(as_vector((g_grav/f0*2*(y-y_0)/sigma**2*elev_expr,
                           -g_grav/f0*2*(x-x_0)/sigma**2*elev_expr)))

solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)

solver_obj.iterate()
