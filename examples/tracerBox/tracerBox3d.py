# Tracer box in 3D
# ================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# This version uses the ALE moving mesh and a constant tracer to check
# tracer local/global tracer conservation.
# NOTE ALE tracer conservation is currently broken
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
#
# Tuomas Karna 2015-03-11
from thetis import *

mesh2d = Mesh('channel_wave_eq.msh')
depth = 50.0
elev_amp = 1.0
n_layers = 6
# estimate of max advective velocity used to estimate time step
u_mag = Constant(0.5)
sloped = True

suffix = ''
if sloped:
    suffix = '_sloped'
outputdir = 'outputs' + suffix

print_info('Loaded mesh '+mesh2d.name)
print_info('Exporting to '+outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Compute lenght of the domain
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, op=MPI.MAX)
lx = x_max - x_min

if sloped:
    bathymetry_2d.interpolate(Expression('h + 20.0*x[0]/lx', h=depth, lx=lx))

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
t_export = dt
t_end = 10*T_cycle + 1e-3

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)

# warp interior mesh
# coords = solver_obj.mesh.coordinates
# z = coords.dat.data[:, 2].copy()
# x = coords.dat.data[:, 0]
# p = 2.5*x/lx + 0.5
# sigma = -depth * (0.5*np.tanh(p*(-2.0*z/depth - 1.0))/np.tanh(p) + 0.5)
# coords.dat.data[:, 2] = sigma
# print coords.dat.data[:, 2].min(), coords.dat.data[:, 2].max()

options = solver_obj.options
options.nonlin = False
options.mimetic = False
options.solve_salt = True
options.solve_temp = False
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False
options.use_limiter_for_tracers = False  # True
# options.use_imex = True  # NOTE why imex fails with const S?
# options.use_semi_implicit_2d = False
# options.use_mode_split = False
# options.baroclinic = True
# options.h_viscosity = Constant(100.0)
options.tracer_lax_friedrichs = None
options.uv_lax_friedrichs = None
if options.use_mode_split:
    options.dt = dt/5.0
else:
    options.dt = dt/40.0
options.t_export = t_export
options.t_end = t_end
options.u_advection = u_mag
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_salt_conservation = True
options.check_salt_overshoot = True
options.outputdir = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_bottom_2d']

# need to call creator to create the function spaces
solver_obj.create_equations()
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/lx)', eta_amp=elev_amp,
                             lx=lx))
if options.solve_salt:
    salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
    # constant tracer field to test consistency with 3d continuity eq
    salt_init3d.assign(4.5)
    # non-trivial tracer field to test overshoots
    # salt_init3d.project(Expression('4.5*(0.5 + 0.5*sin(2*pi*(x[0])/lx)*cos(pi*x[2]/h/5))', lx=lx, h=depth))
else:
    salt_init3d = None

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
solver_obj.iterate()
