# Wave equation in 2D
# ===================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
# This example tests dispersion of surface waves and dissipation of time
# integrators.
#
# Tuomas Karna 2015-03-11
from thetis import *

mesh2d = Mesh('channel_wave_eq.msh')
depth = 50.0
elev_amp = 1.0
# estimate of max advective velocity used to estimate time step
u_mag = Constant(0.5)

outputdir = 'outputs_wave_eq_2d'
print_output('Loaded mesh '+mesh2d.name)
print_output('Exporting to '+outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Compute lenght of the domain
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
comm = x_func.comm
x_min = comm.allreduce(x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, op=MPI.MAX)
lx = x_max - x_min

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
t_export = dt
t_end = 10*T_cycle + 1e-3

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.use_nonlinear_equations = False  # use linear wave equation
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
# options.timestepper_type = 'SSPRK33'
# options.dt = dt/40.0  # for explicit schemes
options.timestepper_type = 'CrankNicolson'
# options.dt = 10.0  # override dt for CrankNicolson (semi-implicit)
# options.timestepper_type = 'SSPIMEX'
options.dt = 10.0  # override dt for IMEX (semi-implicit)

# need to call creator to create the function spaces
solver_obj.create_equations()

# set initial elevation to first standing wave mode
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/lx)', eta_amp=elev_amp,
                             lx=lx))
solver_obj.assign_initial_conditions(elev=elev_init)

# # start from previous time step
# i_exp = 5
# iteration = int(i_exp*t_export/solver_obj.dt)
# time = iteration*solver_obj.dt
# solver_obj.load_state(i_exp, time, iteration)

solver_obj.iterate()
