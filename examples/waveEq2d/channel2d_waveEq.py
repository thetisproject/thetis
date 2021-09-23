"""
Wave equation in 2D
===================

Solves a standing wave in a rectangular basin using wave equation.

Initial condition for elevation corresponds to a standing wave.
Time step and export interval are chosen based on theorethical
oscillation frequency. Initial condition repeats every 20 exports.

This example tests dispersion of surface waves and dissipation of time
integrators.
"""
from thetis import *

lx = 44294.46
ly = 3000.0
nx = 25
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 50.0
elev_amp = 1.0
# estimate of max advective velocity used to estimate time step
u_mag = Constant(0.5)

outputdir = 'outputs_wave_eq_2d'

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
c_wave = float(numpy.sqrt(9.81*depth))
T_cycle = lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
t_export = dt
t_end = 10*T_cycle + 1e-3

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.use_nonlinear_equations = False  # use linear wave equation
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
options.swe_timestepper_type = 'CrankNicolson'
if hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.swe_timestepper_options.use_automatic_timestep = False
    options.timestep = dt/40.0  # for explicit schemes
else:
    options.timestep = 10.0  # override dt for implicit schemes

# need to call creator to create the function spaces
solver_obj.create_equations()

# set initial elevation to first standing wave mode
elev_init = Function(solver_obj.function_spaces.H_2d)
x, y = SpatialCoordinate(mesh2d)
elev_init.interpolate(-elev_amp*cos(2*pi*x/lx))
solver_obj.assign_initial_conditions(elev=elev_init)

# # start from previous time step
# i_exp = 5
# iteration = int(i_exp*t_export/solver_obj.dt)
# time = iteration*solver_obj.dt
# solver_obj.load_state(i_exp, time, iteration)

solver_obj.iterate()
