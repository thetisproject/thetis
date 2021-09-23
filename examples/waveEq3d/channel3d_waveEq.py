"""
Wave equation in 3D
===================

Solves a standing wave in a rectangular basin using wave equation.

Initial condition for elevation corresponds to a standing wave.
Time step and export interval are chosen based on theorethical
oscillation frequency. Initial condition repeats every 20 exports.

This example tests dispersion of surface waves and dissipation of time
integrators, as well as barotropic 2D-3D coupling.
"""
from thetis import *

lx = 44294.46
ly = 3000.0
nx = 25
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 50.0
elev_amp = 1.0
n_layers = 6
# estimate of max advective velocity used to estimate time step
u_mag = Constant(0.5)

outputdir = 'outputs'
print_output('Exporting to ' + outputdir)

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

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.use_nonlinear_equations = False
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'salt_3d']

# need to call creator to create the function spaces
solver_obj.create_equations()
elev_init = Function(solver_obj.function_spaces.H_2d)
x, y = SpatialCoordinate(mesh2d)
elev_init.interpolate(-elev_amp*cos(2*pi*x/lx))
if options.solve_salinity:
    salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
    salt_init3d.assign(4.5)
else:
    salt_init3d = None

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
solver_obj.iterate()
