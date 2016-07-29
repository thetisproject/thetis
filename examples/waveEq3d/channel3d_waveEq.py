# Wave equation in 3D
# ===================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
# This example tests dispersion of surface waves and dissipation of time
# integrators, as well as barotropic 2D-3D coupling.
#
# Tuomas Karna 2015-03-11
from thetis import *

nx = 25
ny = 2
lx = 44294.46
ly = 3000.0
mesh2d = RectangleMesh(nx, ny, lx, ly)
depth = 50.0
elev_amp = 1.0
n_layers = 6
# estimate of max advective velocity used to estimate time step
u_mag = Constant(0.5)

outputdir = 'outputs'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
t_export = dt
t_end = 10*T_cycle + 1e-3

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
# options.timestepper_type = 'ssprk33'
options.timestepper_type = 'leapfrog'
# options.timestepper_type = 'imexale'
# options.timestepper_type = 'erkale'
options.nonlin = False
options.solve_salt = False
options.solve_temp = False
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.dt = dt/5.0
options.t_export = t_export
options.t_end = t_end
options.u_advection = u_mag
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_bottom_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                 'w_3d', 'salt_3d']

# need to call creator to create the function spaces
solver_obj.create_equations()
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/lx)', eta_amp=elev_amp,
                             lx=lx))
if options.solve_salt:
    salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
    # salt_init3d.interpolate(Expression('x[0]/1.0e5*10.0+2.0'))
    salt_init3d.assign(4.5)
else:
    salt_init3d = None

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
solver_obj.iterate()
