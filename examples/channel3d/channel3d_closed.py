# Idealised channel flow in 3D
# ============================
#
# Solves shallow water equations in closed rectangular domain
# with sloping bathymetry.
#
# Initially water elevation is set to a piecewise linear function
# with a slope in the deeper (left) end of the domain. This results
# in a wave that develops a shock as it reaches shallower end of the domain.
# This example tests the integrity of the coupled 2D-3D model and stability
# of momentum advection.
#
# This test is also useful for testing tracer conservation and consistency
# by advecting a constant passive tracer.
#
# Setting
# solver_obj.nonlin = False
# uses linear wave equation instead, and no shock develops.
#
# Tuomas Karna 2015-03-03

from scipy.interpolate import interp1d
from thetis import *

n_layers = 6
outputdir = 'outputs_closed'
mesh2d = Mesh('channel_mesh.msh')
print_info('Loaded mesh '+mesh2d.name)
print_info('Exporting to '+outputdir)
t_end = 48 * 3600
u_mag = Constant(4.2)
t_export = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 7.0
bathymetry_2d.interpolate(Expression('ho - (ho-hr)*x[0]/100e3',
                                     ho=depth_oce, hr=depth_riv))
# bathymetry_2d.interpolate(Expression('ho - (ho-hr)*0.5*(1+tanh((x[0]-50e3)/15e3))',
#                                      ho=depth_oce, hr=depth_riv))

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
# options.nonlin = False
options.mimetic = False
options.solve_salt = True
options.solve_temp = False
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
# options.use_imex = True
# options.use_semi_implicit_2d = False
# options.use_mode_split = False
# options.baroclinic = True
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = u_mag
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_salt_conservation = True
options.check_salt_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_bottom_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'w_mesh_3d', 'salt_3d',
                                 'uv_dav_2d', 'uv_bottom_2d']

# initial conditions, piecewise linear function
elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])


def elevation(x, y, z, x_array, val_array):
    padval = 1e20
    x0 = np.hstack(([-padval], x_array, [padval]))
    vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
    return interp1d(x0, vals0)(x)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
elev_init = Function(P1_2d)
elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0,
                                  elev_x, elev_v)
salt_init3d = Constant(4.5)
solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
solver_obj.iterate()
