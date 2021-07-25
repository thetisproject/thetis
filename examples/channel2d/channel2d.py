"""
2D shallow water equations in a closed channel
==============================================

Solves shallow water equations in closed rectangular domain
with sloping bathymetry.

Initially water elevation is set to a piecewise linear function
with a slope in the deeper (left) end of the domain. This results
in a wave that develops a shock as it reaches shallower end of the domain.
This example tests the integrity of the 2D mode and stability of momentum
advection.

Setting
solver_obj.nonlin = False
uses linear wave equation instead, and no shock develops.
"""
from thetis import *

# generate mesh
lx = 100e3
ly = 3750
nx = 80
ny = 3
mesh2d = RectangleMesh(nx, ny, lx, ly)

t_end = 6 * 3600.  # total duration in seconds
u_mag = Constant(6.0)  # estimate of max velocity to compute time step
t_export = 100.0  # export interval in seconds

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
# assign bathymetry to a linear function
x, y = SpatialCoordinate(mesh2d)
depth_oce = 20.0
depth_riv = 5.0
bathymetry_2d.interpolate(depth_oce + (depth_riv - depth_oce)*x/lx)

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.swe_timestepper_type = 'SSPRK33'
if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.timestep = 10.0
# set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_height = 6.0
elev_ramp_lx = 30e3
elev_init.interpolate(conditional(x < elev_ramp_lx,
                                  elev_height*(1 - x/elev_ramp_lx),
                                  0.0))
solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()
