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
from thetis.tracer_eq_2d import *

# generate mesh
#lx = 100e3
#ly = 3750
#nx = 80
#ny = 3
#mesh2d = RectangleMesh(nx, ny, lx, ly)

mesh2d = Mesh("channel_with_cylinder.msh")


t_end = 300.  # total duration in seconds
u_mag = Constant(10.0)  # estimate of max velocity to compute time step
t_export = 0.1  # export interval in seconds

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1_2dv = VectorFunctionSpace(mesh2d, 'CG', 1)
depth = 0.5
bathymetry_2d = Function(P1_2d, name='Bathymetry')
viscosity_2d = Function(P1_2d, name='viscosity')
# assign bathymetry to a linear function
x, y = SpatialCoordinate(mesh2d)
#depth_oce = 20.0
bathymetry_2d.interpolate(Constant(depth))
viscosity_2d.interpolate(conditional(le(x, 13), 1.0e-6, 1e-1))

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.solve_rans_model = True
options.rans_model_options.l_max = 1.0e2
options.rans_model_options.delta = 1.0e-3
options.rans_model_options.closure_name = 'k-omega'
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.horizontal_viscosity = viscosity_2d
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'rans_tke', 'rans_psi', 'rans_eddy_viscosity',]
options.timestepper_type = 'BackwardEuler'
if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 1.0e-2
# set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_height = 0.0
elev_ramp_lx = 30e3
elev_init.interpolate(conditional(x < elev_ramp_lx,
                                  elev_height*(1 - x/elev_ramp_lx),
                                  0.0))
uv_init = Constant((0.,0))

inlet_e = Constant(elev_height)
inlet_v_full = Constant((1., 0.))

inlet_v = Function(P1_2dv, name='Boundary_velocity')
bnd_time = Constant(0.0)
tri = TrialFunction(P1_2dv)
test = TestFunction(P1_2dv)
t_spin = 1.0
v_in = inlet_v_full*(bnd_time/t_spin)
a = inner(test, tri)*dx
L = inner(test, v_in)*dx
inlet_v_prob = LinearVariationalProblem(a, L, inlet_v)
inlet_v_solver = LinearVariationalSolver(inlet_v_prob)
inlet_v_solver.solve()

outlet_e = Constant(0)
no_flux = Constant(0)

solver_obj.bnd_functions['shallow_water'] = { 1: {'elev': inlet_e, 'uv':inlet_v,'value':Constant(0.0) },
                                              2: {'elev': outlet_e, 'stress': Constant(0.0), 'flux': Constant(0.0)},
                                             3: {'un': no_flux, 'flux': Constant(0.0)},
                                              4: {'un': no_flux, 'flux': Constant(0.0)},
                                              5: {'wall_law': Constant(0.0)}}

solver_obj.bnd_functions['rans_tke'] = { 1: {'value':Constant(0.01), 'uv':inlet_v},
                                       3: {'un': Constant(0.0)},
                                       4: {'un': Constant(0.0)},
                                           5: {'wall_law': Constant(0.0), 'un': Constant(0.0)}}

solver_obj.bnd_functions['rans_psi'] = { 1: {'value':Constant(0.01/1.0e-3), 'uv':inlet_v},
                                       3: {'un': Constant(0.0)},
                                       4: {'un': Constant(0.0)},
                                           5: {'wall_law': Constant(0.0), 'un': Constant(0.0)}}

solver_obj.assign_initial_conditions(elev=elev_init, rans_tke=Constant(0.01), rans_psi=Constant(0.01/1.0e-3))

def update_forcings(t):
    bnd_time.assign(min(t_spin, t))
    inlet_v_solver.solve()

solver_obj.iterate(update_forcings=update_forcings)
