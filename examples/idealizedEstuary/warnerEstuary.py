# Idealized estuary test case
# ===========================
#
# Tidal flow in a rectangual channel with a density gradient.
# Setup according to [1].
#
# Bathymetry varies between 10 m (ocean boundary) and 5 m (river boundary).
# At the ocean boundary tidal flux is prescribed, while a constant influx is
# used at the river boundary.
# Initial salinity field is a linear ramp from 32 psu (at x=30 km) to 0 psu
# (at x=80 km).
# Temperature is fixed to 10 deg Celcius.
# This corresponds to density 1023.05 kg/m3 in the ocean and 999.70 kg/m3
# in the river.
#
# [1] Warner, J. C., Sherwood, C. R., Arango, H. G., and Signell, R. P.
#     (2005). Performance of four turbulence closure models implemented
#     using a generic length scale method. Ocean Modelling, 8(1-2):81-113.
#
# Tuomas Karna 2016-02-17

from cofs import *

# set physical constants
physical_constants['rho0'].assign(1000.0)
physical_constants['z0_friction'].assign(0.005)

reso_str = 'coarse'
outputdir = 'outputs_' + reso_str
refinement = {'coarse': 1, 'normal': 2}
lx = 100.0e3
ly = 1000.0/refinement[reso_str]
nx = int(round(100*refinement[reso_str]))
ny = 2
layers = int(round(10*refinement[reso_str]))
mesh2d = RectangleMesh(nx, ny, lx, ly)
print_info('Exporting to ' + outputdir)
dt = 25.0/refinement[reso_str]  # TODO tune dt
T = 20*24*3600
# export every 9 min, day 16 is export 2720
t_export = 9*60.0

depth_ocean = 10
u_tide = 0.4
t_tide = 12*3600
salt_ocean = 30.0
rho_ocean = 1023.05
depth_river = 5
u_river = -0.08
salt_river = 0.0
rho_river = 999.70

# bathymetry
p1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(p1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('10.0 - 5.0*x[0]/100.0e3'))

# create solver
solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solverobj.options
options.mimetic = False
options.solve_salt = True
options.solve_vert_diffusion = True
options.use_bottom_friction = True
options.use_turbulence = True
options.use_ale_moving_mesh = False
# options.use_semi_implicit_2d = False
# options.use_mode_split = False
options.baroclinic = True
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
options.use_limiter_for_tracers = True
Re_h = 2.0
options.smagorinsky_factor = Constant(1.0/np.sqrt(Re_h))
if options.use_mode_split:
    options.dt = dt
options.t_export = t_export
options.t_end = T
options.outputdir = outputdir
options.u_advection = Constant(2.0)
options.check_vol_conservation_2d = True
options.check_vol_conservation_2d = True
options.check_salt_conservation = True
options.check_salt_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d', 'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'tke_3d', 'psi_3d', 'eps_3d', 'len_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'salt_3d', ]
options.fields_to_export_numpy = ['uv_2d', 'elev_2d', 'uv_3d',
                                  'w_3d', 'salt_3d', ]
options.timer_labels = []

solverobj.create_function_spaces()

# initial conditions
salt_init3d = Function(solverobj.function_spaces.H, name='initial salinity')
salt_init3d.interpolate(Expression('s_oce - (s_oce - s_riv)*(x[0] - 30000)/50000',
                                   s_oce=rho_ocean, s_riv=rho_river))
min_ix = salt_init3d.dat.data < rho_river
salt_init3d.dat.data[min_ix] = rho_river
max_ix = salt_init3d.dat.data > rho_ocean
salt_init3d.dat.data[max_ix] = rho_ocean

# weak boundary conditions
ly = 500*refinement[reso_str]
flux_ocean = u_tide*depth_ocean*ly
flux_river = u_river*depth_river*ly

t = 0.0
t_ramp = 24*3600.0  # NOTE use ramp to avoid stading waves
ocean_flux_func = lambda t: (flux_ocean*sin(2 * pi * t / t_tide) -
                             flux_river)*min(t/t_ramp, 1.0)
ocean_flux = Constant(ocean_flux_func(t))
river_flux_func = lambda t: flux_river*min(t/t_ramp, 1.0)
river_flux = Constant(river_flux_func(t))

ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}
ocean_funcs_3d = {'symm': None}
river_funcs_3d = {'symm': None}
ocean_salt_3d = {'value': salt_init3d}
river_salt_3d = {'value': salt_init3d}
solverobj.bnd_functions['shallow_water'] = {3: ocean_funcs, 2: river_funcs}
solverobj.bnd_functions['momentum'] = {3: ocean_funcs_3d, 2: river_funcs_3d}
solverobj.bnd_functions['salt'] = {3: ocean_salt_3d, 2: river_salt_3d}


def update_forcings(t_new):
    ocean_flux.assign(ocean_flux_func(t_new))
    river_flux.assign(river_flux_func(t_new))

solverobj.assign_initial_conditions(salt=salt_init3d)
solverobj.iterate(update_forcings=update_forcings)
