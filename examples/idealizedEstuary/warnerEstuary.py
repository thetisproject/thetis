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
from thetis import *

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
print_output('Exporting to ' + outputdir)
dt = 25.0  # 25.0/refinement[reso_str]  # TODO tune dt
t_end = 20*24*3600
# export every 9 min, day 16 is export 2720
t_export = 9*60.0

depth_ocean = 10
u_tide = 0.4
t_tide = 12*3600
salt_ocean = 30.0
depth_river = 5
u_river = -0.08
salt_river = 0.0
temp_const = 10.0

# bathymetry
p1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(p1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('h_oce - (h_oce - h_riv)*x[0]/lx', h_oce=depth_ocean, h_riv=depth_river, lx=lx))

simple_barotropic = False  # for testing flux boundary conditions

# create solver
solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solverobj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'ssprk22'
options.solve_salt = not simple_barotropic
options.solve_temp = False
options.constant_temp = Constant(temp_const)
options.solve_vert_diffusion = not simple_barotropic
options.use_bottom_friction = not simple_barotropic
options.use_turbulence = not simple_barotropic
options.use_turbulence_advection = not simple_barotropic
options.baroclinic = not simple_barotropic
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
# options.h_diffusivity = Constant(50.0)
# options.h_viscosity = Constant(50.0)
options.v_viscosity = Constant(1.3e-6)  # background value
options.v_diffusivity = Constant(1.4e-7)  # background value
options.use_limiter_for_tracers = True
Re_h = 5.0
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(Re_h))
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = Constant(2.0)
options.check_salt_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'salt_3d', 'smag_visc_3d',
                                 'eddy_visc_3d', 'shear_freq_3d',
                                 'buoy_freq_3d', 'tke_3d', 'psi_3d',
                                 'eps_3d', 'len_3d']
gls_options = options.gls_options
gls_options.apply_defaults('k-omega')
gls_options.stability_name = 'CB'

solverobj.create_function_spaces()

# initial conditions
salt_init3d = Function(solverobj.function_spaces.H, name='initial salinity')
# original vertically uniform initial condition
salt_init3d.interpolate(Expression('s_oce - (s_oce - s_riv)*(x[0] - 30000 + 10*x[2])/50000 ',
                                   s_oce=salt_ocean, s_riv=salt_river))
# start from idealized salt wedge
# salt_init3d.interpolate(Expression('(s_riv + (s_riv - s_oce)*(x[0] - 80000)/50000 * (0.5 - 0.5*tanh(4*(x[2] + 2.0))) )',
#                                    s_oce=salt_ocean, s_riv=salt_river))
min_ix = salt_init3d.dat.data < salt_river
salt_init3d.dat.data[min_ix] = salt_river
max_ix = salt_init3d.dat.data > salt_ocean
salt_init3d.dat.data[max_ix] = salt_ocean

# weak boundary conditions
flux_ocean = -u_tide*depth_ocean*ly
flux_river = u_river*depth_river*ly

t = 0.0
t_ramp = 3600.0  # NOTE use ramp to avoid stading waves
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
solverobj.bnd_functions['shallow_water'] = {1: ocean_funcs, 2: river_funcs}
solverobj.bnd_functions['momentum'] = {1: ocean_funcs_3d, 2: river_funcs_3d}
solverobj.bnd_functions['salt'] = {1: ocean_salt_3d, 2: river_salt_3d}


def update_forcings(t_new):
    ocean_flux.assign(ocean_flux_func(t_new))
    river_flux.assign(river_flux_func(t_new))


solverobj.assign_initial_conditions(salt=salt_init3d)
solverobj.iterate(update_forcings=update_forcings)
