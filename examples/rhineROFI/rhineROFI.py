# Rhine ROFI test case
# ====================
#
# Idealized Rhine river plume test case according to [2].
#
# Salinity ranges from 32 psu in ocean to 0 in river.
# Temperature is constant 10 deg Celcius. This corresponds to density
# values 1024.611 kg/m3 in the ocean and 999.702 in the river.
#
# [1] de Boer, G., Pietrzak, J., and Winterwerp, J. (2006). On the vertical
#     structure of the Rhine region of freshwater influence. Ocean Dynamics,
#     56(3):198-216.
# [2] Fischer, E., Burchard, H., and Hetland, R. (2009). Numerical
#     investigations of the turbulent kinetic energy dissipation rate in the
#     Rhine region of freshwater influence. Ocean Dynamics, 59:629-641.
#
# Tuomas Karna 2015-06-24

from thetis import *

# set physical constants
physical_constants['rho0'].assign(1000.0)
physical_constants['z0_friction'].assign(0.005)

reso = 'fine'
layers = 12
if reso == 'fine':
    layers = 20
outputdir = 'outputs_{:}'.format(reso)
mesh2d = Mesh('mesh_rhineRofi_{:}.msh'.format(reso))
print_output('Loaded mesh ' + mesh2d.name)
print_output('Exporting to ' + outputdir)

# Physical parameters
eta_amplitude = 1.00  # mean (Fisher et al. 2009 tidal range 2.00 )
eta_phase = 0
H_ocean = 20  # water depth
H_river = 5  # water depth at river inlet
L_river = 45e3
Q_river = 3.0e3  # 1.5e3 river discharge (Fisher et al. 2009)
temp_const = 10.0
salt_river = 0.0
salt_ocean = 32.0

Ttide = 44714.0  # M2 tidal period (Fisher et al. 2009)
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaEarth = 2*np.pi/Tday
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H_ocean)  # [m/s] wave speed
lat_deg = 52.5  # latitude
phi = (np.pi/180)*lat_deg  # latitude in radians
coriolis_f = 2*OmegaEarth*sin(phi)  # [rad/s] Coriolis parameter ~ 1.1e-4
kelvin_k = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvin_m = (coriolis_f/c)  # [-] Cross-shore variation

dt = 7.0
t_end = 32*44714
t_export = 900.0  # 44714/12

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('(x[0] > 0.0) ? H*(1-x[0]/L_river) + H_river*(x[0]/L_river) : H',
                                     H=H_ocean, H_river=H_river, L_river=L_river))

simple_barotropic = False  # for debugging

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.solve_salt = not simple_barotropic
options.solve_temp = False
options.constant_temp = Constant(temp_const)
options.solve_vert_diffusion = not simple_barotropic
options.use_bottom_friction = not simple_barotropic
options.use_turbulence = not simple_barotropic
options.use_turbulence_advection = not simple_barotropic
options.use_ale_moving_mesh = False
# options.use_semi_implicit_2d = False
# options.use_mode_split = False
options.baroclinic = not simple_barotropic
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
# options.h_diffusivity = Constant(50.0)
# options.h_viscosity = Constant(50.0)
options.v_viscosity = Constant(1.3e-6)  # background value
options.v_diffusivity = Constant(1.4e-7)  # background value
options.use_limiter_for_tracers = True
Re_h = 5.0
options.smagorinsky_factor = Constant(1.0/np.sqrt(Re_h))
options.coriolis = Constant(coriolis_f)
# if options.use_mode_split:
#     options.dt = dt
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = Constant(2.0)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_salt_conservation = True
options.check_salt_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d', 'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'w_3d', 'salt_3d', 'smag_visc_3d',
                                 'eddy_visc_3d', 'shear_freq_3d',
                                 'buoy_freq_3d', 'tke_3d', 'psi_3d',
                                 'eps_3d', 'len_3d']

bnd_elev = Function(P1_2d, name='Boundary elevation')
bnd_time = Constant(0)
xyz = solver_obj.mesh2d.coordinates
tri = TrialFunction(P1_2d)
test = TestFunction(P1_2d)
elev = eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*elev*dx
bnd_elev_prob = LinearVariationalProblem(a, L, bnd_elev)
bnd_elev_solver = LinearVariationalSolver(bnd_elev_prob)
bnd_elev_solver.solve()

fs = P1_2d
bnd_v = Function(fs, name='Boundary v velocity')
tri = TrialFunction(fs)
test = TestFunction(fs)
v = -(g*kelvin_k/OmegaTide)*eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*v*dx
bnd_v_prob = LinearVariationalProblem(a, L, bnd_v)
bnd_v_solver = LinearVariationalSolver(bnd_v_prob)
bnd_v_solver.solve()

river_discharge = Constant(-Q_river)
ocean_salt = Constant(salt_ocean)
river_salt = Constant(salt_river)
tide_elev_funcs = {'elev': bnd_elev}
tide_uv_funcs = {'un': bnd_v}
open_funcs = {'symm': None}
river_funcs = {'flux': river_discharge}
bnd_ocean_salt = {'value': ocean_salt}
bnd_river_salt = {'value': river_salt}
solver_obj.bnd_functions['shallow_water'] = {1: tide_elev_funcs, 2: tide_elev_funcs,
                                             3: tide_elev_funcs, 6: river_funcs}
solver_obj.bnd_functions['momentum'] = {1: open_funcs, 2: open_funcs,
                                        3: open_funcs, 6: open_funcs}
solver_obj.bnd_functions['salt'] = {1: bnd_ocean_salt, 2: bnd_ocean_salt,
                                    3: bnd_ocean_salt, 6: bnd_river_salt}

solver_obj.create_equations()
bnd_elev_3d = Function(solver_obj.function_spaces.P1, name='Boundary elevation 3d')
cp_bnd_elev_to_3d = ExpandFunctionTo3d(bnd_elev, bnd_elev_3d)
cp_bnd_elev_to_3d.solve()
tide_elev_funcs_3d = {'elev': bnd_elev_3d}

elev_init = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : amp*cos(x[1]*kelvin_k)',
                      amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
elev_init2 = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init2.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : 0.0',
                       amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
uv_init = Function(solver_obj.function_spaces.U_2d, name='initial velocity')
# uv_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : amp*cos(x[1]*kelvin_k)',
#                       amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
tri = TrialFunction(solver_obj.function_spaces.U_2d)
test = TestFunction(solver_obj.function_spaces.U_2d)
a = inner(test, tri)*dx
uv = (g*kelvin_k/OmegaTide)*elev_init2
l = test[1]*uv*dx
solve(a == l, uv_init)
salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
salt_init3d.interpolate(Expression('d_ocean - (d_ocean - d_river)*(1 + tanh((x[0] - xoff)/sigma))/2',
                                   sigma=8000.0, d_ocean=salt_ocean,
                                   d_river=salt_river, xoff=20.0e3))
# salt_init3d.interpolate(Expression('d_ocean - (d_ocean - d_river)*fmin(fmax((x[0] - xoff)/L, 0.0), 1.0)',
#                                    sigma=12000.0, d_ocean=salt_ocean,
#                                    d_river=salt_river, xoff=2.0e3, L=10e3))


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_solver.solve()
    cp_bnd_elev_to_3d.solve()


solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, uv_2d=uv_init)
solver_obj.iterate(update_forcings=update_forcings)

# tests
# 6572744 - omitting solver_obj.eq_momentum.bnd_functions : FAILS
# 6572791 - init salt only no elev/uv init or forcing : works
# 6572869 - added elev_init : works
# 6572873 - added uv_init : works
# 6572878 - added swe bnds : FAILS salt blows up at river boundary after some iterations
# - added salt bnd conditions
